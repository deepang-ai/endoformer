"""EndoFormer: a hybrid Transformer for endoscopic polyp segmentation.

Reference implementation of

    EndoFormer: A Hybrid Transformer-Based Approach for Enhanced Polyp
    Segmentation in Endoscopic Imaging
"""

from __future__ import annotations

import math
import os
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

__all__ = [
    "EndoFormer",
    "PVTv2Backbone",
    "pvt_v2_b2",
    "DualReparameterizationConvolution",
    "TextureReparameterizationBranch",
    "SpatialReparameterizationBranch",
    "GlobalLocalAttention",
    "SpatialReconstruction",
    "ActivationFunctionSelector",
    "SegmentationHead",
]


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------
class SwishFunction(torch.autograd.Function):
    """Memory-efficient Swish: ``x * sigmoid(x)``."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        sigmoid_x = torch.sigmoid(x)
        return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))


class Swish(nn.Module):
    def forward(self, x):
        return SwishFunction.apply(x)


class ActivationFunctionSelector(nn.Module):
    """AFS: GELU on the shallow level, Swish on the deeper ones.

    The assignment is fixed by level, not learned.
    """

    GELU = "gelu"
    SWISH = "swish"

    def __init__(self, shallow: bool = True):
        super().__init__()
        self.shallow = shallow
        self.act = nn.GELU() if shallow else Swish()

    @property
    def name(self) -> str:
        return self.GELU if self.shallow else self.SWISH

    def forward(self, x):
        return self.act(x)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class PointwiseMLP(nn.Module):
    """Two pointwise convolutions around one activation."""

    def __init__(self, dim, mlp_ratio, act):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, kernel_size=1, bias=False)
        self.act = act
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, kernel_size=1, bias=False)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class ConvNormAct(nn.Module):
    """Conv2d -> BatchNorm2d -> activation."""

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, act=nn.ReLU(inplace=True)):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = act

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
# DRC module
# ---------------------------------------------------------------------------
class TextureReparameterizationBranch(nn.Module):
    """DRC texture branch over the first half of the split channels."""

    def __init__(self, in_dim=1, out_dim=32, kernel_size=3, mlp_ratio=4, act=nn.GELU()):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.norm = nn.GroupNorm(out_dim // 2, out_dim)
        self.act = act
        self.base_conv = nn.Conv2d(out_dim, out_dim, kernel_size, 1,
                                   (kernel_size - 1) // 2, 1, out_dim, bias=False)
        self.base_norm = nn.BatchNorm2d(out_dim)
        self.add_conv = nn.Conv2d(out_dim, out_dim, 1, 1, 0, 1, out_dim, bias=False)
        self.add_norm = nn.BatchNorm2d(out_dim)
        self.mlp = PointwiseMLP(out_dim, mlp_ratio, act=self.act)

    def forward(self, x):
        x = self.act(self.norm(self.conv(x)))
        identity = x
        x = self.base_norm(self.base_conv(x)) + self.add_norm(self.add_conv(x)) + x
        return self.mlp(x) + identity


class SpatialReparameterizationBranch(nn.Module):
    """DRC spatial branch over the second half of the split channels."""

    def __init__(self, in_dim=32, out_dim=32):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.pointwise_conv_0 = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        self.depthwise_conv = nn.Conv2d(in_dim, in_dim, padding=1, kernel_size=3,
                                        groups=in_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(in_dim)
        self.pointwise_conv_1 = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.bn1(x)
        x = self.pointwise_conv_0(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        return self.pointwise_conv_1(x)


class DualReparameterizationConvolution(nn.Module):
    """DRC: refine a feature map with two branches over a fixed channel split.

    ``in_dim`` is split evenly between the texture and spatial branches, so it
    must be even.
    """

    def __init__(self, in_dim=3, out_dim=32, kernel_size=3, mlp_ratio=4, shallow=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.afs = ActivationFunctionSelector(shallow=shallow)
        self.act = self.afs.act
        self.texture_branch = TextureReparameterizationBranch(
            in_dim=in_dim // 2, out_dim=out_dim, kernel_size=kernel_size,
            mlp_ratio=mlp_ratio, act=self.act)
        self.spatial_branch = SpatialReparameterizationBranch(
            in_dim=in_dim // 2, out_dim=out_dim)
        self.projection = ConvNormAct(out_dim * 2, out_dim, 1, act=self.act)

    @property
    def activation_name(self) -> str:
        return self.afs.name

    def forward(self, x):
        fm_sp1, fm_sp2 = x.chunk(2, dim=1)
        fm_sp1 = self.texture_branch(fm_sp1)
        fm_sp2 = self.spatial_branch(fm_sp2)
        return self.projection(torch.cat([fm_sp1, fm_sp2], dim=1))


# ---------------------------------------------------------------------------
# GLA module
# ---------------------------------------------------------------------------
class GlobalLocalAttention(nn.Module):
    """GLA: multi-head attention over tokens sampled at stride ``r``."""

    def __init__(self, channels, r, heads):
        super().__init__()
        self.head_dim = channels // heads
        self.num_heads = heads
        self.sparse_sampler = nn.AvgPool2d(kernel_size=1, stride=r)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.sparse_sampler(x)
        B, C, H, W = x.shape
        q, k, v = self.qkv(x).view(B, self.num_heads, -1, H * W).split(
            [self.head_dim, self.head_dim, self.head_dim], dim=2)
        attn = (q.transpose(-2, -1) @ k).softmax(-1)
        return (v @ attn.transpose(-2, -1)).view(B, -1, H, W)


class SpatialReconstruction(nn.Module):
    """Upsample by ``r`` to undo the GLA sampling."""

    def __init__(self, in_channels, out_channels, r):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        self.conv_trans = nn.ConvTranspose2d(in_channels, in_channels,
                                             kernel_size=r, stride=r, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv_trans(x)
        x = self.norm(x)
        return self.pointwise_conv(x)


# ---------------------------------------------------------------------------
# PVTv2 backbone
# ---------------------------------------------------------------------------
class OverlapPatchEmbed(nn.Module):
    """Image to overlapping patch embedding."""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, H, W


class DWConv(nn.Module):
    """Depthwise convolution inside the PVTv2 MLP."""

    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class PVTv2Mlp(nn.Module):
    """PVTv2 MLP: linear, depthwise convolution, linear."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class SpatialReductionAttention(nn.Module):
    """PVTv2 attention with spatially reduced keys and values."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class PVTv2Block(nn.Module):
    """Transformer block with depthwise-convolution MLP."""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SpatialReductionAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = PVTv2Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                            act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class PVTv2Backbone(nn.Module):
    """PVTv2 encoder returning the four feature maps ``fm1..fm4``."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dims=(64, 128, 256, 512), num_heads=(1, 2, 4, 8),
                 mlp_ratios=(4, 4, 4, 4), qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=(3, 4, 6, 3), sr_ratios=(8, 4, 2, 1)):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4,
                                              in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2,
                                              in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2,
                                              in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2,
                                              in_chans=embed_dims[2], embed_dim=embed_dims[3])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList([
            PVTv2Block(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                       qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                       attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                       norm_layer=norm_layer, sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        cur += depths[0]
        self.block2 = nn.ModuleList([
            PVTv2Block(dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                       qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                       attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                       norm_layer=norm_layer, sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
        cur += depths[1]
        self.block3 = nn.ModuleList([
            PVTv2Block(dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                       qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                       attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                       norm_layer=norm_layer, sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])
        cur += depths[2]
        self.block4 = nn.ModuleList([
            PVTv2Block(dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                       qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                       attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                       norm_layer=norm_layer, sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B = x.shape[0]
        outs = []
        for patch_embed, blocks, norm in (
            (self.patch_embed1, self.block1, self.norm1),
            (self.patch_embed2, self.block2, self.norm2),
            (self.patch_embed3, self.block3, self.norm3),
            (self.patch_embed4, self.block4, self.norm4),
        ):
            x, H, W = patch_embed(x)
            for blk in blocks:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs


class pvt_v2_b2(PVTv2Backbone):
    """PVTv2-B2 encoder, used for all reported results."""

    def __init__(self, in_chans=3, embed_dims=(64, 128, 320, 512), **kwargs):
        super().__init__(
            in_chans=in_chans, patch_size=4, embed_dims=embed_dims,
            num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)


# ---------------------------------------------------------------------------
# Pretrained weight loading
# ---------------------------------------------------------------------------
def _unwrap_state_dict(checkpoint):
    """Return the tensor state dict from common checkpoint wrappers."""
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a state-dict checkpoint, got {type(checkpoint).__name__}")
    for key in ("state_dict", "model_state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            return candidate
    return checkpoint


def _strip_wrapper_prefixes(key):
    for prefix in ("module.", "model.", "backbone."):
        while key.startswith(prefix):
            key = key[len(prefix):]
    return key


def load_backbone_weights(backbone, path):
    """Load ImageNet weights into ``backbone``, matching by name and shape."""
    path = os.path.expanduser(path) if path else ""
    report = {"path": path, "status": "disabled", "matched": 0,
              "parameter_coverage": 0.0}
    if not path:
        print("Backbone pretrained: disabled (random initialization)")
        return report
    if not os.path.isfile(path):
        report["status"] = "missing"
        print(f"Backbone pretrained: not found at {path}")
        return report

    checkpoint = _unwrap_state_dict(torch.load(path, map_location="cpu"))
    model_state = backbone.state_dict()
    compatible = {}
    for raw_key, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue
        key = _strip_wrapper_prefixes(raw_key)
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape):
            compatible[key] = value

    backbone.load_state_dict(compatible, strict=False)
    matched_numel = sum(model_state[k].numel() for k in compatible)
    total_numel = sum(v.numel() for v in model_state.values())
    report.update({
        "status": "loaded",
        "matched": len(compatible),
        "parameter_coverage": matched_numel / max(total_numel, 1),
    })
    print(f"Backbone pretrained: {len(compatible)}/{len(model_state)} tensors, "
          f"{report['parameter_coverage']:.1%} of backbone parameters")
    return report


# ---------------------------------------------------------------------------
# Segmentation head
# ---------------------------------------------------------------------------
class SegmentationHead(nn.Sequential):
    """SegHead: project fused features to ``out_channels`` logits."""

    def __init__(self, in_channels, out_channels):
        super().__init__(
            ConvNormAct(in_channels * 2, in_channels, 1, 1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )


# ---------------------------------------------------------------------------
# EndoFormer
# ---------------------------------------------------------------------------
class EndoFormer(nn.Module):
    """PVTv2-B2 encoder with DRC refinement and GLA fusion.

    ``out_channels`` follows the label space of the dataset (``1`` for binary
    masks, ``5`` for EDD 2020). ``model_dir`` is the ImageNet PVTv2-B2
    checkpoint; an empty string disables loading.
    """

    def __init__(self, in_channels=3, out_channels=1, dims=(64, 128, 320, 512),
                 out_dim=32, kernel_size=3, mlp_ratio=4,
                 model_dir="./pretrained/pvt_v2_b2.pth",
                 gla_ratio=4, gla_heads=2):
        super().__init__()
        self.backbone_variant = "b2"
        self.backbone = pvt_v2_b2(in_chans=in_channels, embed_dims=dims)
        self.pretrained_load_report = load_backbone_weights(self.backbone, model_dir)

        c1, c2, c3, c4 = dims

        # DRC over the deeper levels fm2..fm4, GELU on fm2 and Swish below.
        self.drc2 = DualReparameterizationConvolution(
            in_dim=c2, out_dim=out_dim, kernel_size=kernel_size,
            mlp_ratio=mlp_ratio, shallow=True)
        self.drc3 = DualReparameterizationConvolution(
            in_dim=c3, out_dim=out_dim, kernel_size=kernel_size,
            mlp_ratio=mlp_ratio, shallow=False)
        self.drc4 = DualReparameterizationConvolution(
            in_dim=c4, out_dim=out_dim, kernel_size=kernel_size,
            mlp_ratio=mlp_ratio, shallow=False)

        # fm34 path: pointwise fusion of the two deep levels, then SegHead.
        self.seg_head = SegmentationHead(out_dim, out_channels)
        # fm12 path: shallow projection, then GLA and spatial reconstruction.
        self.shallow_proj = ConvNormAct(c1, out_dim, 3, 1, 1)
        self.fm2_proj = ConvNormAct(out_dim, out_dim, 1)
        self.gla = GlobalLocalAttention(out_dim * 2, r=gla_ratio, heads=gla_heads)
        self.reconstruction = SpatialReconstruction(
            in_channels=out_dim * 2, out_channels=out_channels, r=gla_ratio)

    @staticmethod
    def _upsample(x, size):
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, x):
        fm1, fm2, fm3, fm4 = self.backbone(x)

        # Deep path: fm4 and fm3 -> pointwise fusion -> SegHead.
        f4 = self._upsample(self.drc4(fm4), fm3.size()[2:])
        f3 = self.drc3(fm3)
        fm34 = self.seg_head(torch.cat([self._upsample(f4, fm2.size()[2:]),
                                        self._upsample(f3, fm2.size()[2:])], dim=1))

        # Shallow path: fm1 texture with refined fm2 -> GLA -> reconstruction.
        shallow = self.shallow_proj(fm1)
        deep = self._upsample(self.fm2_proj(self.drc2(fm2)), shallow.size()[2:])
        fm12 = self.reconstruction(self.gla(torch.cat((deep, shallow), dim=1)))

        return (F.interpolate(fm34, scale_factor=8, mode="bilinear")
                + F.interpolate(fm12, scale_factor=4, mode="bilinear"))


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = EndoFormer(in_channels=3, out_channels=2, model_dir="").to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n / 1e6:.2f} M")
    print("output:", tuple(model(torch.randn(1, 3, 352, 352, device=device)).shape))
