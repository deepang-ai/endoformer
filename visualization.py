import os
import sys
from datetime import datetime
from typing import Dict

import monai
import pytz
import torch
import yaml
import cv2
from PIL import Image
import numpy as np
import shutil
from accelerate import Accelerator
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from objprint import objstr
from timm.optim import optim_factory
import albumentations as A
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from albumentations.pytorch import ToTensorV2
from src import utils
from src.models import give_model
from src.optimizer import LinearWarmupCosineAnnealingLR
from src.utils import Logger, load_pretrain_model
import warnings
warnings.filterwarnings('ignore')

def wram_up(model: torch.nn.Module, loss_functions: Dict[str, torch.nn.modules.loss._Loss],
                    train_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
                    metrics: Dict[str, monai.metrics.CumulativeIterationMetric],
                    post_trans: monai.transforms.Compose, accelerator: Accelerator, epoch: int, step: int):
    # wram up
    model.train()
    for i, image_batch in enumerate(train_loader):
        logits = model(image_batch[0])
        total_loss = 0
        for name in loss_functions:
            alpth = 1
            loss = loss_functions[name](logits, image_batch[1])
            total_loss += alpth * loss
        val_outputs = post_trans(logits)
        for metric_name in metrics:
            metrics[metric_name](y_pred=val_outputs, y=image_batch[1])

        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad()
        step += 1
        # break
    scheduler.step(epoch)
    metric = {}
    for metric_name in metrics:
        batch_acc = metrics[metric_name].aggregate()[0].to(accelerator.device)
        if accelerator.num_processes > 1:
            batch_acc = accelerator.reduce(batch_acc) / accelerator.num_processes
        metric.update({
        f'Train/mean {metric_name}': float(batch_acc.mean())})
    return step


def get_file_paths(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths


def give_input(image_name):
    augmentations = A.Compose([
            A.Normalize(),
            A.Resize(352, 352, interpolation=cv2.INTER_NEAREST),
            ToTensorV2()
        ])
    or_augmentations = A.Compose([
            A.Normalize(),
            ToTensorV2()
        ])
    image = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
    
    if len(image.shape) == 2:  # 灰度图
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # 包含Alpha通道
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:  # 假设图像已经是BGR格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 将 NumPy 数组转换为 PyTorch 张量
    or_tensor = or_augmentations(image= image_rgb,mask = image_rgb)
    tensor = augmentations(image= image_rgb,mask = image_rgb)
    
    return tensor['image'].unsqueeze(0), (or_tensor['image'].size()[-2],or_tensor['image'].size()[-1])

def get_mask(img_path, output_path, accelerator):  
    if not os.path.exists(output_path):
        os.makedirs(output_path) 
    else:
        shutil.rmtree(output_path)
        os.makedirs(output_path) 
    # 获取路径为path1的文件夹里的所有文件路径
    image_list = get_file_paths(img_path)
    
    for i in range(len(image_list)):
        file_name, file_ext = os.path.splitext(os.path.basename(image_list[i]))
        input, or_size = give_input(image_list[i])
        
        input = input.to(accelerator.device)
        logits = inference(input, model)
        logits = post_trans(logits)

        logits = F.interpolate(logits, size= or_size, mode='nearest')
        mask_array = logits.cpu().squeeze().numpy()*255
        mask_image = Image.fromarray(mask_array).convert('L')
        mask_image.save(output_path + f'/{file_name}{file_ext}')
        accelerator.print(output_path + f'/{file_name}{file_ext}' + ' has been saved!')

def visualization(path1_files, path1, path2, output_path='visualization/output', error=False):
    if not os.path.exists(output_path):
        os.makedirs(output_path) 
    else:
        shutil.rmtree(output_path)
        os.makedirs(output_path)
    path1_files = get_file_paths(path1_files)
    
    for file_path1 in path1_files:
        file_name, file_ext = os.path.splitext(os.path.basename(file_path1))
        file_path_gt = os.path.join(path1, file_name + file_ext)
        file_path2 = os.path.join(path2, file_name + file_ext)
        if os.path.exists(file_path2):
            # 如果存在同名文件，可以在这里执行相应的操作
            print(f"Found matching file: {file_path1} and {file_path_gt} and {file_path2} and get filename:{file_name}{file_ext}")
            img_path = file_path1
            mask_path = file_path2
            gt_path = file_path_gt
            output_p =f'{output_path}/{file_name}.png'
            error_p = f'{output_path}/{file_name}_error.png'
            image = imageio.imread(img_path)
            seg_image=imageio.imread(mask_path)
            gt_image=imageio.imread(gt_path)
            # 使用plt.imshow显示彩色图像

            # 将图像转换为灰度
            image_bgr = cv2.cvtColor(seg_image, cv2.COLOR_RGB2BGR)
            gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            
            gt_image = cv2.cvtColor(gt_image, cv2.COLOR_RGB2BGR)
            g_gray_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY)
            if error == True:
                error_map = np.zeros_like(image, dtype=np.uint8)
                # 找出分割结果为1的区域     
                interest_mask = gray_image == 255

                # 在这些区域内，比较分割结果和ground truth
                matches = (gray_image == g_gray_image) & interest_mask
                differences = ~matches & interest_mask

                # 将匹配的区域标记为绿色，不匹配的区域标记为红色
                # 对于每个颜色通道分别进行操作
                error_map[matches, 1] = 255  # 绿色
                error_map[differences, 0] = 255  # 红色

                # 将Error map叠加到原始图像上
                overlayed_image = cv2.addWeighted(image, 0.7, error_map, 0.3, 0)

                # 保存结果图像
                imageio.imwrite(error_p, overlayed_image)
                print(f'{file_name}{file_ext} error map has been visualizated!')
                
            # 二值化图像
            _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

            # 在原图上画出分割轮廓
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if error == True:
                cv2.drawContours(image, contours, -1, (255, 223, 146), 2)
            else:
                cv2.drawContours(image, contours, -1, (78, 171, 144), 2)
            imageio.imwrite(output_p, image)
            print(f'{file_name}{file_ext} has been visualizated!')
        else:
            print(f'can not find the file: {file_name}{file_ext}')


if __name__ == '__main__':
    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    utils.same_seeds(50)
    logging_dir = os.getcwd() + '/logs/' + config.finetune.checkpoint +str(datetime.now())
    accelerator = Accelerator(cpu=False, log_with=["tensorboard"], logging_dir=logging_dir)
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config))

    accelerator.print('Load Model...')
    model = give_model(config)
    
    img_path = config.visualization.img_path
    gt_output_path = config.visualization.mask_path + '/GT/'
    mask_output_path= config.visualization.mask_path + f"/{config.finetune.model_choose}/"
    gt_visualization_path = config.visualization.visualization_path + '/GT/'
    visualization_path = config.visualization.visualization_path + f"/{config.finetune.model_choose}/"
    
    image_size = config.dataset.CVC_ClinicDB.image_size
    
    accelerator.print('Load Dataloader...')
    if config.trainer.dataset_choose == 'CVC_ClinicDB':
        from src.CVCLoder import get_dataloader
        train_loader, val_loader = get_dataloader(config,dataset_choose='CVC_ClinicDB')
        include_background = False
    elif config.trainer.dataset_choose == 'Kvasir_SEG':
        from src.CVCLoder import get_dataloader
        train_loader, val_loader = get_dataloader(config,dataset_choose='Kvasir_SEG')
        include_background = False
    elif config.trainer.dataset_choose == 'EDD_seg':
        from src.EDDLoader import get_dataloader
        train_loader, val_loader = get_dataloader(config)
        include_background = True

    inference = monai.inferers.SlidingWindowInferer(roi_size=ensure_tuple_rep(image_size, 2), overlap=0.5,
                                                    sw_device=accelerator.device, device=accelerator.device)
    metrics = {
        'dice_metric': monai.metrics.DiceMetric(include_background=include_background,
                                                reduction=monai.utils.MetricReduction.MEAN_BATCH, get_not_nans=True),
        'miou_metric':monai.metrics.MeanIoU(include_background=include_background),
        'f1': monai.metrics.ConfusionMatrixMetric(include_background=include_background, metric_name='f1 score'),
        'precision': monai.metrics.ConfusionMatrixMetric(include_background=include_background, metric_name="precision"),
        'recall': monai.metrics.ConfusionMatrixMetric(include_background=include_background, metric_name="recall"),
        'hd95_metric': monai.metrics.HausdorffDistanceMetric(percentile=95, include_background=include_background, reduction=monai.utils.MetricReduction.MEAN_BATCH, get_not_nans=True)
    }
    post_trans = monai.transforms.Compose([
        monai.transforms.Activations(sigmoid=True), monai.transforms.AsDiscrete(threshold=0.5)
    ])
    
    # 定义训练参数
    optimizer = optim_factory.create_optimizer_v2(model, opt=config.trainer.optimizer,
                                                  weight_decay=config.trainer.weight_decay,
                                                  lr=config.trainer.lr, betas=(0.9, 0.95))
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=config.trainer.warmup,
                                              max_epochs=config.trainer.num_epochs)
    loss_functions = {
        'focal_loss': monai.losses.FocalLoss(to_onehot_y=False),
        'dice_loss': monai.losses.DiceLoss(smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True),
    }
    
    # 加载最优模型
    model = load_pretrain_model(f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/best/pytorch_model.bin", model,
                                accelerator)
    # 加载验证
    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(model, optimizer, scheduler,
                                                                                train_loader, val_loader)
    # warm up
    step = 0
    for epoch in range(0, config.trainer.warmup):
        step = wram_up(model, loss_functions, train_loader,optimizer, scheduler, metrics,post_trans, accelerator, epoch, step)
    
    # ====================================================visualization==========================================================
    # # GT visualization
    # visualization(path1_files=img_path, path1=gt_output_path, path2=gt_output_path, output_path=gt_visualization_path, error=False)
    
    # models' visualization
    get_mask(img_path=img_path, output_path=mask_output_path, accelerator=accelerator)
    visualization(path1_files=img_path, path1=gt_output_path, path2=mask_output_path, output_path=visualization_path, error=True)
    
    
    
    
    