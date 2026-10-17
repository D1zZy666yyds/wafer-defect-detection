"""
超分辨率模型训练脚本
使用合成数据训练 ESRGAN 超分辨率模型
"""
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import random

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.super_resolution import ESRGANGenerator


class WaferSRDataset(Dataset):
    """晶圆超分辨率数据集 - 从高分辨率图像生成低分辨率对"""
    
    def __init__(self, image_dir, scale=2, patch_size=128, is_train=True):
        self.image_dir = Path(image_dir)
        self.scale = scale
        self.patch_size = patch_size
        self.is_train = is_train
        
        self.image_files = list(self.image_dir.glob('*.png')) + \
                           list(self.image_dir.glob('*.jpg'))
        print(f"[数据集] 找到 {len(self.image_files)} 张图像")
    
    def __len__(self):
        return len(self.image_files) * 10 if self.is_train else len(self.image_files)
    
    def __getitem__(self, idx):
        img_idx = idx % len(self.image_files)
        img = cv2.imread(str(self.image_files[img_idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w = img.shape[:2]
        
        # 随机裁剪 patch
        if self.is_train:
            x = random.randint(0, max(0, w - self.patch_size))
            y = random.randint(0, max(0, h - self.patch_size))
            hr_patch = img[y:y+self.patch_size, x:x+self.patch_size]
        else:
            # 验证用中心裁剪
            x = (w - self.patch_size) // 2
            y = (h - self.patch_size) // 2
            hr_patch = img[y:y+self.patch_size, x:x+self.patch_size]
        
        # 生成低分辨率图像
        lr_size = self.patch_size // self.scale
        lr_patch = cv2.resize(hr_patch, (lr_size, lr_size), 
                              interpolation=cv2.INTER_CUBIC)
        
        # 数据增强
        if self.is_train:
            # 随机翻转
            if random.random() > 0.5:
                hr_patch = np.fliplr(hr_patch)
                lr_patch = np.fliplr(lr_patch)
            if random.random() > 0.5:
                hr_patch = np.flipud(hr_patch)
                lr_patch = np.flipud(lr_patch)
            
            # 随机旋转 90 度
            k = random.randint(0, 3)
            hr_patch = np.rot90(hr_patch, k)
            lr_patch = np.rot90(lr_patch, k)
        
        # 归一化
        hr_tensor = torch.from_numpy(hr_patch).float().permute(2, 0, 1) / 255.0
        lr_tensor = torch.from_numpy(lr_patch).float().permute(2, 0, 1) / 255.0
        
        return lr_tensor, hr_tensor


def train_sr(
    train_dir: str = None,
    val_dir: str = None,
    scale: int = 2,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 2e-4,
    device: str = None,
    save_dir: str = None
):
    """训练超分辨率模型"""
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    if train_dir is None:
        train_dir = ROOT / "data" / "images" / "train"
    if val_dir is None:
        val_dir = ROOT / "data" / "images" / "val"
    if save_dir is None:
        save_dir = ROOT / "weights"
        save_dir.mkdir(exist_ok=True)
    
    print(f"[超分训练] 设备: {device}")
    print(f"[超分训练] 放大倍数: {scale}x")
    
    # 数据集
    train_dataset = WaferSRDataset(train_dir, scale=scale, patch_size=128, is_train=True)
    val_dataset = WaferSRDataset(val_dir, scale=scale, patch_size=128, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=0)
    
    # 模型
    model = ESRGANGenerator(scale=scale).to(device)
    
    # 损失函数
    criterion = nn.L1Loss()
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_loss = float('inf')
    
    print(f"\n开始训练 ({epochs} epochs)...")
    print("-" * 60)
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for lr_imgs, hr_imgs in pbar:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            optimizer.zero_grad()
            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for lr_imgs, hr_imgs in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)
                sr_imgs = model(lr_imgs)
                loss = criterion(sr_imgs, hr_imgs)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 保存最佳模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_path = save_dir / f"esrgan_x{scale}_best.pth"
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch + 1,
                'loss': best_loss,
                'scale': scale
            }, save_path)
            print(f"  -> 保存最佳模型: {save_path} (loss: {best_loss:.6f})")
        
        # 定期保存
        if (epoch + 1) % 10 == 0:
            save_path = save_dir / f"esrgan_x{scale}_epoch{epoch+1}.pth"
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_val_loss,
                'scale': scale
            }, save_path)
    
    print("\n训练完成！")
    print(f"最佳验证损失: {best_loss:.6f}")
    print(f"模型保存在: {save_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ESRGAN 超分辨率训练')
    parser.add_argument('--scale', type=int, default=2, help='放大倍数')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
    parser.add_argument('--device', type=str, default=None, help='训练设备')
    
    args = parser.parse_args()
    
    train_sr(
        scale=args.scale,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device=args.device
    )
