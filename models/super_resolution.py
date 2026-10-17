"""
超分辨率模块 - ESRGAN
用于提升晶圆图像分辨率，增强微小缺陷的检测能力
基于 PyTorch 实现的轻量级 ESRGAN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import cv2


class ResidualDenseBlock(nn.Module):
    """残差密集块"""
    def __init__(self, nf=64, gc=32, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """残差密集块"""
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class ESRGANGenerator(nn.Module):
    """ESRGAN 生成器网络"""
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=10, gc=32, scale=2):
        super().__init__()
        self.scale = scale
        
        # 第一层卷积
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        
        # RRDB 块
        self.RRDB_trunk = nn.Sequential(
            *[RRDB(nf, gc) for _ in range(nb)]
        )
        
        # 第二层卷积
        self.conv_trunk = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        # 上采样
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        # 输出层
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.conv_trunk(self.RRDB_trunk(fea))
        fea = fea + trunk
        
        # 上采样
        if self.scale >= 2:
            fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        if self.scale >= 4:
            fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        return out


class SuperResolution:
    """超分辨率处理类"""
    
    def __init__(self, scale_factor: int = 2, device: str = None, weights_path: str = None):
        """
        初始化超分辨率模型
        
        Args:
            scale_factor: 放大倍数 (2 or 4)
            device: 设备 (cpu/cuda)
            weights_path: 预训练权重路径
        """
        self.scale_factor = scale_factor
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 构建模型
        self.model = ESRGANGenerator(scale=scale_factor).to(self.device)
        self.model.eval()
        
        # 加载权重
        if weights_path and Path(weights_path).exists():
            self.load_weights(weights_path)
        else:
            print(f"[超分辨率] 未加载预训练权重，使用初始化模型 (scale={scale_factor}x)")
            print(f"[超分辨率] 提示: 可使用 scripts/train_sr.py 训练超分模型")
    
    def load_weights(self, weights_path: str):
        """加载模型权重"""
        state_dict = torch.load(weights_path, map_location=self.device)
        # 兼容不同的 key 格式
        if 'model' in state_dict:
            state_dict = state_dict['model']
        self.model.load_state_dict(state_dict, strict=False)
        print(f"[超分辨率] 已加载权重: {weights_path}")
    
    @torch.no_grad()
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        对图像进行超分辨率增强
        
        Args:
            image: 输入图像 (BGR, H, W, C)
            
        Returns:
            超分后的图像 (BGR, H*scale, W*scale, C)
        """
        # BGR -> RGB, HWC -> CHW, 归一化
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)
        
        # 推理
        output = self.model(img_tensor)
        
        # 反归一化, CHW -> HWC, RGB -> BGR
        output = output.squeeze(0).clamp(0, 1).cpu().numpy()
        output = (output * 255.0).round().astype(np.uint8)
        output = np.transpose(output, (1, 2, 0))
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        return output
    
    def enhance_file(self, input_path: str, output_path: str = None) -> np.ndarray:
        """
        从文件读取并增强
        
        Args:
            input_path: 输入图像路径
            output_path: 输出图像路径 (可选)
            
        Returns:
            超分后的图像
        """
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"无法读取图像: {input_path}")
        
        result = self.enhance(img)
        
        if output_path:
            cv2.imwrite(output_path, result)
            print(f"[超分辨率] 已保存: {output_path}")
        
        return result


class BicubicUpsample:
    """双三次插值上采样（作为超分辨率的轻量级替代/对比）"""
    
    def __init__(self, scale_factor: int = 2):
        self.scale_factor = scale_factor
    
    def enhance(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        new_h, new_w = h * self.scale_factor, w * self.scale_factor
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def create_sr_model(scale: int = 2, use_esrgan: bool = True, device: str = None):
    """
    工厂函数：创建超分辨率模型
    
    Args:
        scale: 放大倍数
        use_esrgan: 是否使用 ESRGAN (False 则使用双三次插值)
        device: 设备
    """
    if use_esrgan:
        return SuperResolution(scale_factor=scale, device=device)
    else:
        return BicubicUpsample(scale_factor=scale)


if __name__ == "__main__":
    # 快速测试
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    print("测试超分辨率模块...")
    
    # 创建一个测试图像
    test_img = np.random.randint(100, 200, (100, 100, 3), dtype=np.uint8)
    
    # 测试双三次插值
    bicubic = BicubicUpsample(scale_factor=2)
    result_bicubic = bicubic.enhance(test_img)
    print(f"双三次插值: {test_img.shape} -> {result_bicubic.shape}")
    
    # 测试 ESRGAN (初始化模型)
    esrgan = SuperResolution(scale_factor=2, device='cpu')
    result_esrgan = esrgan.enhance(test_img)
    print(f"ESRGAN: {test_img.shape} -> {result_esrgan.shape}")
    
    print("测试完成！")
