"""
YOLOv8-seg 晶圆缺陷检测模型训练脚本
使用 ultralytics 框架进行实例分割训练
"""
import os
import sys
from pathlib import Path
import yaml
import argparse

import cv2
import numpy as np

# 项目根目录
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# OpenCV 在 Windows 中文路径下 imread/imwrite 会失败，改用 imdecode/imencode 兼容
def _imread_unicode(filename, flags=cv2.IMREAD_COLOR):
    try:
        return cv2.imdecode(np.fromfile(str(filename), dtype=np.uint8), flags)
    except Exception:
        return cv2.imread(str(filename), flags)


def _imwrite_unicode(filename, img, params=None):
    try:
        ext = '.' + str(filename).rsplit('.', 1)[-1]
        ok, buf = cv2.imencode(ext, img, params if params else [])
        if ok:
            buf.tofile(str(filename))
            return True
        return False
    except Exception:
        return cv2.imwrite(str(filename), img, params if params else None)


cv2.imread = _imread_unicode
cv2.imwrite = _imwrite_unicode


def train_model(
    config_path: str = None,
    epochs: int = 100,
    imgsz: int = 640,
    batch_size: int = 16,
    model_size: str = "n",  # n, s, m, l, x
    device: str = "0",
    resume: bool = False
):
    """
    训练 YOLOv8 分割模型
    
    Args:
        config_path: 数据集配置文件路径
        epochs: 训练轮数
        imgsz: 输入图像尺寸
        batch_size: 批次大小
        model_size: 模型大小 (nano, small, medium, large, xlarge)
        device: 训练设备 (0 for GPU, cpu for CPU)
        resume: 是否继续训练
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("请先安装 ultralytics: pip install ultralytics")
        return
    
    if config_path is None:
        config_path = str(ROOT / "wafer_config.yaml")
    
    # 加载预训练模型
    model_name = f"yolov8{model_size}-seg.pt"
    print(f"加载模型: {model_name}")
    model = YOLO(model_name)
    
    # 训练参数
    train_args = {
        'data': config_path,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch_size,
        'device': device,
        'project': str(ROOT / "runs" / "train"),
        'name': 'wafer_defect_seg',
        'exist_ok': True,
        'patience': 20,
        'save': True,
        'save_period': 10,
        'cache': True,
        'workers': 0,
        'plots': True,
        'val': True,
        'augment': True,
        'mosaic': 0.5,
        'mixup': 0.2,
        'copy_paste': 0.3,
    }
    
    if resume:
        last_weights = ROOT / "runs" / "train" / "wafer_defect_seg" / "weights" / "last.pt"
        if last_weights.exists():
            print(f"继续训练，加载权重: {last_weights}")
            model = YOLO(str(last_weights))
            train_args['resume'] = True
        else:
            print(f"未找到上次训练权重: {last_weights}，从头开始训练")
    
    print(f"\n开始训练...")
    print(f"  数据集: {config_path}")
    print(f"  轮数: {epochs}")
    print(f"  图像尺寸: {imgsz}")
    print(f"  批次大小: {batch_size}")
    print(f"  设备: {device}")
    print()
    
    # 开始训练
    results = model.train(**train_args)
    
    print("\n训练完成！")
    print(f"最佳权重保存在: {ROOT / 'runs' / 'train' / 'wafer_defect_seg' / 'weights' / 'best.pt'}")
    
    # 验证
    print("\n运行验证...")
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    return model


def export_model(model_path: str = None, format: str = "onnx"):
    """导出模型为指定格式"""
    from ultralytics import YOLO
    
    if model_path is None:
        model_path = str(ROOT / "runs" / "train" / "wafer_defect_seg" / "weights" / "best.pt")
    
    if not Path(model_path).exists():
        print(f"模型文件不存在: {model_path}")
        print("请先运行训练脚本")
        return
    
    print(f"导出模型: {model_path} -> {format}")
    model = YOLO(model_path)
    model.export(format=format, imgsz=640)
    print("导出完成！")


def quick_demo():
    """快速演示：生成少量数据并训练少量轮次验证流程"""
    print("=== 晶圆缺陷检测快速演示 ===")
    print()
    
    # 1. 生成小数据集
    print("1. 生成演示数据集 (50张训练 + 10张验证)...")
    from scripts.generate_dataset import generate_dataset
    generate_dataset(num_train=50, num_val=10)
    print()
    
    # 2. 训练
    print("2. 开始训练 (10轮演示)...")
    train_model(
        epochs=10,
        batch_size=8,
        model_size="n",
        device="cpu"  # 演示用CPU
    )
    print()
    
    print("演示完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv8-seg 晶圆缺陷检测训练')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'export', 'demo'],
                       help='运行模式: train/export/demo')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--size', type=str, default='s', 
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='模型大小')
    parser.add_argument('--device', type=str, default='0', help='训练设备')
    parser.add_argument('--resume', action='store_true', help='继续训练')
    parser.add_argument('--format', type=str, default='onnx', help='导出格式')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_model(
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch_size=args.batch,
            model_size=args.size,
            device=args.device,
            resume=args.resume
        )
    elif args.mode == 'export':
        export_model(format=args.format)
    elif args.mode == 'demo':
        quick_demo()
