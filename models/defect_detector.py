"""
缺陷检测核心模块
封装 YOLOv8-seg 推理，支持超分辨率增强、缺陷分类和良率分析
"""
import cv2
import numpy as np
from pathlib import Path
import yaml
from typing import List, Dict, Tuple, Optional
import time

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

# 缺陷类别名称
DEFECT_NAMES = {
    0: 'scratch',   # 划痕
    1: 'particle',  # 颗粒
    2: 'notch'      # 缺口
}

# 中文名称
DEFECT_NAMES_CN = {
    0: '划痕',
    1: '颗粒',
    2: '缺口'
}

# 缺陷严重程度
DEFECT_SEVERITY = {
    'scratch': 'critical',   # 严重
    'particle': 'minor',     # 轻微
    'notch': 'critical'      # 严重
}


class WaferDefectDetector:
    """晶圆缺陷检测器"""
    
    def __init__(
        self,
        model_path: str = None,
        config_path: str = None,
        use_sr: bool = False,
        sr_scale: int = 2,
        device: str = None,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45
    ):
        """
        初始化缺陷检测器
        
        Args:
            model_path: YOLO 模型权重路径
            config_path: 配置文件路径
            use_sr: 是否启用超分辨率
            sr_scale: 超分辨率放大倍数
            device: 推理设备
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU 阈值
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.use_sr = use_sr
        
        # 加载配置
        if config_path is None:
            config_path = Path(__file__).parent.parent / "wafer_config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 确定设备
        if device is None:
            import torch
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # 加载 YOLO 模型
        self._load_yolo_model(model_path)
        
        # 超分辨率模型 (可选)
        self.sr_model = None
        self.sr_scale = sr_scale
        if use_sr:
            self._init_sr_model()
    
    def _load_yolo_model(self, model_path: str = None):
        """加载 YOLO 模型"""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("请先安装 ultralytics: pip install ultralytics")
        
        if model_path is None:
            # 尝试查找训练好的模型
            possible_paths = [
                Path(__file__).parent.parent / "runs" / "train" / "wafer_defect_seg" / "weights" / "best.pt",
                Path(__file__).parent.parent / "weights" / "best.pt",
                Path(__file__).parent.parent / "weights" / "yolov8s-seg.pt",
            ]
            for p in possible_paths:
                if p.exists():
                    model_path = str(p)
                    break
            
            if model_path is None:
                print("[警告] 未找到训练好的模型，使用预训练 yolov8s-seg.pt")
                print("[提示] 请先运行 scripts/train.py 训练自定义模型")
                model_path = "yolov8s-seg.pt"
        
        print(f"[检测器] 加载模型: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
    
    def _init_sr_model(self):
        """初始化超分辨率模型"""
        from models.super_resolution import create_sr_model
        self.sr_model = create_sr_model(
            scale=self.sr_scale,
            use_esrgan=True,
            device=self.device if 'cpu' in self.device else 'cuda'
        )
        print(f"[检测器] 超分辨率已启用 ({self.sr_scale}x)")
    
    def detect(self, image: np.ndarray) -> Dict:
        """
        检测晶圆缺陷
        
        Args:
            image: 输入图像 (BGR 格式)
            
        Returns:
            检测结果字典
        """
        start_time = time.time()
        
        # 预处理：超分辨率增强
        if self.use_sr and self.sr_model is not None:
            img_sr = self.sr_model.enhance(image)
            inference_img = img_sr
            sr_scale = self.sr_scale
        else:
            inference_img = image
            sr_scale = 1.0
        
        # YOLO 推理
        results = self.model(
            inference_img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # 解析结果
        detections = []
        result = results[0]
        
        if result.masks is not None and result.boxes is not None:
            masks = result.masks.data.cpu().numpy()  # (N, H, W)
            boxes = result.boxes.xyxy.cpu().numpy()   # (N, 4)
            classes = result.boxes.cls.cpu().numpy().astype(int)  # (N,)
            confs = result.boxes.conf.cpu().numpy()   # (N,)
            
            for i in range(len(boxes)):
                # 缩放坐标回原图尺寸
                x1, y1, x2, y2 = boxes[i] / sr_scale
                mask = masks[i]
                
                # 将 mask 调整回原图尺寸
                if sr_scale != 1.0:
                    mask = cv2.resize(
                        mask.astype(np.float32),
                        (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_LINEAR
                    )
                
                # 获取 mask 轮廓点
                mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
                contours, _ = cv2.findContours(
                    mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                # 简化轮廓点
                seg_points = []
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                    seg_points = approx.reshape(-1, 2).tolist()
                
                cls_id = int(classes[i])
                cls_name = DEFECT_NAMES.get(cls_id, f'class_{cls_id}')
                cls_name_cn = DEFECT_NAMES_CN.get(cls_id, f'类别{cls_id}')
                
                detections.append({
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'class_name_cn': cls_name_cn,
                    'confidence': float(confs[i]),
                    'bbox': {
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2)
                    },
                    'segmentation': seg_points,
                    'severity': DEFECT_SEVERITY.get(cls_name, 'minor'),
                    'area': float(cv2.countNonZero(mask_uint8)) if mask_uint8.size > 0 else 0
                })
        
        # 计算统计信息
        defect_counts = {}
        severity_counts = {'critical': 0, 'major': 0, 'minor': 0}
        
        for det in detections:
            name = det['class_name']
            defect_counts[name] = defect_counts.get(name, 0) + 1
            severity = det['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # 良率判断
        is_pass = severity_counts['critical'] == 0
        
        inference_time = (time.time() - start_time) * 1000
        
        return {
            'success': True,
            'detections': detections,
            'total_defects': len(detections),
            'defect_counts': defect_counts,
            'severity_counts': severity_counts,
            'is_pass': is_pass,
            'yield_rate': 100.0 if len(detections) == 0 else max(0, 100 - len(detections) * 5),
            'inference_time_ms': round(inference_time, 2),
            'use_sr': self.use_sr,
            'image_size': {
                'width': image.shape[1],
                'height': image.shape[0]
            }
        }
    
    def detect_file(self, image_path: str) -> Dict:
        """从文件读取并检测"""
        img = cv2.imread(image_path)
        if img is None:
            return {'success': False, 'error': f'无法读取图像: {image_path}'}
        return self.detect(img)
    
    def visualize(self, image: np.ndarray, result: Dict, show_label: bool = True) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image: 原始图像
            result: 检测结果
            show_label: 是否显示标签
            
        Returns:
            可视化图像
        """
        vis_img = image.copy()
        
        # 颜色映射
        colors = {
            0: (0, 0, 255),    # 划痕 - 红色
            1: (255, 165, 0),  # 颗粒 - 橙色
            2: (255, 0, 255)   # 缺口 - 品红
        }
        
        for det in result.get('detections', []):
            cls_id = det['class_id']
            color = colors.get(cls_id, (0, 255, 0))
            
            # 绘制分割区域
            if det['segmentation']:
                pts = np.array(det['segmentation'], dtype=np.int32)
                # 填充半透明
                overlay = vis_img.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.4, vis_img, 0.6, 0, vis_img)
                # 绘制轮廓
                cv2.polylines(vis_img, [pts], True, color, 2)
            
            # 绘制边界框
            bbox = det['bbox']
            x1, y1 = int(bbox['x1']), int(bbox['y1'])
            x2, y2 = int(bbox['x2']), int(bbox['y2'])
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            
            # 标签
            if show_label:
                label = f"{det['class_name_cn']} {det['confidence']:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(vis_img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                cv2.putText(vis_img, label, (x1, y1 - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 绘制结果统计
        h, w = vis_img.shape[:2]
        status = "PASS" if result.get('is_pass', False) else "FAIL"
        status_color = (0, 255, 0) if result.get('is_pass', False) else (0, 0, 255)
        
        cv2.rectangle(vis_img, (10, 10), (200, 90), (0, 0, 0), -1)
        cv2.rectangle(vis_img, (10, 10), (200, 90), status_color, 2)
        cv2.putText(vis_img, f"状态: {status}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(vis_img, f"缺陷数: {result.get('total_defects', 0)}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_img, f"耗时: {result.get('inference_time_ms', 0):.1f}ms", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return vis_img


class YieldAnalyzer:
    """良率分析器 - 批量分析晶圆检测结果"""
    
    def __init__(self):
        self.results = []
    
    def add_result(self, result: Dict):
        """添加一个检测结果"""
        self.results.append(result)
    
    def analyze(self) -> Dict:
        """分析所有结果并返回统计"""
        if not self.results:
            return {'total': 0, 'yield_rate': 0}
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.get('is_pass', False))
        
        # 缺陷统计
        total_defects = sum(r.get('total_defects', 0) for r in self.results)
        
        defect_type_counts = {}
        for r in self.results:
            for name, count in r.get('defect_counts', {}).items():
                defect_type_counts[name] = defect_type_counts.get(name, 0) + count
        
        # 严重程度统计
        severity_counts = {'critical': 0, 'major': 0, 'minor': 0}
        for r in self.results:
            for sev, count in r.get('severity_counts', {}).items():
                severity_counts[sev] = severity_counts.get(sev, 0) + count
        
        # 平均推理时间
        avg_inference = sum(r.get('inference_time_ms', 0) for r in self.results) / total
        
        return {
            'total_wafers': total,
            'passed_wafers': passed,
            'failed_wafers': total - passed,
            'yield_rate': round(passed / total * 100, 2),
            'total_defects': total_defects,
            'avg_defects_per_wafer': round(total_defects / total, 2),
            'defect_type_counts': defect_type_counts,
            'severity_counts': severity_counts,
            'avg_inference_ms': round(avg_inference, 2)
        }
    
    def reset(self):
        """清空结果"""
        self.results = []


if __name__ == "__main__":
    # 快速测试
    print("测试缺陷检测器...")
    
    # 创建测试图像
    test_img = np.ones((640, 640, 3), dtype=np.uint8) * 150
    cv2.circle(test_img, (320, 320), 260, (120, 130, 140), -1)
    
    # 尝试加载检测器
    try:
        detector = WaferDefectDetector(conf_threshold=0.5)
        result = detector.detect(test_img)
        print(f"检测完成: {result['total_defects']} 个缺陷")
        vis_img = detector.visualize(test_img, result)
        print(f"可视化完成: {vis_img.shape}")
    except Exception as e:
        print(f"测试失败: {e}")
        print("请先安装依赖并准备模型")
