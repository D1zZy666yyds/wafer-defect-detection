"""
FastAPI 后端服务 - 晶圆缺陷检测系统 API
提供图像上传、缺陷检测、结果查询、良率统计等接口
"""
import sys
import os
from pathlib import Path
import uuid
import json
import time
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import cv2
import numpy as np

# 项目根目录
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# 禁止写入 __pycache__（避免对环境目录的写入需求）
sys.dont_write_bytecode = True

# 项目自带的后端依赖目录
_deps = ROOT / ".deps"
if _deps.exists():
    sys.path.insert(0, str(_deps))

from models.defect_detector import WaferDefectDetector, YieldAnalyzer

# 结果存储目录
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
UPLOAD_DIR = RESULTS_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
VISUAL_DIR = RESULTS_DIR / "visualizations"
VISUAL_DIR.mkdir(exist_ok=True)

# 初始化 FastAPI
app = FastAPI(
    title="半导体晶圆缺陷检测系统 API",
    description="基于 YOLOv8-seg + 超分辨率的晶圆缺陷检测系统",
    version="1.0.0"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局检测器实例
detector = None
yield_analyzer = YieldAnalyzer()

# 检测结果缓存
detection_history = []


def get_detector():
    """获取检测器单例"""
    global detector
    if detector is None:
        try:
            detector = WaferDefectDetector(
                use_sr=False,
                conf_threshold=0.35,
                iou_threshold=0.45
            )
            print("[服务器] 检测器初始化成功")
        except Exception as e:
            print(f"[服务器] 检测器初始化失败: {e}")
            raise
    return detector


@app.on_event("startup")
async def startup_event():
    """服务器启动时初始化模型"""
    print("=" * 60)
    print("  半导体晶圆缺陷检测系统 - 后端服务启动中...")
    print("=" * 60)
    try:
        det = get_detector()
        # 预热：执行一次空推理，把 CUDA 上下文初始化耗时移到启动阶段
        import numpy as np
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        t0 = time.time()
        det.detect(dummy)
        print(f"[服务器] GPU 预热完成 ({(time.time()-t0)*1000:.0f} ms)")

        # 预加载超分辨率模型，避免首次开启超分时卡顿
        try:
            det._init_sr_model()
            print("[服务器] 超分辨率模型预加载完成")
        except Exception as e:
            print(f"[服务器] 超分模型预加载失败（不影响默认检测）: {e}")

        print("[服务器] 服务启动完成！")
    except Exception as e:
        print(f"[服务器] 警告: 模型加载失败 ({e})")
        print("[服务器] 部分功能可能不可用，请检查模型文件")


@app.get("/")
async def root():
    """根路径 - 重定向到前端界面"""
    return RedirectResponse(url="/frontend/index.html")


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "detector_ready": detector is not None,
        "timestamp": datetime.now().isoformat()
    }


def filter_by_conf_ranges(detections: list, conf_ranges: dict) -> list:
    """按各类别置信度下限过滤（值为数字表示"≥该值"，兼容 [min, max] 区间）"""
    filtered = []
    for det in detections:
        v = conf_ranges.get(det['class_name'])
        if v is None:
            filtered.append(det)
            continue
        if isinstance(v, (list, tuple)):
            lo, hi = float(v[0]), float(v[1])
            if lo <= det['confidence'] <= hi:
                filtered.append(det)
        else:
            if det['confidence'] >= float(v):
                filtered.append(det)
    return filtered


def recompute_stats(result: dict) -> dict:
    """根据过滤后的检测结果重算统计信息"""
    detections = result['detections']
    defect_counts = {}
    severity_counts = {'critical': 0, 'major': 0, 'minor': 0}
    for d in detections:
        defect_counts[d['class_name']] = defect_counts.get(d['class_name'], 0) + 1
        severity_counts[d['severity']] = severity_counts.get(d['severity'], 0) + 1
    result['defect_counts'] = defect_counts
    result['severity_counts'] = severity_counts
    result['total_defects'] = len(detections)
    result['is_pass'] = severity_counts['critical'] == 0
    result['yield_rate'] = 100.0 if not detections else max(0, 100 - len(detections) * 5)
    return result


@app.post("/api/detect")
async def detect_defects(
    file: UploadFile = File(...),
    use_sr: bool = Form(False),
    conf_threshold: Optional[float] = Form(None),
    conf_ranges: Optional[str] = Form(None)
):
    """
    检测晶圆图像中的缺陷

    - **file**: 晶圆图像文件 (png, jpg, bmp)
    - **use_sr**: 是否启用超分辨率增强
    - **conf_threshold**: 全局置信度阈值 (可选，默认 0.35)
    - **conf_ranges**: 各类别置信度范围 JSON，如 {"scratch":[0.3,0.9],"particle":[0.5,1.0]}
    """
    # 验证文件
    if not file.filename:
        raise HTTPException(status_code=400, detail="未上传文件")
    
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {ext}，支持: {', '.join(allowed_extensions)}"
        )
    
    # 读取图像
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("图像解码失败")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图像读取失败: {str(e)}")
    
    # 获取检测器
    try:
        det = get_detector()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测器未就绪: {str(e)}")
    
    # 临时设置参数
    original_conf = det.conf_threshold
    if conf_threshold is not None:
        det.conf_threshold = conf_threshold
    
    original_sr = det.use_sr
    if use_sr:
        det.use_sr = use_sr
        if det.sr_model is None:
            det._init_sr_model()
    
    try:
        # 执行检测
        result = det.detect(img)

        # 按各类别置信度范围过滤
        if conf_ranges:
            try:
                ranges = json.loads(conf_ranges)
            except Exception:
                raise HTTPException(status_code=400, detail="conf_ranges 必须是合法 JSON")
            if isinstance(ranges, dict) and ranges:
                result['detections'] = filter_by_conf_ranges(result['detections'], ranges)
                result['conf_ranges_applied'] = ranges
                result = recompute_stats(result)

        # 生成可视化图像（基于过滤后的结果）
        vis_img = det.visualize(img, result)
        
        # 保存结果
        task_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存上传图像
        upload_path = UPLOAD_DIR / f"{timestamp}_{task_id}{ext}"
        cv2.imwrite(str(upload_path), img)
        
        # 保存可视化结果
        vis_path = VISUAL_DIR / f"{timestamp}_{task_id}_result{ext}"
        cv2.imwrite(str(vis_path), vis_img)
        
        # 保存 JSON 结果
        json_path = VISUAL_DIR / f"{timestamp}_{task_id}_result.json"
        result_data = {
            'task_id': task_id,
            'timestamp': timestamp,
            'filename': file.filename,
            'result': result
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        # 添加到历史记录
        history_item = {
            'task_id': task_id,
            'timestamp': timestamp,
            'filename': file.filename,
            'total_defects': result['total_defects'],
            'is_pass': result['is_pass'],
            'defect_counts': result['defect_counts'],
            'vis_image': f"/api/results/visual/{vis_path.name}",
            'original_image': f"/api/results/upload/{upload_path.name}"
        }
        detection_history.insert(0, history_item)
        if len(detection_history) > 100:
            detection_history.pop()
        
        # 添加到良率分析器
        yield_analyzer.add_result(result)
        
        # 返回结果
        return {
            'task_id': task_id,
            'success': True,
            'result': result,
            'visualization_url': f"/api/results/visual/{vis_path.name}",
            'original_url': f"/api/results/upload/{upload_path.name}"
        }
        
    finally:
        # 恢复原始参数
        det.conf_threshold = original_conf
        det.use_sr = original_sr


@app.get("/api/results")
async def get_results(limit: int = 20):
    """获取检测历史记录"""
    return {
        'total': len(detection_history),
        'results': detection_history[:limit]
    }


@app.get("/api/results/visual/{filename}")
async def get_visual_result(filename: str):
    """获取可视化结果图像"""
    file_path = VISUAL_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(str(file_path))


@app.get("/api/results/upload/{filename}")
async def get_upload_image(filename: str):
    """获取原始上传图像"""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(str(file_path))


@app.get("/api/stats/yield")
async def get_yield_stats():
    """获取良率统计信息"""
    stats = yield_analyzer.analyze()
    return stats


@app.get("/api/stats/defect-distribution")
async def get_defect_distribution():
    """获取缺陷类型分布统计"""
    stats = yield_analyzer.analyze()
    return {
        'defect_types': stats.get('defect_type_counts', {}),
        'severity': stats.get('severity_counts', {}),
        'total_defects': stats.get('total_defects', 0)
    }


@app.post("/api/stats/reset")
async def reset_stats():
    """重置统计数据"""
    yield_analyzer.reset()
    detection_history.clear()
    return {'success': True, 'message': '统计数据已重置'}


@app.get("/api/config")
async def get_config():
    """获取当前配置"""
    if detector:
        return {
            'conf_threshold': detector.conf_threshold,
            'iou_threshold': detector.iou_threshold,
            'use_sr': detector.use_sr,
            'sr_scale': detector.sr_scale,
            'device': detector.device
        }
    return {'error': '检测器未初始化'}


@app.post("/api/config/update")
async def update_config(
    conf_threshold: Optional[float] = Form(None),
    iou_threshold: Optional[float] = Form(None),
    use_sr: Optional[bool] = Form(None)
):
    """更新检测配置"""
    if not detector:
        raise HTTPException(status_code=500, detail="检测器未初始化")
    
    if conf_threshold is not None:
        detector.conf_threshold = conf_threshold
    if iou_threshold is not None:
        detector.iou_threshold = iou_threshold
    if use_sr is not None:
        detector.use_sr = use_sr
        if use_sr and detector.sr_model is None:
            detector._init_sr_model()
    
    return {
        'success': True,
        'config': {
            'conf_threshold': detector.conf_threshold,
            'iou_threshold': detector.iou_threshold,
            'use_sr': detector.use_sr
        }
    }


def main():
    """启动服务器"""
    print("启动晶圆缺陷检测后端服务...")
    print(f"结果目录: {RESULTS_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
