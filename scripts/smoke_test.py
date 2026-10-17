"""训练模型端到端推理冒烟测试"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import cv2
from models.defect_detector import WaferDefectDetector


def main():
    print("1. 加载训练好的模型...")
    detector = WaferDefectDetector(conf_threshold=0.35)
    print()

    # 取3张验证集图像测试
    val_dir = ROOT / "data" / "images" / "val"
    val_images = sorted(val_dir.glob("*.png"))[:3]

    out_dir = ROOT / "results" / "smoke_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in val_images:
        print(f"2. 检测: {img_path.name}")
        img = cv2.imread(str(img_path))
        result = detector.detect(img)
        vis = detector.visualize(img, result)

        counts = result["defect_counts"]
        status = "PASS" if result["is_pass"] else "FAIL"
        print(f"   状态: {status} | 缺陷总数: {result['total_defects']} "
              f"| 划痕: {counts.get('scratch', 0)} 颗粒: {counts.get('particle', 0)} "
              f"缺口: {counts.get('notch', 0)} | 耗时: {result['inference_time_ms']}ms")

        out_path = out_dir / f"{img_path.stem}_result.png"
        ok = cv2.imwrite(str(out_path), vis)
        print(f"   可视化: {out_path.name} {'保存成功' if ok else '保存失败'}")
        print()

    print("冒烟测试完成！结果保存在:", out_dir)


if __name__ == "__main__":
    main()
