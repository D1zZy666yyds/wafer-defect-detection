"""
批量检测脚本
批量检测目录中的晶圆图像并生成报告
"""
import sys
from pathlib import Path
import cv2
import json
import csv
from datetime import datetime
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from models.defect_detector import WaferDefectDetector, YieldAnalyzer


def batch_detect(
    input_dir: str,
    output_dir: str = None,
    model_path: str = None,
    use_sr: bool = False,
    conf_threshold: float = 0.35,
    save_visual: bool = True
):
    """
    批量检测晶圆图像
    
    Args:
        input_dir: 输入图像目录
        output_dir: 输出结果目录
        model_path: 模型路径
        use_sr: 是否使用超分辨率
        conf_threshold: 置信度阈值
        save_visual: 是否保存可视化结果
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        return
    
    # 输出目录
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT / "results" / f"batch_{timestamp}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if save_visual:
        vis_dir = output_path / "visualizations"
        vis_dir.mkdir(exist_ok=True)
    
    # 获取图像文件
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"错误: 目录中没有找到图像文件: {input_dir}")
        return
    
    print(f"找到 {len(image_files)} 张图像")
    print(f"输出目录: {output_dir}")
    print()
    
    # 初始化检测器
    print("初始化检测器...")
    detector = WaferDefectDetector(
        model_path=model_path,
        use_sr=use_sr,
        conf_threshold=conf_threshold
    )
    
    analyzer = YieldAnalyzer()
    results = []
    
    # 批量检测
    print("\n开始批量检测...")
    for img_file in tqdm(image_files):
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"  跳过无法读取的文件: {img_file.name}")
            continue
        
        result = detector.detect(img)
        result['filename'] = img_file.name
        results.append(result)
        analyzer.add_result(result)
        
        # 保存可视化
        if save_visual:
            vis_img = detector.visualize(img, result)
            vis_path = vis_dir / f"{img_file.stem}_result{img_file.suffix}"
            cv2.imwrite(str(vis_path), vis_img)
    
    # 生成统计
    stats = analyzer.analyze()
    
    # 保存 JSON 报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'input_dir': str(input_dir),
        'total_images': len(image_files),
        'detected_images': len(results),
        'statistics': stats,
        'details': results
    }
    
    json_path = output_path / "report.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 保存 CSV 报告
    csv_path = output_path / "report.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            '文件名', '是否合格', '缺陷总数', '划痕数', '颗粒数', '缺口数',
            '严重缺陷数', '轻微缺陷数', '推理时间(ms)', '良率估算(%)'
        ])
        
        for r in results:
            counts = r.get('defect_counts', {})
            sev = r.get('severity_counts', {})
            writer.writerow([
                r.get('filename', ''),
                'PASS' if r.get('is_pass', False) else 'FAIL',
                r.get('total_defects', 0),
                counts.get('scratch', 0),
                counts.get('particle', 0),
                counts.get('notch', 0),
                sev.get('critical', 0),
                sev.get('minor', 0),
                r.get('inference_time_ms', 0),
                r.get('yield_rate', 0)
            ])
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("  批量检测完成 - 结果汇总")
    print("=" * 60)
    print(f"  检测晶圆数:     {stats['total_wafers']}")
    print(f"  合格数:         {stats['passed_wafers']}")
    print(f"  不合格数:       {stats['failed_wafers']}")
    print(f"  总体良率:       {stats['yield_rate']:.2f}%")
    print(f"  缺陷总数:       {stats['total_defects']}")
    print(f"  平均缺陷/晶圆:  {stats['avg_defects_per_wafer']}")
    print(f"  平均推理时间:   {stats['avg_inference_ms']:.2f} ms")
    print()
    print("  缺陷类型分布:")
    for name, count in stats['defect_type_counts'].items():
        print(f"    - {name}: {count}")
    print()
    print(f"  报告已保存: {output_dir}")
    print(f"    - JSON: {json_path.name}")
    print(f"    - CSV: {csv_path.name}")
    if save_visual:
        print(f"    - 可视化图像: {vis_dir}")
    print("=" * 60)
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='批量晶圆缺陷检测')
    parser.add_argument('--input', '-i', type=str, required=True, help='输入图像目录')
    parser.add_argument('--output', '-o', type=str, default=None, help='输出目录')
    parser.add_argument('--model', '-m', type=str, default=None, help='模型路径')
    parser.add_argument('--sr', action='store_true', help='启用超分辨率')
    parser.add_argument('--conf', type=float, default=0.35, help='置信度阈值')
    parser.add_argument('--no-vis', action='store_true', help='不保存可视化图像')
    
    args = parser.parse_args()
    
    batch_detect(
        input_dir=args.input,
        output_dir=args.output,
        model_path=args.model,
        use_sr=args.sr,
        conf_threshold=args.conf,
        save_visual=not args.no_vis
    )
