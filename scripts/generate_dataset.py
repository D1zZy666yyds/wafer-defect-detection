"""
半导体晶圆缺陷数据集生成器 v2（专业版）
========================================
特性：
  - 三类缺陷：划痕(scratch)、颗粒(particle)、缺口(notch)
  - 难度分层：easy(50%) / medium(30%) / hard(20%)，hard 含微小/低对比度缺陷
  - 约 18% 无缺陷 PASS 样本（真实产线分布，可考察模型误报率）
  - 缺陷形态多样化：划痕(直线/折线/弧线)、颗粒(单体/聚簇)、缺口(随机角度/深度)
  - 图像真实性：晶圆基底 + 电路网格 + 渐晕 + 高斯噪声
  - 固定随机种子，完全可复现
  - 自动输出 dataset_info.json 与统计图表

输出格式：YOLO 分割格式 (class x1 y1 x2 y2 ... xn yn)
"""
import json
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

# 配置
CONFIG_PATH = Path(__file__).parent.parent / "wafer_config.yaml"
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

DATA_DIR = Path(__file__).parent.parent / "data"
IMG_SIZE = 640
WAFER_RADIUS = int(IMG_SIZE * 0.42)
CENTER = (IMG_SIZE // 2, IMG_SIZE // 2)
BASE_COLOR = np.array([140, 150, 160])  # 硅片基底灰

DEFECT_CLASSES = {'scratch': 0, 'particle': 1, 'notch': 2}

# 难度分层参数：(下限, 上限)
DIFFICULTY_DIST = [('easy', 0.5), ('medium', 0.3), ('hard', 0.2)]
DIFFICULTY_PARAMS = {
    'easy':   {'scratch_w': (2, 4), 'scratch_len': (100, 250), 'particle_r': (6, 12), 'contrast': (0.55, 1.0)},
    'medium': {'scratch_w': (2, 3), 'scratch_len': (60, 150),  'particle_r': (3, 7),  'contrast': (0.35, 0.7)},
    'hard':   {'scratch_w': (1, 2), 'scratch_len': (40, 90),   'particle_r': (2, 4),  'contrast': (0.15, 0.4)},
}


def defect_color(contrast, rng):
    """根据对比度生成缺陷颜色（暗或亮于基底）"""
    delta = 115 * contrast
    if rng.random() > 0.5:
        return tuple(int(max(0, BASE_COLOR[c] - delta * rng.uniform(0.7, 1.0))) for c in range(3))
    return tuple(int(min(255, BASE_COLOR[c] + delta * rng.uniform(0.7, 1.0))) for c in range(3))


def create_wafer_base(rng):
    """生成晶圆基础图像：基底 + 电路网格 + 渐晕 + 噪声"""
    img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 180
    for c in range(3):
        img[:, :, c] = BASE_COLOR[c]

    # 芯片阵列网格（die grid）
    grid_spacing = rng.randint(24, 40)
    grid_delta = rng.randint(8, 14)
    grid_color = tuple(int(BASE_COLOR[c] - grid_delta) for c in range(3))
    for i in range(0, IMG_SIZE, grid_spacing):
        cv2.line(img, (i, 0), (i, IMG_SIZE), grid_color, 1)
        cv2.line(img, (0, i), (IMG_SIZE, i), grid_color, 1)

    # 高斯噪声
    noise_sigma = rng.uniform(6, 10)
    noise = np.random.normal(0, noise_sigma, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 圆形掩膜
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    cv2.circle(mask, CENTER, WAFER_RADIUS, 255, -1)
    img[mask == 0] = (20, 20, 25)

    # 渐晕（vignette）：晶圆边缘略暗，模拟光照不均
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    dist = np.sqrt((xx - CENTER[0]) ** 2 + (yy - CENTER[1]) ** 2) / WAFER_RADIUS
    vignette = 1.0 - 0.18 * np.clip(dist - 0.55, 0, 1) ** 2
    img = np.clip(img.astype(np.float32) * vignette[:, :, None], 0, 255).astype(np.uint8)
    img[mask == 0] = (20, 20, 25)

    # 边缘高光
    edge_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    cv2.circle(edge_mask, CENTER, WAFER_RADIUS, 255, 3)
    edge_blur = cv2.GaussianBlur(edge_mask, (5, 5), 0)
    img[edge_blur > 50] = np.clip(img[edge_blur > 50] + 30, 0, 255)

    # 晶圆平边（notch 平边，随机方位）
    flat_angle = rng.choice([0, 90, 180, 270])
    flat_width = int(WAFER_RADIUS * 0.15)
    flat_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    r3 = WAFER_RADIUS // 3
    if flat_angle == 0:
        cv2.rectangle(flat_mask, (CENTER[0] + WAFER_RADIUS - flat_width, CENTER[1] - r3),
                      (CENTER[0] + WAFER_RADIUS + 10, CENTER[1] + r3), 255, -1)
    elif flat_angle == 90:
        cv2.rectangle(flat_mask, (CENTER[0] - r3, CENTER[1] + WAFER_RADIUS - flat_width),
                      (CENTER[0] + r3, CENTER[1] + WAFER_RADIUS + 10), 255, -1)
    elif flat_angle == 180:
        cv2.rectangle(flat_mask, (CENTER[0] - WAFER_RADIUS - 10, CENTER[1] - r3),
                      (CENTER[0] - WAFER_RADIUS + flat_width, CENTER[1] + r3), 255, -1)
    else:
        cv2.rectangle(flat_mask, (CENTER[0] - r3, CENTER[1] - WAFER_RADIUS - 10),
                      (CENTER[0] + r3, CENTER[1] - WAFER_RADIUS + flat_width), 255, -1)
    img[flat_mask > 0] = (20, 20, 25)

    return img, mask


def is_inside_wafer(x, y, margin=10):
    dist = np.sqrt((x - CENTER[0]) ** 2 + (y - CENTER[1]) ** 2)
    return dist < WAFER_RADIUS - margin


def _stroke_polygon(pts, half_w):
    """沿中心线 pts 生成走廊多边形（划痕分割标注用）"""
    pts = np.array(pts, dtype=np.float32)
    n = len(pts)
    left, right = [], []
    for i in range(n):
        if i == 0:
            d = pts[1] - pts[0]
        elif i == n - 1:
            d = pts[-1] - pts[-2]
        else:
            d = pts[i + 1] - pts[i - 1]
        norm = np.linalg.norm(d)
        if norm < 1e-6:
            d = np.array([1.0, 0.0])
            norm = 1.0
        tangent = d / norm
        normal = np.array([-tangent[1], tangent[0]])
        left.append(pts[i] + normal * half_w)
        right.append(pts[i] - normal * half_w)
    poly = np.array(left + right[::-1], dtype=np.float32)
    return poly


def _normalize_poly(poly):
    norm = poly.copy()
    norm[:, 0] = np.clip(norm[:, 0] / IMG_SIZE, 0, 1)
    norm[:, 1] = np.clip(norm[:, 1] / IMG_SIZE, 0, 1)
    return norm.flatten().tolist()


def generate_scratch(img, difficulty, rng):
    """划痕：直线 / 折线 / 弧线 三种形态"""
    p = DIFFICULTY_PARAMS[difficulty]
    w_lo, w_hi = p['scratch_w']
    l_lo, l_hi = p['scratch_len']

    # 随机起点
    x1 = y1 = None
    for _ in range(50):
        ang = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(50, WAFER_RADIUS - 30)
        x1 = int(CENTER[0] + dist * np.cos(ang))
        y1 = int(CENTER[1] + dist * np.sin(ang))
        if is_inside_wafer(x1, y1, margin=20):
            break
    else:
        return []

    width = rng.randint(w_lo, w_hi)
    half_w = max(width / 2.0, 1.0)
    color = defect_color(p['contrast'][0] + rng.random() * (p['contrast'][1] - p['contrast'][0]), rng)

    shape = rng.choices(['line', 'polyline', 'arc'], weights=[0.45, 0.35, 0.20])[0]

    if shape == 'line':
        length = rng.randint(l_lo, l_hi)
        ang2 = rng.uniform(0, 2 * np.pi)
        pts = [(x1, y1), (int(x1 + length * np.cos(ang2)), int(y1 + length * np.sin(ang2)))]
    elif shape == 'polyline':
        n_seg = rng.randint(2, 3)
        length = rng.randint(l_lo, l_hi) / n_seg
        cur_ang = rng.uniform(0, 2 * np.pi)
        pts = [(x1, y1)]
        cx, cy = float(x1), float(y1)
        for _ in range(n_seg):
            cx += length * np.cos(cur_ang)
            cy += length * np.sin(cur_ang)
            cur_ang += rng.uniform(-np.pi / 3, np.pi / 3)
            pts.append((int(cx), int(cy)))
    else:  # arc 弧线
        radius = rng.uniform(80, WAFER_RADIUS * 0.8)
        start = rng.uniform(0, 2 * np.pi)
        span = rng.uniform(np.pi / 6, np.pi * 2 / 3)
        n_pts = max(6, int(span * radius / 12))
        pts = [(int(x1 + radius * np.cos(start + span * t / n_pts)),
                int(y1 + radius * np.sin(start + span * t / n_pts)))
               for t in range(n_pts + 1)]

    # 端点裁剪到晶圆内
    pts = [(min(max(px, 0), IMG_SIZE - 1), min(max(py, 0), IMG_SIZE - 1)) for px, py in pts]
    if not any(is_inside_wafer(px, py, margin=5) for px, py in pts):
        return []

    cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, width, lineType=cv2.LINE_AA)

    poly = _stroke_polygon(pts, half_w)
    if len(poly) < 3:
        return []
    return [(DEFECT_CLASSES['scratch'], _normalize_poly(poly))]


def generate_particle(img, difficulty, rng):
    """颗粒：单体 / 聚簇（2-4 颗成片，各自独立标注）"""
    p = DIFFICULTY_PARAMS[difficulty]
    r_lo, r_hi = p['particle_r']
    c_lo, c_hi = p['contrast']

    results = []
    n_cluster = rng.choices([1, 2, 3, 4], weights=[0.6, 0.22, 0.12, 0.06])[0]

    # 聚簇中心
    cx = cy = None
    for _ in range(50):
        x = rng.randint(CENTER[0] - WAFER_RADIUS + 30, CENTER[0] + WAFER_RADIUS - 30)
        y = rng.randint(CENTER[1] - WAFER_RADIUS + 30, CENTER[1] + WAFER_RADIUS - 30)
        if is_inside_wafer(x, y, margin=15):
            cx, cy = x, y
            break
    else:
        return []

    cluster_spread = r_hi * 3
    color = defect_color(c_lo + rng.random() * (c_hi - c_lo), rng)

    for k in range(n_cluster):
        if k == 0:
            x, y = cx, cy
        else:
            x = int(cx + rng.uniform(-cluster_spread, cluster_spread))
            y = int(cy + rng.uniform(-cluster_spread, cluster_spread))
            if not is_inside_wafer(x, y, margin=12):
                continue

        radius = rng.randint(r_lo, r_hi)
        num_points = rng.randint(6, 10)
        poly_pts = []
        for i in range(num_points):
            ang = 2 * np.pi * i / num_points
            r = radius * rng.uniform(0.7, 1.3)
            poly_pts.append([int(x + r * np.cos(ang)), int(y + r * np.sin(ang))])
        poly_np = np.array(poly_pts, dtype=np.int32)
        cv2.fillPoly(img, [poly_np], color)
        # 轻微边缘晕染
        overlay = img.copy()
        cv2.circle(overlay, (x, y), radius + 1, color, -1)
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

        results.append((DEFECT_CLASSES['particle'], _normalize_poly(np.array(poly_pts, dtype=np.float32))))

    return results


def generate_notch(img, difficulty, rng):
    """缺口：晶圆边缘缺失"""
    p = DIFFICULTY_PARAMS[difficulty]
    if difficulty == 'easy':
        depth_lo, depth_hi, width_span = 25, 45, rng.uniform(0.12, 0.20)
    elif difficulty == 'medium':
        depth_lo, depth_hi, width_span = 18, 32, rng.uniform(0.08, 0.15)
    else:
        depth_lo, depth_hi, width_span = 10, 20, rng.uniform(0.05, 0.10)

    angle = rng.uniform(0, 2 * np.pi)
    notch_depth = rng.randint(depth_lo, depth_hi)
    half_angle = width_span / 2

    num_points = rng.randint(5, 8)
    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        curr_angle = angle - half_angle + 2 * t * half_angle
        depth_ratio = np.sin(t * np.pi)
        r = WAFER_RADIUS - notch_depth * depth_ratio
        points.append([int(CENTER[0] + r * np.cos(curr_angle)), int(CENTER[1] + r * np.sin(curr_angle))])

    outer_points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        curr_angle = angle - half_angle + 2 * t * half_angle
        r = WAFER_RADIUS + 20
        outer_points.append([int(CENTER[0] + r * np.cos(curr_angle)), int(CENTER[1] + r * np.sin(curr_angle))])

    notch_poly = np.array(points + outer_points[::-1], dtype=np.int32)
    cv2.fillPoly(img, [notch_poly], (20, 20, 25))

    return [(DEFECT_CLASSES['notch'], _normalize_poly(np.array(points, dtype=np.float32)))]


def generate_sample():
    """生成一张晶圆图像。返回 (img, annotations, difficulty)"""
    difficulty = random.choices([d for d, _ in DIFFICULTY_DIST], weights=[w for _, w in DIFFICULTY_DIST])[0]
    img, _ = create_wafer_base(random)

    annotations = []
    defect_types = ['scratch', 'particle', 'notch']
    defect_weights = [0.38, 0.44, 0.18]

    num_defects = random.choices([1, 2, 3, 4, 5, 6], weights=[0.2, 0.3, 0.25, 0.15, 0.07, 0.03])[0]
    for _ in range(num_defects):
        dtype = random.choices(defect_types, weights=defect_weights)[0]
        if dtype == 'scratch':
            annotations.extend(generate_scratch(img, difficulty, random))
        elif dtype == 'particle':
            annotations.extend(generate_particle(img, difficulty, random))
        else:
            annotations.extend(generate_notch(img, difficulty, random))

    return img, annotations, difficulty


def imwrite_unicode(path, img):
    ext = Path(path).suffix
    success, buf = cv2.imencode(ext, img)
    if success:
        buf.tofile(str(path))
    return success


def generate_dataset(num_train=1200, num_val=240, pass_ratio=0.18, seed=42):
    """生成完整数据集（含 PASS 负样本、难度分层、元数据与统计图）"""
    random.seed(seed)
    np.random.seed(seed)

    metadata = []
    for split, count in [('train', num_train), ('val', num_val)]:
        img_dir = DATA_DIR / 'images' / split
        label_dir = DATA_DIR / 'labels' / split
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        n_pass = int(round(count * pass_ratio))
        pass_idx = set(random.sample(range(count), n_pass))

        print(f"生成 {split} 集: {count} 张（含 PASS 负样本 {n_pass} 张）...")
        for i in tqdm(range(count), desc=split):
            name = f"wafer_{split}_{i:04d}"
            if i in pass_idx:
                img = create_wafer_base(random)[0]
                annotations = []
                difficulty = 'pass'
            else:
                img, annotations, difficulty = generate_sample()

            imwrite_unicode(img_dir / f"{name}.png", img)
            with open(label_dir / f"{name}.txt", 'w') as f:
                for cls_id, seg in annotations:
                    f.write(f"{cls_id} " + " ".join(f"{p:.6f}" for p in seg) + '\n')

            metadata.append({
                'file': f"{name}.png", 'split': split, 'difficulty': difficulty,
                'has_defect': bool(annotations), 'num_instances': len(annotations),
                'classes': [DEFECT_CLASSES_INV[c] for c, _ in annotations],
            })

    # 元数据与统计
    info_path = DATA_DIR / 'dataset_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump({
            'description': '半导体晶圆缺陷分割数据集（合成）',
            'classes': DEFECT_CLASSES, 'img_size': IMG_SIZE, 'seed': seed,
            'pass_ratio': pass_ratio,
            'difficulty_dist': {d: w for d, w in DIFFICULTY_DIST},
            'num_train': num_train, 'num_val': num_val,
            'samples': metadata,
        }, f, ensure_ascii=False, indent=2)

    _print_stats(metadata, num_train, num_val)
    _plot_stats(metadata)


DEFECT_CLASSES_INV = {0: 'scratch', 1: 'particle', 2: 'notch'}


def _print_stats(metadata, num_train, num_val):
    from collections import Counter
    for split in ('train', 'val'):
        sub = [m for m in metadata if m['split'] == split]
        cls_cnt = Counter(c for m in sub for c in m['classes'])
        diff_cnt = Counter(m['difficulty'] for m in sub)
        print(f"\n[{split}] {len(sub)} 张 | PASS 负样本: {diff_cnt.get('pass', 0)} "
              f"({diff_cnt.get('pass', 0)/len(sub)*100:.0f}%)")
        print(f"  实例分布: scratch={cls_cnt.get('scratch',0)} particle={cls_cnt.get('particle',0)} notch={cls_cnt.get('notch',0)}")
        print(f"  难度分布: " + " ".join(f"{k}={v}" for k, v in sorted(diff_cnt.items()) if k != 'pass'))
    print(f"\n元数据已保存: {DATA_DIR / 'dataset_info.json'}")


def _plot_stats(metadata):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    splits = ['train', 'val']
    colors = {'scratch': '#dc2626', 'particle': '#d97706', 'notch': '#db2777'}

    # 类别分布
    ax = axes[0]
    x = np.arange(2)
    bottom = np.zeros(2)
    for cls in ['scratch', 'particle', 'notch']:
        vals = [sum(1 for m in metadata if m['split'] == s and cls in m['classes']) for s in splits]
        ax.bar(x, vals, 0.5, bottom=bottom, label=cls, color=colors[cls])
        bottom += np.array(vals)
    ax.set_xticks(x, ['train', 'val'])
    ax.set_title('Instances per class')
    ax.legend()

    # 难度分布（缺陷图）
    ax = axes[1]
    diffs = ['easy', 'medium', 'hard']
    vals = [[sum(1 for m in metadata if m['split'] == s and m['difficulty'] == d) for s in splits] for d in diffs]
    x = np.arange(2)
    bottom = np.zeros(2)
    for d, v in zip(diffs, vals):
        ax.bar(x, v, 0.5, bottom=bottom, label=d)
        bottom += np.array(v)
    ax.set_xticks(x, ['train', 'val'])
    ax.set_title('Defect images by difficulty')
    ax.legend()

    # PASS 占比
    ax = axes[2]
    vals = [sum(1 for m in metadata if m['split'] == s and m['difficulty'] == 'pass') for s in splits]
    ax.bar(['train', 'val'], vals, 0.5, color='#16a34a')
    ax.set_title('PASS (defect-free) images')

    plt.tight_layout()
    out = DATA_DIR / 'dataset_stats.png'
    plt.savefig(out, dpi=110)
    plt.close()
    print(f"统计图表已保存: {out}")


if __name__ == '__main__':
    generate_dataset(num_train=1200, num_val=240, pass_ratio=0.18, seed=42)
