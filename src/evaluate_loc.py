"""像素级定位精度评估：预测缺陷区域 vs 人工标注的 IoU"""
from pathlib import Path
import csv
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

DATA = Path(r"D:\datasets\mvtec_ad")
SIZE = 320
CORESET_RATIO = 0.1
K_NEIGHBORS = 9
K_SIGMA = 3.0          # 阈值 = 正常均值 + 3×标准差

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
feat = {}
net.layer2.register_forward_hook(lambda m, i, o: feat.__setitem__("l2", o))
net.layer3.register_forward_hook(lambda m, i, o: feat.__setitem__("l3", o))

MEAN_T = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
STD_T = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


def to_tensor(bgr):
    rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (SIZE, SIZE))
    t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.
    return (t.to(device) - MEAN_T) / STD_T


@torch.no_grad()
def extract(t):
    net(t)
    l2 = feat["l2"]
    l3 = F.interpolate(feat["l3"], size=l2.shape[-2:], mode="bilinear", align_corners=False)
    return F.avg_pool2d(torch.cat([l2, l3], dim=1), 3, 1, 1)


@torch.no_grad()
def score_map(bgr, bank):
    f = extract(to_tensor(bgr))[0]
    q = f.permute(1, 2, 0).reshape(-1, f.shape[0])
    d = torch.cdist(q, bank)
    d9 = d.topk(min(K_NEIGHBORS, bank.shape[0]), dim=1, largest=False).values
    s = d9.mean(1).reshape(f.shape[1], f.shape[2])
    s = F.interpolate(s[None, None], (SIZE, SIZE), mode="bilinear", align_corners=False)[0, 0]
    return cv2.GaussianBlur(s.cpu().numpy(), (7, 7), 0)


def largest_component(mask, min_area=30):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = np.zeros_like(mask, dtype=np.uint8)
    if n > 1:
        areas = st[1:, cv2.CC_STAT_AREA]
        if areas.max() >= min_area:
            out = (lab == 1 + int(np.argmax(areas))).astype(np.uint8)
    return out


cats = sorted([d for d in DATA.iterdir() if d.is_dir() and not d.name.startswith(".")])
rows = []

print(f"\n{'Category':<14}{'IoU':>8}{'Recall':>9}{'n':>6}{'Time':>7}")
print("-" * 48)

for cat_dir in cats:
    name = cat_dir.name
    t0 = time.time()
    train_paths = sorted((cat_dir / "train" / "good").glob("*.png"))

    # 建记忆库
    chunks = []
    for p in train_paths:
        f = extract(to_tensor(cv2.imread(str(p))))
        chunks.append(f[0].permute(1, 2, 0).reshape(-1, f.shape[1]).cpu())
    bank = torch.cat(chunks, 0)
    torch.manual_seed(0)
    n = max(500, int(bank.shape[0] * CORESET_RATIO))
    bank = bank[torch.randperm(bank.shape[0])[:n]].contiguous().to(device)

    # 正常品分数分布 → 阈值
    ref = []
    for p in train_paths[:min(30, len(train_paths))]:
        ref.append(float(score_map(cv2.imread(str(p)), bank).max()))
    ref = np.array(ref)
    thr = ref.mean() + K_SIGMA * ref.std()

    # 逐张缺陷图算 IoU
    ious, recalls = [], []
    for sub in sorted((cat_dir / "test").iterdir()):
        if not sub.is_dir() or sub.name == "good":
            continue
        for p in sorted(sub.glob("*.png")):
            sm = score_map(cv2.imread(str(p)), bank)
            pred = largest_component(sm > thr)

            gp = cat_dir / "ground_truth" / sub.name / f"{p.stem}_mask.png"
            if not gp.exists():
                continue
            gt = cv2.resize(cv2.imread(str(gp), cv2.IMREAD_GRAYSCALE),
                            (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 127

            union = (pred.astype(bool) | gt).sum()
            inter = (pred.astype(bool) & gt).sum()
            if gt.sum() > 0:
                ious.append(inter / union if union > 0 else 0.0)
                recalls.append(inter / gt.sum())

    m_iou = float(np.mean(ious)) if ious else 0.0
    m_rec = float(np.mean(recalls)) if recalls else 0.0
    rows.append((name, round(m_iou, 4), round(m_rec, 4), len(ious)))
    print(f"{name:<14}{m_iou:>8.3f}{m_rec:>9.3f}{len(ious):>6}{time.time() - t0:>6.0f}s")
    torch.cuda.empty_cache()

print("-" * 48)
print(f"{'平均':<14}{np.mean([r[1] for r in rows]):>8.3f}{np.mean([r[2] for r in rows]):>9.3f}")

with open("results_localization.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["category", "iou", "recall", "n_images"])
    w.writerows(rows)
print("\n已保存: results_localization.csv")
