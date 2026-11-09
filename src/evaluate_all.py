"""全类别对比评估：传统差分法 vs PatchCore"""
from pathlib import Path
import time
import csv
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import roc_auc_score

DATA = Path(r"D:\datasets\mvtec_ad")
SIZE = 320
CORESET_RATIO = 0.1
K_NEIGHBORS = 9

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
feat = {}
net.layer2.register_forward_hook(lambda m, i, o: feat.__setitem__("l2", o))
net.layer3.register_forward_hook(lambda m, i, o: feat.__setitem__("l3", o))

MEAN_T = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
STD_T = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


def load_gray(p):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def load_rgb(p):
    bgr = cv2.imread(str(p))
    rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (SIZE, SIZE))
    t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.
    return (t.to(device) - MEAN_T) / STD_T


@torch.no_grad()
def extract(t):
    net(t)
    l2 = feat["l2"]
    l3 = F.interpolate(feat["l3"], size=l2.shape[-2:], mode="bilinear", align_corners=False)
    f = torch.cat([l2, l3], dim=1)
    return F.avg_pool2d(f, 3, 1, 1)


@torch.no_grad()
def pc_score(t, bank):
    f = extract(t)[0]
    q = f.permute(1, 2, 0).reshape(-1, f.shape[0])
    d = torch.cdist(q, bank)
    d9 = d.topk(K_NEIGHBORS, dim=1, largest=False).values
    s = d9.mean(1).reshape(f.shape[1], f.shape[2])
    s = F.interpolate(s[None, None], size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0, 0]
    s = cv2.GaussianBlur(s.cpu().numpy(), (7, 7), 0)
    return float(s.max())


cats = sorted([d for d in DATA.iterdir() if d.is_dir() and not d.name.startswith(".")])

print(f"\n{'Category':<14}{'Trad':>8}{'PatchCore':>11}{'Delta':>8}{'Time':>7}")
print("-" * 50)

rows = []
for cat_dir in cats:
    name = cat_dir.name
    t0 = time.time()
    train_paths = sorted((cat_dir / "train" / "good").glob("*.png"))

    # --- 传统方法：均值 + 标准差 ---
    garr = np.stack([load_gray(p) for p in train_paths]).astype(np.float32)
    mean_g, std_g = garr.mean(0), garr.std(0)

    # --- PatchCore：构建记忆库 ---
    chunks = []
    for p in train_paths:
        f = extract(load_rgb(p))
        chunks.append(f[0].permute(1, 2, 0).reshape(-1, f.shape[1]))
    bank = torch.cat(chunks, 0)
    torch.manual_seed(0)
    n = max(1000, int(bank.shape[0] * CORESET_RATIO))
    bank = bank[torch.randperm(bank.shape[0])[:n]].contiguous()

    labels, s_trad, s_pc = [], [], []
    for sub in sorted((cat_dir / "test").iterdir()):
        if not sub.is_dir():
            continue
        lab = 0 if sub.name == "good" else 1
        for p in sorted(sub.glob("*.png")):
            g = load_gray(p).astype(np.float32)
            sm = cv2.GaussianBlur(np.abs(g - mean_g) / (std_g + 1e-6), (5, 5), 0)
            s_trad.append(float(sm.max()))
            s_pc.append(pc_score(load_rgb(p), bank))
            labels.append(lab)

    auc_t = roc_auc_score(labels, s_trad)
    auc_p = roc_auc_score(labels, s_pc)
    rows.append((name, round(auc_t, 4), round(auc_p, 4)))
    print(f"{name:<14}{auc_t:>8.3f}{auc_p:>11.3f}{auc_p - auc_t:>+8.3f}{time.time() - t0:>6.0f}s")
    torch.cuda.empty_cache()

print("-" * 50)
mt = np.mean([r[1] for r in rows])
mp = np.mean([r[2] for r in rows])
print(f"{'平均':<14}{mt:>8.3f}{mp:>11.3f}{mp - mt:>+8.3f}")

with open("results_compare.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["category", "trad_auroc", "patchcore_auroc"])
    w.writerows(rows)
print("\n已保存: results_compare.csv")
