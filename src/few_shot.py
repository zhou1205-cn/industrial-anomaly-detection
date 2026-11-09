"""少样本实验：训练样本数量对 PatchCore 效果的影响"""
from pathlib import Path
import time
import csv
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

DATA = Path(r"D:\datasets\mvtec_ad")
SIZE = 256
CORESET_RATIO = 0.1
K_NEIGHBORS = 9
N_LIST = [5, 10, 20, 50, 999]      # 999 代表"用全部训练图"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| 分辨率:", SIZE, "| 样本档位:", N_LIST)

net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
feat = {}
net.layer2.register_forward_hook(lambda m, i, o: feat.__setitem__("l2", o))
net.layer3.register_forward_hook(lambda m, i, o: feat.__setitem__("l3", o))

MEAN_T = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
STD_T = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


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
def build_bank(paths):
    chunks = []
    for p in paths:
        f = extract(load_rgb(p))
        chunks.append(f[0].permute(1, 2, 0).reshape(-1, f.shape[1]))
    bank = torch.cat(chunks, 0)
    torch.manual_seed(0)
    n = max(500, int(bank.shape[0] * CORESET_RATIO))
    return bank[torch.randperm(bank.shape[0])[:n]].contiguous()


@torch.no_grad()
def score(q, bank):
    d = torch.cdist(q, bank)
    d9 = d.topk(min(K_NEIGHBORS, bank.shape[0]), dim=1, largest=False).values
    return d9.mean(1).max().item()


cats = sorted([d for d in DATA.iterdir() if d.is_dir() and not d.name.startswith(".")])
rows = []

print(f"\n{'Category':<13}" + "".join(f"{('N=' + str(n)) if n != 999 else '全部':>10}" for n in N_LIST))
print("-" * (13 + 10 * len(N_LIST)))

for cat_dir in cats:
    name = cat_dir.name
    train_paths = sorted((cat_dir / "train" / "good").glob("*.png"))

    # 测试集特征只算一次，各个 N 复用（很关键的加速）
    labels, q_cache = [], []
    for sub in sorted((cat_dir / "test").iterdir()):
        if not sub.is_dir():
            continue
        lab = 0 if sub.name == "good" else 1
        for p in sorted(sub.glob("*.png")):
            f = extract(load_rgb(p))[0]
            q_cache.append(f.permute(1, 2, 0).reshape(-1, f.shape[0]).cpu())
            labels.append(lab)

    line = f"{name:<13}"
    for n in N_LIST:
        k = min(n, len(train_paths))
        rng = np.random.RandomState(42)
        idx = sorted(rng.choice(len(train_paths), size=k, replace=False))
        bank = build_bank([train_paths[i] for i in idx])

        scores = [score(q.to(device), bank) for q in q_cache]
        auc = roc_auc_score(labels, scores)
        rows.append((name, n, round(auc, 4)))
        line += f"{auc:>10.3f}"
        torch.cuda.empty_cache()

    print(line)

print("-" * (13 + 10 * len(N_LIST)))
avg = [np.mean([r[2] for r in rows if r[1] == n]) for n in N_LIST]
print(f"{'平均':<13}" + "".join(f"{a:>10.3f}" for a in avg))

# 保存 CSV
with open("results_few_shot.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["category", "n_train", "auroc"])
    w.writerows(rows)


# ---------- 画曲线 ----------
plt.figure(figsize=(9, 5.5))
xs = [str(n) if n != 999 else "ALL" for n in N_LIST]

for name in [r[0] for r in rows[::len(N_LIST)]]:
    ys = [r[2] for r in rows if r[0] == name]
    plt.plot(xs, ys, color="gray", alpha=0.25, linewidth=1)

plt.plot(xs, avg, marker="o", color="#C0392B", linewidth=2.5, label="平均 AUROC")
for x, a in zip(xs, avg):
    plt.annotate(f"{a:.3f}", (x, a), textcoords="offset points",
                 xytext=(0, 9), ha="center", fontsize=9)

plt.xlabel("训练样本数量（只含正常品）")
plt.ylabel("AUROC")
plt.title("样本数量对 PatchCore 检测效果的影响（MVTec AD, 15 类）")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("few_shot_curve.png", dpi=130)
print("\n已保存: results_few_shot.csv 和 few_shot_curve.png")
plt.show()
