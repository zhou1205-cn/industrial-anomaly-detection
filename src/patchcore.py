"""
PatchCore 简化实现
参考: Towards Total Recall in Industrial Anomaly Detection (CVPR 2022)
"""
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

DATA = Path(r"D:\datasets\mvtec_ad")
CATEGORY = "screw"        # 先用最难的这类，看能提升多少
SIZE = 320
CORESET_RATIO = 0.1      # 记忆库采样比例

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# ---------- backbone ----------
net = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).to(device).eval()
feat= {}
net.layer2.register_forward_hook(lambda m, i, o: feat.__setitem__("l2", o))
net.layer3.register_forward_hook(lambda m, i, o: feat.__setitem__("l3", o))

MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


def load_img(path):
    bgr = cv2.imread(str(path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (SIZE, SIZE))
    t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return (t.to(device) - MEAN) / STD


@torch.no_grad()
def extract(t):
    """返回 (1, C, h, w) 的特征图"""
    net(t)
    l2 = feat["l2"]
    l3 = F.interpolate(feat["l3"], size=l2.shape[-2:], mode="bilinear", align_corners=False)
    f = torch.cat([l2, l3], dim=1)
    f = F.avg_pool2d(f, 3, 1, 1)      # 邻域聚合，让特征更稳
    return f


# ---------- 1. 构建记忆库 ----------
train_paths = sorted((DATA / CATEGORY / "train" / "good").glob("*.png"))
chunks = []
for p in train_paths:
    f = extract(load_img(p))
    chunks.append(f[0].permute(1, 2, 0).reshape(-1, f.shape[1]))
bank = torch.cat(chunks, dim=0)
print(f"[1] 原始 patch 数: {bank.shape[0]}")

torch.manual_seed(0)
n_keep = max(1000, int(bank.shape[0] * CORESET_RATIO))
bank = bank[torch.randperm(bank.shape[0])[:n_keep]].contiguous()
print(f"[2] coreset 后: {bank.shape[0]} 个 patch, 每块 {bank.shape[1]} 维")


# ---------- 2. 检测 ----------
@torch.no_grad()
def score_map(path):
    f = extract(load_img(path))[0]                    # C,h,w
    q = f.permute(1, 2, 0).reshape(-1, f.shape[0])    # (h*w, C)
    d = torch.cdist(q, bank)  # (h*w, M) 到每个记忆特征的距离
    d9 = d.topk(9, dim=1, largest=False).values  # 取最近的 9 个
    d = d9.mean(dim=1)  # 用平均，抵抗单个噪声干扰
    # 每个 patch 到记忆库的最近距离
    sm = d.reshape(f.shape[1], f.shape[2])
    sm = F.interpolate(sm[None, None], size=(SIZE, SIZE),
                       mode="bilinear", align_corners=False)[0, 0]
    sm = cv2.GaussianBlur(sm.cpu().numpy(), (7, 7), 0)
    return sm


# ---------- 3. 评估 ----------
scores, labels, samples = [], [], []

for sub in sorted((DATA / CATEGORY / "test").iterdir()):
    if not sub.is_dir():
        continue
    lab = 0 if sub.name == "good" else 1
    for p in sorted(sub.glob("*.png")):
        sm = score_map(p)
        scores.append(float(sm.max()))
        labels.append(lab)
        samples.append((p, sub.name, sm))

auc = roc_auc_score(labels, scores)
print(f"\n[3] {CATEGORY} 的 AUROC = {auc:.3f}   (传统方法基线约 0.467)")

# ---------- 4. 可视化 ----------
show = [s for s in samples if s[1] == "good"][:1] + \
       [s for s in samples if s[1] != "good"][:4]

fig, axes = plt.subplots(len(show), 3, figsize=(10, 3 * len(show)))
for i, (p, label, sm) in enumerate(show):
    raw = cv2.resize(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), (SIZE, SIZE))
    axes[i, 0].imshow(raw)
    axes[i, 0].set_title(f"{label}", fontsize=10)
    axes[i, 0].axis("off")

    axes[i, 1].imshow(sm, cmap="jet")
    axes[i, 1].set_title(f"score (max={sm.max():.2f})", fontsize=10)
    axes[i, 1].axis("off")

    thr = sm.max() * 0.75
    ov = raw.copy()
    ov[sm > thr] = [255, 0, 0]
    axes[i, 2].imshow(ov)
    axes[i, 2].set_title("overlay", fontsize=10)
    axes[i, 2].axis("off")

plt.tight_layout()
plt.savefig(f"patchcore_{CATEGORY}.png", dpi=110, bbox_inches="tight")
print(f"已保存: patchcore_{CATEGORY}.png")
plt.show()

