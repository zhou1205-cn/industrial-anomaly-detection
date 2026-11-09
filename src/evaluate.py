from pathlib import Path
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

DATA = Path(r"D:\datasets\mvtec_ad")
SIZE = (256, 256)


def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if img is None else cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA)


def build_model(cat_dir):
    """从正常训练图学出均值图和标准差图"""
    paths = sorted((cat_dir / "train" / "good").glob("*.png"))
    arr = np.stack([load_gray(p) for p in paths]).astype(np.float32)
    return arr.mean(axis=0), arr.std(axis=0)


def anomaly_score(img, mean, std):
    """返回这张图最可疑区域的异常强度"""
    s = np.abs(img - mean) / (std + 1e-6)
    s = cv2.GaussianBlur(s, (5, 5), 0)
    return float(s.max())


# ========== 逐类别评估 ==========
cats = sorted([d for d in DATA.iterdir() if d.is_dir() and not d.name.startswith(".")])

print(f"{'Category':<14}{'AUROC':>9}{'BestF1':>9}{'Thr':>8}")
print("-" * 42)

results = []
for cat_dir in cats:
    name = cat_dir.name
    mean, std = build_model(cat_dir)

    scores, labels = [], []
    for sub in sorted((cat_dir / "test").iterdir()):
        if not sub.is_dir():
            continue
        label = 0 if sub.name == "good" else 1
        for p in sorted(sub.glob("*.png")):
            img = load_gray(p).astype(np.float32)
            scores.append(anomaly_score(img, mean, std))
            labels.append(label)

    scores = np.array(scores)
    labels = np.array(labels)

    # AUROC：衡量"缺陷图分数是否普遍高于正常图"，1.0 为完美
    auc = roc_auc_score(labels, scores)

    # 扫一遍阈值，找 F1 最高的那个
    best_f1, best_thr = 0.0, 0.0
    for thr in np.linspace(scores.min(), scores.max(), 300):
        pred = (scores > thr).astype(int)
        f1 = f1_score(labels, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    results.append((name, auc, best_f1, best_thr))
    print(f"{name:<14}{auc:>9.3f}{best_f1:>9.3f}{best_thr:>8.2f}")

print("-" * 42)
aucs = [r[1] for r in results]
f1s = [r[2] for r in results]
print(f"{'平均':<14}{np.mean(aucs):>9.3f}{np.mean(f1s):>9.3f}")
