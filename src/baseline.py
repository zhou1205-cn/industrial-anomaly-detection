from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

DATA = Path(r"D:\datasets\mvtec_ad")
CATEGORY = "bottle"          # 先用瓶子试
SIZE = (256, 256)            # 统一尺寸 (宽, 高)
K = 4.0                      # 阈值：超过平均值的 4 倍标准差才算异常


def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if img is None else cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA)


# ========== 1. 从正常图里学出「正常长什么样」 ==========
train_paths = sorted((DATA / CATEGORY / "train" / "good").glob("*.png"))
train_arr = np.stack([load_gray(p) for p in train_paths]).astype(np.float32)

mean_img = train_arr.mean(axis=0)      # 平均图：正常品的样子
std_img = train_arr.std(axis=0)        # 标准差图：每个像素允许波动的范围

print(f"[1] 学到 {len(train_paths)} 张正常图")
print(f"    平均图范围: {mean_img.min():.1f} ~ {mean_img.max():.1f}")
print(f"    平均波动(std)均值: {std_img.mean():.2f}")

cv2.imwrite(f"mean_{CATEGORY}.png", mean_img.astype(np.uint8))
cv2.imwrite(f"std_{CATEGORY}.png", (std_img * 4).clip(0, 255).astype(np.uint8))


# ========== 2. 检测函数 ==========
def detect(path):
    img = load_gray(path).astype(np.float32)

    # 关键一步：按「这台机器平时的波动」做标准化
    score = np.abs(img - mean_img) / (std_img + 1e-6)
    score = cv2.GaussianBlur(score, (5, 5), 0)

    # 阈值分割
    mask = (score > K).astype(np.uint8) * 255

    # 形态学：先开运算去掉零星噪点，再闭运算把断裂的缺陷连成一片
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    return score, mask


# ========== 3. 挑样本看效果：2 张正常 + 每类缺陷各 1 张 ==========
samples = [(p, "GOOD") for p in sorted((DATA / CATEGORY / "test" / "good").glob("*.png"))[:2]]
for sub in sorted((DATA / CATEGORY / "test").iterdir()):
    if sub.is_dir() and sub.name != "good":
        picks = sorted(sub.glob("*.png"))[:1]
        samples += [(p, sub.name) for p in picks]

fig, axes = plt.subplots(len(samples), 4, figsize=(14, 2.8 * len(samples)))

for i, (path, label) in enumerate(samples):
    raw = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    raw = cv2.resize(raw, SIZE)
    score, mask = detect(path)
    peak = score.max()          # 异常分数：全图最可疑的那个点

    axes[i, 0].imshow(raw)
    axes[i, 0].set_title(f"{label}", fontsize=10)
    axes[i, 0].axis("off")

    im = axes[i, 1].imshow(score, cmap="jet", vmin=0, vmax=K * 2)
    axes[i, 1].set_title(f"score (max={peak:.1f})", fontsize=10)
    axes[i, 1].axis("off")

    axes[i, 2].imshow(mask, cmap="gray")
    axes[i, 2].set_title(f"mask (K={K})", fontsize=10)
    axes[i, 2].axis("off")

    # 叠加显示
    overlay = raw.copy()
    overlay[mask > 0] = [255, 0, 0]
    axes[i, 3].imshow(overlay)
    axes[i, 3].set_title("overlay", fontsize=10)
    axes[i, 3].axis("off")

    print(f"{label:<18} 峰值分数 = {peak:6.2f}   异常像素占比 = {mask.mean()/255*100:5.2f}%")

plt.tight_layout()
plt.savefig(f"detect_{CATEGORY}.png", dpi=110, bbox_inches="tight")
print(f"\n已保存：detect_{CATEGORY}.png")
plt.show()
