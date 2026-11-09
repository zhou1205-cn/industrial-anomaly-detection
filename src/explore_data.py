from pathlib import Path
import cv2
import matplotlib.pyplot as plt

# ========== 数据集路径 ==========
DATA = Path(r"D:\datasets\mvtec_ad")
IMG_EXT = (".png", ".jpg", ".jpeg")


def count_imgs(p: Path) -> int:
    """统计文件夹里的图片数量"""
    if not p.is_dir():
        return 0
    return sum(1 for f in p.iterdir() if f.suffix.lower() in IMG_EXT)


# ========== 1. 打印统计表 ==========
print("=" * 78)
print(f"{'Category':<14}{'Train':>8}{'Test':>8}   Defect types")
print("=" * 78)

cats = sorted([d for d in DATA.iterdir() if d.is_dir() and not d.name.startswith(".")])
tot_train = tot_test = 0

for cat in cats:
    n_train = count_imgs(cat / "train" / "good")
    test_dir = cat / "test"
    n_test = 0
    defects = []

    if test_dir.is_dir():
        for sub in sorted(test_dir.iterdir()):
            if sub.is_dir():
                c = count_imgs(sub)
                n_test += c
                if sub.name != "good":
                    defects.append(f"{sub.name}({c})")

    tot_train += n_train
    tot_test += n_test
    print(f"{cat.name:<14}{n_train:>8}{n_test:>8}   {', '.join(defects)}")

print("=" * 78)
print(f"类别总数：{len(cats)}")
print(f"训练图合计：{tot_train} 张（全部是正常品）")
print(f"测试图合计：{tot_test} 张（正常 + 各类缺陷）")


# ========== 2. 每个类别挑一张正常 + 一张缺陷看 ==========
SHOW = ["bottle", "hazelnut", "screw", "carpet", "transistor", "zipper"]

fig, axes = plt.subplots(len(SHOW), 2, figsize=(10, 3.0 * len(SHOW)))

for i, name in enumerate(SHOW):
    cat = DATA / name

    good_imgs = sorted((cat / "train" / "good").glob("*.png"))
    defect_imgs = []
    for sub in sorted((cat / "test").iterdir()):
        if sub.is_dir() and sub.name != "good":
            defect_imgs += sorted(sub.glob("*.png"))

    if not good_imgs or not defect_imgs:
        continue

    g = cv2.imread(str(good_imgs[0]))
    d = cv2.imread(str(defect_imgs[0]))

    axes[i, 0].imshow(cv2.cvtColor(g, cv2.COLOR_BGR2RGB))
    axes[i, 0].set_title(f"{name} — GOOD", fontsize=11)
    axes[i, 0].axis("off")

    axes[i, 1].imshow(cv2.cvtColor(d, cv2.COLOR_BGR2RGB))
    axes[i, 1].set_title(f"{name} — DEFECT", fontsize=11)
    axes[i, 1].axis("off")

plt.tight_layout()
plt.savefig("data_preview.png", dpi=110, bbox_inches="tight")
print("\n已保存预览图：data_preview.png")
plt.show()

