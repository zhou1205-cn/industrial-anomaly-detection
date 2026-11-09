"""工业缺陷检测演示界面"""
from pathlib import Path
import csv
import random

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

DATA = Path(r"D:\datasets\mvtec_ad")
SIZE = 256
CORESET_RATIO = 0.1
K_NEIGHBORS = 9

device = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="工业缺陷检测系统", layout="wide")

MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


# ---------------- 模型 ----------------
@st.cache_resource
def get_backbone():
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
    feat = {}
    net.layer2.register_forward_hook(lambda m, i, o: feat.__setitem__("l2", o))
    net.layer3.register_forward_hook(lambda m, i, o: feat.__setitem__("l3", o))
    return net, feat


def to_tensor(bgr):
    rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (SIZE, SIZE))
    t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.
    return (t.to(device) - MEAN) / STD


@torch.no_grad()
def extract(net, feat, t):
    net(t)
    l2 = feat["l2"]
    l3 = F.interpolate(feat["l3"], size=l2.shape[-2:], mode="bilinear", align_corners=False)
    return F.avg_pool2d(torch.cat([l2, l3], dim=1), 3, 1, 1)


# ---------------- 记忆库 ----------------
@st.cache_resource(show_spinner="正在学习正常样本（首次约 20-40 秒）…")
def build_bank(category: str, n_train: int):
    net, feat = get_backbone()
    paths = sorted((DATA / category / "train" / "good").glob("*.png"))
    if n_train < len(paths):
        rng = np.random.RandomState(42)
        idx = sorted(rng.choice(len(paths), size=n_train, replace=False))
        paths = [paths[i] for i in idx]

    chunks = []
    for p in paths:
        f = extract(net, feat, to_tensor(cv2.imread(str(p))))
        chunks.append(f[0].permute(1, 2, 0).reshape(-1, f.shape[1]).cpu())

    bank = torch.cat(chunks, 0)
    torch.manual_seed(0)
    n = max(500, int(bank.shape[0] * CORESET_RATIO))
    bank = bank[torch.randperm(bank.shape[0])[:n]].contiguous().to(device)

    # 用正常训练图自身的分数分布，算出"正常品波动的上限"
    ref = []
    for p in paths[:min(30, len(paths))]:
        sm = detect(cv2.imread(str(p)), bank)
        ref.append(float(sm.max()))
    ref = np.array(ref)
    return bank, len(paths), float(ref.mean()), float(ref.std())


@torch.no_grad()
def detect(bgr, bank):
    net, feat = get_backbone()
    f = extract(net, feat, to_tensor(bgr))[0]
    q = f.permute(1, 2, 0).reshape(-1, f.shape[0])
    d = torch.cdist(q, bank)
    d9 = d.topk(min(K_NEIGHBORS, bank.shape[0]), dim=1, largest=False).values
    s = d9.mean(1).reshape(f.shape[1], f.shape[2])
    s = F.interpolate(s[None, None], (SIZE, SIZE), mode="bilinear", align_corners=False)[0, 0]
    return cv2.GaussianBlur(s.cpu().numpy(), (7, 7), 0)


# ---------------- 读历史成绩 ----------------
def load_ref():
    ref = {}
    p = Path("results_compare.csv")
    if p.exists():
        with open(p, encoding="utf-8") as f:
            for row in csv.reader(f):
                if row[0] == "category":
                    continue
                ref[row[0]] = float(row[2])
    return ref


# ---------------- 界面 ----------------
st.title("工业缺陷检测系统")
st.caption("只学习正常样本 · PatchCore · MVTec AD 基准")

cats = sorted([d.name for d in DATA.iterdir() if d.is_dir() and not d.name.startswith(".")])
ref = load_ref()

with st.sidebar:
    st.header("检测设置")
    cat = st.selectbox("产品类别", cats)
    n_train = st.selectbox(
        "训练样本数", [5, 10, 20, 50, 999], index=2,
        format_func=lambda x: "全部" if x == 999 else str(x),
    )
    k_sigma = st.slider("判定严格度 (k × σ)", 2.0, 5.0, 3.0, 0.5,
                        help="阈值 = 正常品平均分 + k×波动。k 越大越难被判为不良")

    if cat in ref:
        st.metric("该类别历史 AUROC", f"{ref[cat]:.3f}")

bank, used_n, ref_mean, ref_std = build_bank(cat, n_train)

st.success(f"已用 {used_n} 张正常样本建立记忆库（{bank.shape[0]} 个特征）")

st.subheader("选择待检图片")
src = st.radio("图片来源", ["从测试集抽取", "上传自己的图片"], horizontal=True)

if src == "上传自己的图片":
    up = st.file_uploader("上传图片", type=["png", "jpg", "jpeg"])
    if up is None:
        st.info("请上传一张待检测的图片")
        st.stop()
    bgr = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
    label_truth = "未知"
else:
    test_dir = DATA / cat / "test"
    subs = [s for s in sorted(test_dir.iterdir()) if s.is_dir()]
    opt = st.selectbox(
        "缺陷类型", ["随机"] + [s.name for s in subs],
        format_func=lambda x: "随机抽一张" if x == "随机" else ("正常品" if x == "good" else f"缺陷：{x}"),
    )
    sub = random.choice(subs) if opt == "随机" else test_dir / opt
    files = sorted(sub.glob("*.png"))
    fp = random.choice(files)
    bgr = cv2.imread(str(fp))
    label_truth = "正常" if sub.name == "good" else sub.name

    # 读取真值掩码（数据集自带的人工标注）
    gt_mask = None
    if sub.name != "good":
        gp = DATA / cat / "ground_truth" / sub.name / f"{fp.stem}_mask.png"
        if gp.exists():
            gt_mask = cv2.resize(
                cv2.imread(str(gp), cv2.IMREAD_GRAYSCALE), (SIZE, SIZE),
                interpolation=cv2.INTER_NEAREST,
            )

# ---------------- 检测 ----------------
sm = detect(bgr, bank)
peak = float(sm.max())

# 绝对阈值：正常品的平均分 + k 倍波动
thr = ref_mean + k_sigma * ref_std

# 二值化后只保留最大连通域，去掉零散噪点
raw_mask = (sm > thr).astype(np.uint8)
n_cc, labels_cc, stats_cc, _ = cv2.connectedComponentsWithStats(raw_mask, 8)
pred = np.zeros_like(raw_mask)
if n_cc > 1:
    areas = stats_cc[1:, cv2.CC_STAT_AREA]
    biggest = 1 + int(np.argmax(areas))
    if areas.max() >= 30:          # 面积太小的不要
        pred = (labels_cc == biggest).astype(np.uint8)

is_ng = pred.sum() > 0 and peak > thr



raw = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (SIZE, SIZE))

heat = cv2.applyColorMap(np.uint8(np.clip(sm / (sm.max() + 1e-6) * 255, 0, 255)), cv2.COLORMAP_JET)
heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

overlay = raw.copy()
overlay[sm > thr] = [255, 40, 40]

pred = sm > thr
col1, col2, col3 = st.columns(3)
col1.image(raw, caption=f"输入图像（真值：{label_truth}）", use_container_width=True)
col2.image(heat, caption="异常热力图（越红越可疑）", use_container_width=True)
col3.image(overlay, caption=f"预测标记（阈值 {thr:.2f}）", use_container_width=True)

if gt_mask is not None:
    gt_bin = gt_mask > 127
    inter = (pred & gt_bin).sum()
    union = (pred | gt_bin).sum()
    iou = inter / union if union > 0 else 0.0

    gt_col = raw.copy()
    gt_col[gt_bin] = [0, 200, 0]

    st.markdown("#### 与人工标注对比")
    c4, c5 = st.columns(2)
    c4.image(gt_col, caption="真值标注（绿色）", use_container_width=True)
    c5.metric("定位 IoU", f"{iou:.3f}",
              help="预测区域与真实缺陷区域的重叠比例。0 = 完全错位，1 = 完全重合")


st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("峰值异常分数", f"{peak:.2f}")
c2.metric("判定阈值", f"{thr:.2f}",help=f"正常品平均 {ref_mean:.2f} + {k_sigma:.1f}×{ref_std:.2f}")

c3.metric("检测结论", "NG（不良）" if is_ng else "OK（合格）")
