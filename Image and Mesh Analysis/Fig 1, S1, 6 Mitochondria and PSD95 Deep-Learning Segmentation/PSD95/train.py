from pathlib import Path
import json
import csv
import time
import random
import hashlib

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from scipy.ndimage import gaussian_filter


# ============================================================
# SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "TrainingTifs"

MITO_TIF = DATA_DIR / "PSD95.tif"
MASK_TIF = DATA_DIR / "PSD95Mask.tif"

OUTPUT_DIR = ROOT / "Outputs"
CACHE_DIR = OUTPUT_DIR / "cache"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
VIS_DIR = OUTPUT_DIR / "visualizations"

METRICS_CSV = OUTPUT_DIR / "metrics.csv"
CONFIG_JSON = OUTPUT_DIR / "config.json"
SPLIT_JSON = OUTPUT_DIR / "frame_split.json"

OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
VIS_DIR.mkdir(exist_ok=True)

SEED = 1

# Full-frame training.
# One sample = one full 2048x2048 frame.
BATCH_SIZE = 1
NUM_WORKERS = 0

VAL_FRACTION = 0.20

EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Full frame is memory heavy. 24 is a good starting point.
# If memory is still fine and you want larger, try 32.
SEG_BASE_CHANNELS = 24
AVG_BASE_CHANNELS = 16

LOCAL_Z_SIGMA = 12.0
EDGE_SIGMA = 1.0
AVG_MASK_BLUR_SIGMA = 3.0
Z_CLIP = 6.0

AUGMENT = True

FLIP_PROB = 0.50
ROT90_PROB = 0.50

INTENSITY_SCALE_RANGE = (0.70, 1.30)
INTENSITY_OFFSET_RANGE = (-0.25, 0.25)
GAMMA_RANGE = (0.80, 1.25)
NOISE_STD_RANGE = (0.00, 0.08)

SAVE_EVERY = 1
VIS_EVERY = 1
RESUME = True

THRESHOLD = 0.5
MAX_POS_WEIGHT = 50.0

AVG_GATE_LOSS_WEIGHT = 0.20

GATE_MIN_SCALE = 0.10
GATE_MAX_SCALE = 2.00

CACHE_DTYPE = np.float16


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(SEED)

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


# ============================================================
# DEVICE
# ============================================================

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()
print(f"Using device: {DEVICE}")


# ============================================================
# IMAGE HELPERS
# ============================================================

def read_tif_stack(path):
    arr = tiff.imread(str(path))
    arr = np.asarray(arr)

    if arr.ndim == 2:
        arr = arr[None, ...]

    if arr.ndim != 3:
        raise ValueError(f"Expected T,Y,X stack. Got {arr.shape} from {path}")

    return arr.astype(np.float32)


def safe_mask(mask):
    return (mask > 0).astype(np.float32)


def global_zscore(img, eps=1e-6):
    mean = float(np.mean(img))
    std = float(np.std(img))

    if std < eps:
        std = eps

    z = (img - mean) / std
    z = np.clip(z, -Z_CLIP, Z_CLIP)

    return z.astype(np.float32)


def local_zscore(img, sigma=12.0, eps=1e-6):
    local_mean = gaussian_filter(img, sigma=sigma)
    local_sq_mean = gaussian_filter(img * img, sigma=sigma)

    local_var = np.maximum(local_sq_mean - local_mean * local_mean, 0)
    local_std = np.sqrt(local_var + eps)

    z = (img - local_mean) / local_std
    z = np.clip(z, -Z_CLIP, Z_CLIP)

    return z.astype(np.float32)


def gradient_map(img, sigma=1.0):
    sm = gaussian_filter(img, sigma=sigma)

    gy, gx = np.gradient(sm)
    g = np.sqrt(gx * gx + gy * gy)

    return global_zscore(g)


def make_soft_avg_mask(mask_stack):
    avg = np.mean(safe_mask(mask_stack), axis=0).astype(np.float32)

    if AVG_MASK_BLUR_SIGMA > 0:
        avg = gaussian_filter(avg, sigma=AVG_MASK_BLUR_SIGMA)

    mn = float(avg.min())
    mx = float(avg.max())

    if mx > mn:
        avg = (avg - mn) / (mx - mn)

    avg = np.clip(avg, 0, 1)

    return avg.astype(np.float32)


def normalize_for_display(img):
    arr = img.astype(np.float32).copy()
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-8)
    return arr


# ============================================================
# SERIES HELPERS
# ============================================================

def get_series_ranges(n_frames):
    if n_frames == 50:
        return [(0, 25), (25, 50)]

    print(f"Warning: expected 50 frames, got {n_frames}. Using first half and second half.")
    mid = n_frames // 2

    return [(0, mid), (mid, n_frames)]


def make_frame_items(stack_shape):
    n_frames, height, width = stack_shape
    series_ranges = get_series_ranges(n_frames)

    items = []

    for series_id, (start_t, end_t) in enumerate(series_ranges):
        for t_idx in range(start_t, end_t):
            items.append({
                "series_id": int(series_id),
                "series_start": int(start_t),
                "series_end": int(end_t),
                "t": int(t_idx),
            })

    return items


def split_indices(n_items, val_fraction, seed):
    indices = list(range(n_items))

    rng = random.Random(seed)
    rng.shuffle(indices)

    n_val = int(round(n_items * val_fraction))

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    return train_indices, val_indices


# ============================================================
# CACHE IDS
# ============================================================

def get_current_cache_id(mito_shape):
    txt = json.dumps({
        "mito_shape": tuple(int(x) for x in mito_shape),
        "local_z_sigma": LOCAL_Z_SIGMA,
        "edge_sigma": EDGE_SIGMA,
        "z_clip": Z_CLIP,
    }, sort_keys=True)

    return hashlib.md5(txt.encode("utf-8")).hexdigest()[:12]


def get_avg_cache_id(mito_shape):
    txt = json.dumps({
        "mito_shape": tuple(int(x) for x in mito_shape),
        "local_z_sigma": LOCAL_Z_SIGMA,
        "edge_sigma": EDGE_SIGMA,
        "avg_mask_blur_sigma": AVG_MASK_BLUR_SIGMA,
        "z_clip": Z_CLIP,
    }, sort_keys=True)

    return hashlib.md5(txt.encode("utf-8")).hexdigest()[:12]


# ============================================================
# FEATURE PRECOMPUTE
# ============================================================

def precompute_current_features(mito, current_cache_id):
    path = CACHE_DIR / f"current_features_{current_cache_id}.npy"

    n_frames, h, w = mito.shape
    expected_shape = (n_frames, 3, h, w)

    if path.exists():
        arr = np.load(path, mmap_mode="r")
        if arr.shape == expected_shape:
            print(f"Using cached current features: {path.name}")
            return arr
        print("Cached current features shape mismatch. Rebuilding.")

    for old_path in sorted(CACHE_DIR.glob("current_features_*.npy")):
        try:
            arr = np.load(old_path, mmap_mode="r")
            if arr.shape == expected_shape:
                print(f"Using existing current features with matching shape: {old_path.name}")
                return arr
        except Exception:
            pass

    print("Precomputing current-frame features...")
    print("Channels: global_z, local_z, gradient")

    arr = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=CACHE_DTYPE,
        shape=expected_shape,
    )

    for t_idx in range(n_frames):
        print(f"  frame {t_idx + 1}/{n_frames}")

        img = mito[t_idx]

        gz = global_zscore(img)
        lz = local_zscore(img, sigma=LOCAL_Z_SIGMA)
        grad = gradient_map(lz, sigma=EDGE_SIGMA)

        arr[t_idx, 0] = gz.astype(CACHE_DTYPE)
        arr[t_idx, 1] = lz.astype(CACHE_DTYPE)
        arr[t_idx, 2] = grad.astype(CACHE_DTYPE)

        arr.flush()

    print("Current-frame feature precompute done.")

    return np.load(path, mmap_mode="r")


def precompute_average_features_and_targets(current_features, mask, avg_cache_id):
    avg_feat_path = CACHE_DIR / f"average_features_{avg_cache_id}.npy"
    avg_target_path = CACHE_DIR / f"average_mask_targets_{avg_cache_id}.npy"

    n_frames, _, h, w = current_features.shape
    series_ranges = get_series_ranges(n_frames)

    expected_feat_shape = (len(series_ranges), 4, h, w)
    expected_target_shape = (len(series_ranges), 1, h, w)

    if avg_feat_path.exists() and avg_target_path.exists():
        avg_features = np.load(avg_feat_path, mmap_mode="r")
        avg_targets = np.load(avg_target_path, mmap_mode="r")

        if avg_features.shape == expected_feat_shape and avg_targets.shape == expected_target_shape:
            print(f"Using cached average features: {avg_feat_path.name}")
            print(f"Using cached average mask targets: {avg_target_path.name}")
            return avg_features, avg_targets

        print("Cached average feature/target shape mismatch. Rebuilding.")

    print("Precomputing average-projection features and average-mask targets...")
    print("Average feature channels: mean_global_z, mean_local_z, mean_gradient, candidate_occupancy")

    avg_features = np.lib.format.open_memmap(
        avg_feat_path,
        mode="w+",
        dtype=CACHE_DTYPE,
        shape=expected_feat_shape,
    )

    avg_targets = np.lib.format.open_memmap(
        avg_target_path,
        mode="w+",
        dtype=CACHE_DTYPE,
        shape=expected_target_shape,
    )

    for series_id, (start_t, end_t) in enumerate(series_ranges):
        print(f"  series {series_id + 1}/{len(series_ranges)}: frames {start_t + 1}-{end_t}")

        gz_accum = np.zeros((h, w), dtype=np.float32)
        lz_accum = np.zeros((h, w), dtype=np.float32)
        grad_accum = np.zeros((h, w), dtype=np.float32)
        cand_accum = np.zeros((h, w), dtype=np.float32)

        for tt in range(start_t, end_t):
            gz = np.asarray(current_features[tt, 0], dtype=np.float32)
            lz = np.asarray(current_features[tt, 1], dtype=np.float32)
            grad = np.asarray(current_features[tt, 2], dtype=np.float32)
            cand = (lz > 0.5).astype(np.float32)

            gz_accum += gz
            lz_accum += lz
            grad_accum += grad
            cand_accum += cand

        denom = max(end_t - start_t, 1)

        avg_features[series_id, 0] = (gz_accum / denom).astype(CACHE_DTYPE)
        avg_features[series_id, 1] = (lz_accum / denom).astype(CACHE_DTYPE)
        avg_features[series_id, 2] = (grad_accum / denom).astype(CACHE_DTYPE)
        avg_features[series_id, 3] = (cand_accum / denom).astype(CACHE_DTYPE)

        avg_mask = make_soft_avg_mask(mask[start_t:end_t])
        avg_targets[series_id, 0] = avg_mask.astype(CACHE_DTYPE)

        avg_features.flush()
        avg_targets.flush()

    print("Average feature and target precompute done.")

    return (
        np.load(avg_feat_path, mmap_mode="r"),
        np.load(avg_target_path, mmap_mode="r"),
    )


# ============================================================
# CLASS WEIGHT
# ============================================================

def derive_pos_weight(mask):
    m = safe_mask(mask)

    pos = float(m.sum())
    total = float(m.size)
    neg = total - pos

    if pos <= 0:
        raise ValueError("Mask has no positive pixels.")

    derived = neg / pos
    pos_weight = min(derived, MAX_POS_WEIGHT)

    return pos_weight, derived, pos, neg


# ============================================================
# DATASET
# ============================================================

class MitoFullFrameDataset(Dataset):
    def __init__(
        self,
        mask,
        current_features,
        average_features,
        average_targets,
        frame_items,
        indices,
        augment=False,
    ):
        self.mask = mask
        self.current_features = current_features
        self.average_features = average_features
        self.average_targets = average_targets
        self.frame_items = frame_items
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def _augment(self, current, avg_input, target, avg_target):
        scale = random.uniform(*INTENSITY_SCALE_RANGE)
        offset = random.uniform(*INTENSITY_OFFSET_RANGE)
        gamma = random.uniform(*GAMMA_RANGE)
        noise_std = random.uniform(*NOISE_STD_RANGE)

        current = current * scale + offset
        avg_input[0:3] = avg_input[0:3] * scale + offset

        sign_current = np.sign(current)
        current = sign_current * (np.abs(current) ** gamma)

        sign_avg = np.sign(avg_input[0:3])
        avg_input[0:3] = sign_avg * (np.abs(avg_input[0:3]) ** gamma)

        if noise_std > 0:
            current = current + np.random.normal(0, noise_std, current.shape).astype(np.float32)
            avg_input[0:3] = avg_input[0:3] + np.random.normal(0, noise_std, avg_input[0:3].shape).astype(np.float32)

        if random.random() < FLIP_PROB:
            current = current[:, :, ::-1].copy()
            avg_input = avg_input[:, :, ::-1].copy()
            target = target[:, ::-1].copy()
            avg_target = avg_target[:, ::-1].copy()

        if random.random() < FLIP_PROB:
            current = current[:, ::-1, :].copy()
            avg_input = avg_input[:, ::-1, :].copy()
            target = target[::-1, :].copy()
            avg_target = avg_target[::-1, :].copy()

        if random.random() < ROT90_PROB:
            k = random.randint(0, 3)

            current = np.rot90(current, k=k, axes=(1, 2)).copy()
            avg_input = np.rot90(avg_input, k=k, axes=(1, 2)).copy()
            target = np.rot90(target, k=k, axes=(0, 1)).copy()
            avg_target = np.rot90(avg_target, k=k, axes=(0, 1)).copy()

        current = np.clip(current, -Z_CLIP, Z_CLIP)
        avg_input[0:3] = np.clip(avg_input[0:3], -Z_CLIP, Z_CLIP)
        avg_input[3] = np.clip(avg_input[3], 0, 1)

        target = np.clip(target, 0, 1)
        avg_target = np.clip(avg_target, 0, 1)

        return current, avg_input, target, avg_target

    def __getitem__(self, idx):
        item_index = self.indices[idx]
        item = self.frame_items[item_index]

        series_id = item["series_id"]
        t_idx = item["t"]

        current = np.asarray(self.current_features[t_idx], dtype=np.float32)
        avg_input = np.asarray(self.average_features[series_id], dtype=np.float32)

        target = safe_mask(self.mask[t_idx])
        avg_target = np.asarray(self.average_targets[series_id, 0], dtype=np.float32)

        if self.augment:
            current, avg_input, target, avg_target = self._augment(
                current,
                avg_input,
                target,
                avg_target,
            )

        current = torch.from_numpy(current.astype(np.float32))
        avg_input = torch.from_numpy(avg_input.astype(np.float32))
        target = torch.from_numpy(target[None, ...].astype(np.float32))
        avg_target = torch.from_numpy(avg_target[None, ...].astype(np.float32))

        return current, avg_input, target, avg_target


# ============================================================
# MODEL
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()

        g = min(groups, out_ch)

        while out_ch % g != 0 and g > 1:
            g -= 1

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(g, out_ch)
        self.act1 = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(g, out_ch)

        if in_ch != out_ch:
            self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.proj = nn.Identity()

        self.act2 = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = self.proj(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + identity
        out = self.act2(out)

        return out


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.pool = nn.MaxPool2d(2)
        self.block = ResidualBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.block(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = ResidualBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip, gate=None):
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        if gate is not None:
            g = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            skip = skip * (GATE_MIN_SCALE + (GATE_MAX_SCALE - GATE_MIN_SCALE) * g)

        x = torch.cat([x, skip], dim=1)
        x = self.block(x)

        return x


class AverageGateNet(nn.Module):
    def __init__(self, in_ch=4, base_ch=16):
        super().__init__()

        self.enc1 = ResidualBlock(in_ch, base_ch)
        self.enc2 = DownBlock(base_ch, base_ch * 2)
        self.enc3 = DownBlock(base_ch * 2, base_ch * 4)
        self.enc4 = DownBlock(base_ch * 4, base_ch * 8)

        self.bottleneck = nn.Sequential(
            DownBlock(base_ch * 8, base_ch * 16),
            ResidualBlock(base_ch * 16, base_ch * 16),
        )

        self.up4 = UpBlock(base_ch * 16, base_ch * 8, base_ch * 8)
        self.up3 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up1 = UpBlock(base_ch * 2, base_ch, base_ch)

        self.out = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, avg_input):
        e1 = self.enc1(avg_input)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        gate_logits = self.out(d1)
        gate = torch.sigmoid(gate_logits)

        return gate_logits, gate


class FrameSegNet(nn.Module):
    def __init__(self, in_ch=4, base_ch=24):
        super().__init__()

        self.enc1 = ResidualBlock(in_ch, base_ch)
        self.enc2 = DownBlock(base_ch, base_ch * 2)
        self.enc3 = DownBlock(base_ch * 2, base_ch * 4)
        self.enc4 = DownBlock(base_ch * 4, base_ch * 8)

        self.bottleneck = nn.Sequential(
            DownBlock(base_ch * 8, base_ch * 16),
            ResidualBlock(base_ch * 16, base_ch * 16),
        )

        self.up4 = UpBlock(base_ch * 16, base_ch * 8, base_ch * 8)
        self.up3 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up1 = UpBlock(base_ch * 2, base_ch, base_ch)

        self.out = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, current, gate):
        current_with_gate = torch.cat([current, gate], dim=1)

        e1 = self.enc1(current_with_gate)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d4 = self.up4(b, e4, gate=gate)
        d3 = self.up3(d4, e3, gate=gate)
        d2 = self.up2(d3, e2, gate=gate)
        d1 = self.up1(d2, e1, gate=gate)

        logits = self.out(d1)

        return logits


class AvgGuidedSegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.avg_gate_net = AverageGateNet(
            in_ch=4,
            base_ch=AVG_BASE_CHANNELS,
        )

        self.frame_seg_net = FrameSegNet(
            in_ch=4,
            base_ch=SEG_BASE_CHANNELS,
        )

    def forward(self, current, avg_input):
        avg_gate_logits, avg_gate = self.avg_gate_net(avg_input)

        seg_logits = self.frame_seg_net(
            current=current,
            gate=avg_gate,
        )

        return seg_logits, avg_gate_logits, avg_gate


# ============================================================
# LOSSES AND METRICS
# ============================================================

def dice_loss_from_logits(logits, target):
    prob = torch.sigmoid(logits)

    smooth = 1.0

    intersection = torch.sum(prob * target, dim=(1, 2, 3))
    denom = torch.sum(prob, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3))

    dice = (2 * intersection + smooth) / (denom + smooth)

    return 1.0 - dice.mean()


def soft_dice_loss_from_prob(prob, target):
    smooth = 1.0

    intersection = torch.sum(prob * target, dim=(1, 2, 3))
    denom = torch.sum(prob, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3))

    dice = (2 * intersection + smooth) / (denom + smooth)

    return 1.0 - dice.mean()


class AvgGuidedLoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()

        self.register_buffer(
            "pos_weight",
            torch.tensor([pos_weight], dtype=torch.float32),
        )

    def forward(self, seg_logits, target, avg_gate_logits, avg_gate, avg_target):
        pos_weight = self.pos_weight.to(seg_logits.device)

        seg_bce = F.binary_cross_entropy_with_logits(
            seg_logits,
            target,
            pos_weight=pos_weight,
        )

        seg_dice = dice_loss_from_logits(seg_logits, target)
        seg_loss = seg_bce + seg_dice

        # Average gate target is soft occupancy, not a strict frame segmentation.
        avg_bce = F.binary_cross_entropy_with_logits(
            avg_gate_logits,
            avg_target,
        )

        avg_dice = soft_dice_loss_from_prob(avg_gate, avg_target)
        avg_loss = avg_bce + avg_dice

        total_loss = seg_loss + AVG_GATE_LOSS_WEIGHT * avg_loss

        parts = {
            "seg_bce": float(seg_bce.detach().cpu().item()),
            "seg_dice": float(seg_dice.detach().cpu().item()),
            "seg_loss": float(seg_loss.detach().cpu().item()),
            "avg_bce": float(avg_bce.detach().cpu().item()),
            "avg_dice": float(avg_dice.detach().cpu().item()),
            "avg_loss": float(avg_loss.detach().cpu().item()),
        }

        return total_loss, parts


@torch.no_grad()
def batch_counts(logits, target, threshold=0.5):
    prob = torch.sigmoid(logits)
    pred = (prob >= threshold).float()

    tp = torch.sum(pred * target).item()
    fp = torch.sum(pred * (1 - target)).item()
    fn = torch.sum((1 - pred) * target).item()
    tn = torch.sum((1 - pred) * (1 - target)).item()

    return tp, fp, fn, tn


def metrics_from_counts(tp, fp, fn, tn):
    eps = 1e-8

    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }


@torch.no_grad()
def gate_stats(gate):
    return {
        "mean": float(gate.mean().detach().cpu().item()),
        "min": float(gate.min().detach().cpu().item()),
        "max": float(gate.max().detach().cpu().item()),
    }


# ============================================================
# CHECKPOINTS
# ============================================================

def get_settings_dict(current_cache_id=None, avg_cache_id=None):
    return {
        "current_cache_id": current_cache_id,
        "avg_cache_id": avg_cache_id,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "val_fraction": VAL_FRACTION,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "seg_base_channels": SEG_BASE_CHANNELS,
        "avg_base_channels": AVG_BASE_CHANNELS,
        "local_z_sigma": LOCAL_Z_SIGMA,
        "edge_sigma": EDGE_SIGMA,
        "avg_mask_blur_sigma": AVG_MASK_BLUR_SIGMA,
        "z_clip": Z_CLIP,
        "augment": AUGMENT,
        "avg_gate_loss_weight": AVG_GATE_LOSS_WEIGHT,
        "gate_min_scale": GATE_MIN_SCALE,
        "gate_max_scale": GATE_MAX_SCALE,
        "max_pos_weight": MAX_POS_WEIGHT,
        "device": str(DEVICE),
    }


def save_checkpoint(path, model, optimizer, epoch, best_val_loss, current_cache_id, avg_cache_id):
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "settings": get_settings_dict(current_cache_id, avg_cache_id),
    }

    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    return ckpt["epoch"], ckpt["best_val_loss"]


# ============================================================
# METRICS CSV
# ============================================================

def init_metrics_csv(path):
    header = [
        "epoch",
        "lr",
        "train_loss",
        "train_seg_loss",
        "train_avg_loss",
        "val_loss",
        "val_seg_loss",
        "val_avg_loss",
        "val_dice",
        "val_iou",
        "val_precision",
        "val_recall",
        "val_accuracy",
        "val_gate_mean",
        "val_gate_min",
        "val_gate_max",
        "seconds",
    ]

    if path.exists():
        try:
            with open(path, "r", newline="") as f:
                reader = csv.reader(f)
                existing_header = next(reader)
            if existing_header == header:
                return
        except Exception:
            pass

        backup = path.with_name(f"{path.stem}_previous_{int(time.time())}{path.suffix}")
        path.rename(backup)
        print(f"Existing metrics.csv had a different header. Moved it to {backup.name}")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_metrics(path, row):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# ============================================================
# VISUALIZATION
# ============================================================

@torch.no_grad()
def save_visualization(model, dataset, epoch, device, vis_idx=0):
    model.eval()

    current, avg_input, target, avg_target = dataset[vis_idx]

    current_b = current[None, ...].to(device)
    avg_b = avg_input[None, ...].to(device)

    seg_logits, avg_gate_logits, avg_gate = model(current_b, avg_b)

    prob = torch.sigmoid(seg_logits)[0, 0].detach().cpu().numpy()
    pred = (prob >= THRESHOLD).astype(np.float32)

    current_np = current.detach().cpu().numpy()
    avg_input_np = avg_input.detach().cpu().numpy()
    target_np = target[0].detach().cpu().numpy()
    avg_target_np = avg_target[0].detach().cpu().numpy()
    gate_np = avg_gate[0, 0].detach().cpu().numpy()

    current_display = normalize_for_display(current_np[0])
    avg_display = normalize_for_display(avg_input_np[0])

    overlay = np.zeros((target_np.shape[0], target_np.shape[1], 3), dtype=np.float32)
    overlay[..., 0] = current_display
    overlay[..., 1] = current_display
    overlay[..., 2] = current_display

    overlay[..., 0] = np.maximum(overlay[..., 0], pred)
    overlay[..., 1] = np.maximum(overlay[..., 1], target_np)

    fig, axes = plt.subplots(3, 4, figsize=(18, 13))

    axes[0, 0].imshow(avg_input_np[0], cmap="gray")
    axes[0, 0].set_title("avg global z")

    axes[0, 1].imshow(avg_input_np[1], cmap="gray")
    axes[0, 1].set_title("avg local z")

    axes[0, 2].imshow(avg_input_np[2], cmap="gray")
    axes[0, 2].set_title("avg gradient")

    axes[0, 3].imshow(avg_input_np[3], cmap="gray", vmin=0, vmax=1)
    axes[0, 3].set_title("avg candidate occupancy")

    axes[1, 0].imshow(avg_target_np, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("average GT target")

    axes[1, 1].imshow(gate_np, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("predicted average gate")

    axes[1, 2].imshow(gate_np - avg_target_np, cmap="bwr", vmin=-1, vmax=1)
    axes[1, 2].set_title("gate - average GT")

    axes[1, 3].imshow(avg_display, cmap="gray")
    axes[1, 3].set_title("average projection display")

    axes[2, 0].imshow(current_np[0], cmap="gray")
    axes[2, 0].set_title("current global z")

    axes[2, 1].imshow(target_np, cmap="gray", vmin=0, vmax=1)
    axes[2, 1].set_title("current GT mask")

    axes[2, 2].imshow(prob, cmap="gray", vmin=0, vmax=1)
    axes[2, 2].set_title("current prediction probability")

    axes[2, 3].imshow(overlay)
    axes[2, 3].set_title("overlay: red pred, green GT")

    for ax in axes.ravel():
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(VIS_DIR / f"epoch_{epoch:04d}.png", dpi=150)
    plt.close(fig)


# ============================================================
# TRAIN / VALIDATE
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_seg_loss = 0.0
    total_avg_loss = 0.0
    n_batches = 0

    for current, avg_input, target, avg_target in loader:
        current = current.to(device, non_blocking=False)
        avg_input = avg_input.to(device, non_blocking=False)
        target = target.to(device, non_blocking=False)
        avg_target = avg_target.to(device, non_blocking=False)

        optimizer.zero_grad(set_to_none=True)

        seg_logits, avg_gate_logits, avg_gate = model(current, avg_input)

        loss, parts = criterion(
            seg_logits=seg_logits,
            target=target,
            avg_gate_logits=avg_gate_logits,
            avg_gate=avg_gate,
            avg_target=avg_target,
        )

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_seg_loss += parts["seg_loss"]
        total_avg_loss += parts["avg_loss"]
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "seg_loss": total_seg_loss / max(n_batches, 1),
        "avg_loss": total_avg_loss / max(n_batches, 1),
    }


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_seg_loss = 0.0
    total_avg_loss = 0.0
    n_batches = 0

    tp_total = 0.0
    fp_total = 0.0
    fn_total = 0.0
    tn_total = 0.0

    gate_mean_total = 0.0
    gate_min_global = 1.0
    gate_max_global = 0.0

    for current, avg_input, target, avg_target in loader:
        current = current.to(device, non_blocking=False)
        avg_input = avg_input.to(device, non_blocking=False)
        target = target.to(device, non_blocking=False)
        avg_target = avg_target.to(device, non_blocking=False)

        seg_logits, avg_gate_logits, avg_gate = model(current, avg_input)

        loss, parts = criterion(
            seg_logits=seg_logits,
            target=target,
            avg_gate_logits=avg_gate_logits,
            avg_gate=avg_gate,
            avg_target=avg_target,
        )

        tp, fp, fn, tn = batch_counts(seg_logits, target, threshold=THRESHOLD)
        stats = gate_stats(avg_gate)

        tp_total += tp
        fp_total += fp
        fn_total += fn
        tn_total += tn

        total_loss += float(loss.item())
        total_seg_loss += parts["seg_loss"]
        total_avg_loss += parts["avg_loss"]

        gate_mean_total += stats["mean"]
        gate_min_global = min(gate_min_global, stats["min"])
        gate_max_global = max(gate_max_global, stats["max"])

        n_batches += 1

    metrics = metrics_from_counts(tp_total, fp_total, fn_total, tn_total)

    metrics["loss"] = total_loss / max(n_batches, 1)
    metrics["seg_loss"] = total_seg_loss / max(n_batches, 1)
    metrics["avg_loss"] = total_avg_loss / max(n_batches, 1)
    metrics["gate_mean"] = gate_mean_total / max(n_batches, 1)
    metrics["gate_min"] = gate_min_global
    metrics["gate_max"] = gate_max_global

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():
    print("Reading training data...")

    mito = read_tif_stack(MITO_TIF)
    mask = read_tif_stack(MASK_TIF)

    if mito.shape != mask.shape:
        raise ValueError(f"Mito and mask shapes differ: {mito.shape} vs {mask.shape}")

    print(f"Mito shape: {mito.shape}")
    print(f"Mask shape: {mask.shape}")

    current_cache_id = get_current_cache_id(mito.shape)
    avg_cache_id = get_avg_cache_id(mito.shape)

    pos_weight, derived_pos_weight, pos_pixels, neg_pixels = derive_pos_weight(mask)

    print(f"Positive pixels: {pos_pixels:.0f}")
    print(f"Negative pixels: {neg_pixels:.0f}")
    print(f"derived_pos_weight: {derived_pos_weight:.4f}")
    print(f"used_pos_weight: {pos_weight:.4f}")

    frame_items = make_frame_items(mito.shape)
    train_indices, val_indices = split_indices(
        n_items=len(frame_items),
        val_fraction=VAL_FRACTION,
        seed=SEED,
    )

    print(f"Total frames: {len(frame_items)}")
    print(f"Training frames: {len(train_indices)}")
    print(f"Validation frames: {len(val_indices)}")

    with open(SPLIT_JSON, "w") as f:
        json.dump({
            "frame_items": frame_items,
            "train_indices": train_indices,
            "val_indices": val_indices,
        }, f, indent=2)

    settings = get_settings_dict(current_cache_id, avg_cache_id)
    settings["positive_pixels"] = pos_pixels
    settings["negative_pixels"] = neg_pixels
    settings["derived_pos_weight"] = derived_pos_weight
    settings["used_pos_weight"] = pos_weight
    settings["train_frames"] = len(train_indices)
    settings["val_frames"] = len(val_indices)

    with open(CONFIG_JSON, "w") as f:
        json.dump(settings, f, indent=2)

    current_features = precompute_current_features(mito, current_cache_id)

    average_features, average_targets = precompute_average_features_and_targets(
        current_features=current_features,
        mask=mask,
        avg_cache_id=avg_cache_id,
    )

    train_dataset = MitoFullFrameDataset(
        mask=mask,
        current_features=current_features,
        average_features=average_features,
        average_targets=average_targets,
        frame_items=frame_items,
        indices=train_indices,
        augment=AUGMENT,
    )

    val_dataset = MitoFullFrameDataset(
        mask=mask,
        current_features=current_features,
        average_features=average_features,
        average_targets=average_targets,
        frame_items=frame_items,
        indices=val_indices,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False,
    )

    model = AvgGuidedSegmentationModel().to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    criterion = AvgGuidedLoss(pos_weight=pos_weight).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    start_epoch = 1
    best_val_loss = float("inf")

    latest_ckpt = CHECKPOINT_DIR / "latest.pt"

    if RESUME and latest_ckpt.exists():
        print(f"Resuming from {latest_ckpt}")
        loaded_epoch, best_val_loss = load_checkpoint(latest_ckpt, model, optimizer, DEVICE)
        start_epoch = loaded_epoch + 1
        print(f"Starting at epoch {start_epoch}")

    init_metrics_csv(METRICS_CSV)

    vis_idx = 0

    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()

        lr = optimizer.param_groups[0]["lr"]

        train_out = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE,
        )

        val_out = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=DEVICE,
        )

        seconds = time.time() - t0

        print(
            f"Epoch {epoch:04d} | "
            f"train_loss={train_out['loss']:.5f} | "
            f"train_seg_loss={train_out['seg_loss']:.5f} | "
            f"train_avg_loss={train_out['avg_loss']:.5f} | "
            f"val_loss={val_out['loss']:.5f} | "
            f"val_seg_loss={val_out['seg_loss']:.5f} | "
            f"val_avg_loss={val_out['avg_loss']:.5f} | "
            f"dice={val_out['dice']:.5f} | "
            f"iou={val_out['iou']:.5f} | "
            f"precision={val_out['precision']:.5f} | "
            f"recall={val_out['recall']:.5f} | "
            f"gate_mean={val_out['gate_mean']:.5f} | "
            f"gate_min={val_out['gate_min']:.5f} | "
            f"gate_max={val_out['gate_max']:.5f} | "
            f"{seconds:.1f}s"
        )

        append_metrics(
            METRICS_CSV,
            [
                epoch,
                lr,
                train_out["loss"],
                train_out["seg_loss"],
                train_out["avg_loss"],
                val_out["loss"],
                val_out["seg_loss"],
                val_out["avg_loss"],
                val_out["dice"],
                val_out["iou"],
                val_out["precision"],
                val_out["recall"],
                val_out["accuracy"],
                val_out["gate_mean"],
                val_out["gate_min"],
                val_out["gate_max"],
                seconds,
            ],
        )

        if epoch % VIS_EVERY == 0:
            save_visualization(
                model=model,
                dataset=val_dataset,
                epoch=epoch,
                device=DEVICE,
                vis_idx=vis_idx,
            )

        if epoch % SAVE_EVERY == 0:
            save_checkpoint(
                CHECKPOINT_DIR / "latest.pt",
                model,
                optimizer,
                epoch,
                best_val_loss,
                current_cache_id,
                avg_cache_id,
            )

        if val_out["loss"] < best_val_loss:
            best_val_loss = val_out["loss"]

            save_checkpoint(
                CHECKPOINT_DIR / "best.pt",
                model,
                optimizer,
                epoch,
                best_val_loss,
                current_cache_id,
                avg_cache_id,
            )

        if DEVICE.type == "mps":
            torch.mps.empty_cache()

    print("Done.")


if __name__ == "__main__":
    main()
