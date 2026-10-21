from pathlib import Path
import argparse
import json
import csv
import random
import time

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ============================================================
# SETTINGS
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR / "manual segmentations"
EM_NAME = "EM_stack"  # TIFF filename stem; change this to match your EM images.
TRAIN_ROOT = DATA_ROOT / "train"
VALIDATION_ROOT = DATA_ROOT / "validate"
TEST_ROOT = DATA_ROOT / "test"
OUT_DIR = SCRIPT_DIR / "runs"

# Resume.
RESUME_FROM_LAST = True
RESUME_CHECKPOINT_PATH = OUT_DIR / "last.pt"

# Keep None to resume with optimizer LR saved in last.pt.
# Set to 1e-4 or 5e-5 only if you want to force a lower LR after resume.
NEW_LR_AFTER_RESUME = None

# Label meaning:
# OMM  = whole mitochondrial volume parent mask
# IBM  = inner-boundary-enclosed volume, including matrix + cristae lumen
# IMS  = intermembrane space + cristae lumen
# ER   = outside mitochondria
# Ribo = outside mitochondria, punctate
CHANNELS = ["OMM", "IBM", "IMS", "ER", "Ribo"]  # Change filename stems; preserve the five biological roles in this order.
N_CLASSES = len(CHANNELS)

OMM_IDX = 0
IBM_IDX = 1
IMS_IDX = 2
ER_IDX = 3
RIBO_IDX = 4

# Affinity is useful for continuous volumes, not punctate ribosomes.
AFFINITY_CHANNELS = CHANNELS[:4]
AFFINITY_CLASS_IDXS = [CHANNELS.index(ch) for ch in AFFINITY_CHANNELS]
N_AFFINITY_CLASSES = len(AFFINITY_CHANNELS)

SEED = 42

PATCH_SIZE = (32, 256, 256)   # Z, Y, X
INFER_STRIDE = (16, 128, 128)  # 50% overlap for complete-tomogram evaluation
BASE_CH = 24

# EPOCHS is the final epoch number, not the number of extra epochs.
# Example: if last.pt is epoch 56 and you want 100 total, keep EPOCHS = 100.
EPOCHS = 200
TRAIN_SAMPLES_PER_EPOCH = 256

VALIDATION_MODE = "complete_held_out_tomograms"

BATCH_SIZE = 1
NUM_WORKERS = 0

LR = 2e-4
WEIGHT_DECAY = 1e-4

CLASS_WEIGHT_CLIP = (0.5, 8.0)
BCE_POS_WEIGHT_CLIP = (1.0, 50.0)

CLASS_SAMPLING = {
    CHANNELS[0]: 0.16,
    CHANNELS[1]: 0.12,
    CHANNELS[2]: 0.12,
    CHANNELS[3]: 0.18,
    CHANNELS[4]: 0.28,
    "background": 0.10,
    "random": 0.04,
}

LOSS_W_BCE = 1.0
LOSS_W_DICE = 1.0
LOSS_W_BOUNDARY = 0.15
LOSS_W_AFFINITY = 0.15
LOSS_W_HIER = 0.20

BOUNDARY_POS_WEIGHT = 8.0
AFFINITY_POS_WEIGHT = 3.0

# ER/Ribo exclusivity penalty is added inside hierarchy_loss().
# Keep modest because output/targets already enforce the rule.
LOSS_W_ER_RIBO_EXCLUSIVE = 0.20

# Bottleneck attention.
USE_BOTTLENECK_ATTENTION = True
ATTENTION_HEADS = 4
ATTENTION_QK_DIM = 96
ATTENTION_FFN_MULT = 2

VIZ_SLICE_INDEX = 31   # Python index. This is slice 32 in ImageJ-style counting.
VIZ_GRID_STEP = 16
VIZ_RANDOM_CANDIDATES_PER_CLASS = 750

# Best model selection.
BEST_MODEL_METRIC = "mean_complete_validation_dice"
THRESHOLDS = {ch: 0.50 for ch in CHANNELS}

# ============================================================
# BASIC UTILITIES
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def safe_torch_load(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")

def load_training_log(path):
    if not path.exists():
        return []

    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fixed = {}
            for k, v in row.items():
                if v is None or v == "":
                    fixed[k] = v
                    continue

                try:
                    if k == "epoch":
                        fixed[k] = int(float(v))
                    else:
                        fixed[k] = float(v)
                except ValueError:
                    fixed[k] = v

            rows.append(fixed)

    return rows

def best_values_from_logs(logs):
    best_metric = -float("inf")
    best_loss = float("inf")

    for r in logs:
        v_dice = r.get("val_mean_dice", None)
        v_loss = r.get("val_loss", None)

        if isinstance(v_dice, (float, int)) and np.isfinite(v_dice):
            best_metric = max(best_metric, float(v_dice))

        if isinstance(v_loss, (float, int)) and np.isfinite(v_loss):
            best_loss = min(best_loss, float(v_loss))

    return best_metric, best_loss

def find_file(folder, stem, required=True):
    stem = str(stem)
    if Path(stem).suffix.lower() in (".tif", ".tiff"):
        direct = folder / stem
        if direct.is_file():
            return direct
        stem = str(Path(stem).with_suffix(""))
    for suffix in (".tif", ".tiff", ".TIF", ".TIFF"):
        path = folder / f"{stem}{suffix}"
        if path.is_file():
            return path
    if required:
        raise FileNotFoundError(f"Missing required TIFF: {stem} in {folder}")
    return None

def read_zyx(path):
    arr = tiff.imread(str(path))
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D ZYX tif, got shape {arr.shape} for {path}")
    return arr

def normalize_em_zscore(em):
    em = em.astype(np.float32)
    mean = float(em.mean())
    std = float(em.std())
    if std < 1e-6:
        std = 1.0
    em = (em - mean) / std
    em = np.clip(em, -5.0, 5.0)
    return em.astype(np.float32)

def crop_pad_3d(arr, center, patch_size, fill_value=0):
    zc, yc, xc = center
    pz, py, px = patch_size
    spatial_offset = arr.ndim - 3

    z0 = int(round(zc - pz // 2))
    y0 = int(round(yc - py // 2))
    x0 = int(round(xc - px // 2))
    z1, y1, x1 = z0 + pz, y0 + py, x0 + px

    Z, Y, X = arr.shape[-3:]

    src_z0, src_z1 = max(0, z0), min(Z, z1)
    src_y0, src_y1 = max(0, y0), min(Y, y1)
    src_x0, src_x1 = max(0, x0), min(X, x1)

    dst_z0, dst_z1 = src_z0 - z0, src_z1 - z0
    dst_y0, dst_y1 = src_y0 - y0, src_y1 - y0
    dst_x0, dst_x1 = src_x0 - x0, src_x1 - x0

    out_shape = arr.shape[:spatial_offset] + tuple(patch_size)
    out = np.full(out_shape, fill_value, dtype=arr.dtype)

    src_slices = (slice(None),) * spatial_offset + (slice(src_z0, src_z1), slice(src_y0, src_y1), slice(src_x0, src_x1))
    dst_slices = (slice(None),) * spatial_offset + (slice(dst_z0, dst_z1), slice(dst_y0, dst_y1), slice(dst_x0, dst_x1))
    out[dst_slices] = arr[src_slices]

    return out

def discover_tomogram_folders(root):
    if not root.exists():
        raise FileNotFoundError(f"Missing input folder: {root}")
    folders = [p for p in [root, *root.rglob("*")] if p.is_dir() and find_file(p, EM_NAME, required=False) is not None]
    if not folders:
        raise RuntimeError(f"No tomograms containing {EM_NAME}.tif/.tiff found under {root}")
    return sorted(set(folders))

def discover_split_folders():
    train = discover_tomogram_folders(TRAIN_ROOT)
    validation = discover_tomogram_folders(VALIDATION_ROOT)
    test = discover_tomogram_folders(TEST_ROOT)
    sets = {"train": train, "validation": validation, "test": test}
    resolved = {k: {p.resolve() for p in v} for k, v in sets.items()}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = resolved[a] & resolved[b]
        if overlap:
            raise RuntimeError(f"Tomogram leakage between {a} and {b}: {sorted(map(str, overlap))}")
    return train, validation, test

def save_log_csv(rows, path):
    if len(rows) == 0:
        return

    fieldnames = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# ============================================================
# TOMOGRAM LOADING
# ============================================================
class TomogramData:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.name = self.folder.name

        em_path = find_file(self.folder, EM_NAME, required=True)
        self.em = normalize_em_zscore(read_zyx(em_path))

        Z, Y, X = self.em.shape
        self.labels = np.zeros((N_CLASSES, Z, Y, X), dtype=np.uint8)
        self.valid = np.ones((N_CLASSES,), dtype=np.float32)

        for ci, ch in enumerate(CHANNELS):
            p = find_file(self.folder, ch, required=(ci != RIBO_IDX))
            if p is None:
                self.valid[ci] = 0.0
                continue
            lab = read_zyx(p)
            if lab.shape != self.em.shape:
                raise ValueError(f"Shape mismatch in {self.folder}: {ch} {lab.shape} vs EM {self.em.shape}")
            self.labels[ci] = (lab > 0).astype(np.uint8)

        # Biological rule: ER and Ribo are mutually exclusive.
        # If a ribosome voxel exists, ER target is forced to 0 there.
        if self.valid[ER_IDX] > 0 and self.valid[RIBO_IDX] > 0:
            overlap = (self.labels[ER_IDX] > 0) & (self.labels[RIBO_IDX] > 0)
            n_overlap = int(overlap.sum())
            if n_overlap > 0:
                print(f"  ER/Ribo overlap in targets removed: {n_overlap} voxels")
                self.labels[ER_IDX][overlap] = 0

        self.union = (self.labels.sum(axis=0) > 0).astype(np.uint8)

        self.pos_flat = {}
        for ci, ch in enumerate(CHANNELS):
            if self.valid[ci] > 0:
                idx = np.flatnonzero(self.labels[ci].reshape(-1))
                self.pos_flat[ch] = idx.astype(np.int64)
            else:
                self.pos_flat[ch] = np.array([], dtype=np.int64)

        print(f"Loaded {self.name}: EM {self.em.shape}")
        for ci, ch in enumerate(CHANNELS):
            status = "present" if self.valid[ci] > 0 else "missing"
            print(f"  {ch}: {status}, voxels={int(self.labels[ci].sum())}")

# ============================================================
# DATA-DERIVED WEIGHTS
# ============================================================
def derive_channel_weights_from_tomos(tomos):
    pos = np.zeros(N_CLASSES, dtype=np.float64)
    valid_vox = np.zeros(N_CLASSES, dtype=np.float64)

    for tomo in tomos:
        n_vox = float(np.prod(tomo.em.shape))
        for ci, ch in enumerate(CHANNELS):
            if tomo.valid[ci] > 0:
                pos[ci] += float(tomo.labels[ci].sum())
                valid_vox[ci] += n_vox

    fg_frac = np.zeros(N_CLASSES, dtype=np.float64)
    for ci in range(N_CLASSES):
        if valid_vox[ci] > 0:
            fg_frac[ci] = pos[ci] / valid_vox[ci]

    nonzero = fg_frac[(fg_frac > 0) & (valid_vox > 0)]
    if len(nonzero) == 0:
        raise RuntimeError("No positive label voxels found in training data.")

    median_fg = np.median(nonzero)

    channel_weights = np.ones(N_CLASSES, dtype=np.float32)
    bce_pos_weights = np.ones(N_CLASSES, dtype=np.float32)

    print("\nDerived weights from training tomograms:")
    for ci, ch in enumerate(CHANNELS):
        if valid_vox[ci] == 0:
            channel_weights[ci] = 0.0
            bce_pos_weights[ci] = 1.0
            print(f"  {ch}: missing from all training tomograms, channel_weight=0, bce_pos_weight=1")
            continue

        if pos[ci] <= 0:
            channel_weights[ci] = CLASS_WEIGHT_CLIP[1]
            bce_pos_weights[ci] = BCE_POS_WEIGHT_CLIP[1]
        else:
            channel_weights[ci] = np.sqrt(median_fg / fg_frac[ci])
            channel_weights[ci] = np.clip(channel_weights[ci], CLASS_WEIGHT_CLIP[0], CLASS_WEIGHT_CLIP[1])

            neg = valid_vox[ci] - pos[ci]
            bce_pos_weights[ci] = neg / max(pos[ci], 1.0)
            bce_pos_weights[ci] = np.clip(bce_pos_weights[ci], BCE_POS_WEIGHT_CLIP[0], BCE_POS_WEIGHT_CLIP[1])

        print(
            f"  {ch}: pos={int(pos[ci])}, valid_vox={int(valid_vox[ci])}, "
            f"fg_frac={fg_frac[ci]:.8f}, channel_weight={channel_weights[ci]:.4f}, "
            f"bce_pos_weight={bce_pos_weights[ci]:.4f}"
        )

    weights_dict = {
        "channels": CHANNELS,
        "pos_voxels": pos.tolist(),
        "valid_voxels": valid_vox.tolist(),
        "fg_fraction": fg_frac.tolist(),
        "channel_weights": channel_weights.tolist(),
        "bce_pos_weights": bce_pos_weights.tolist(),
        "class_weight_clip": list(CLASS_WEIGHT_CLIP),
        "bce_pos_weight_clip": list(BCE_POS_WEIGHT_CLIP),
        "weights_from": "training_tomograms_only",
    }

    return torch.tensor(channel_weights, dtype=torch.float32), torch.tensor(bce_pos_weights, dtype=torch.float32), weights_dict

# ============================================================
# AUXILIARY TARGETS
# ============================================================
def make_boundary_target(labels):
    labels = labels.astype(bool)
    boundary = np.zeros_like(labels, dtype=bool)

    diff = labels[:, 1:, :, :] != labels[:, :-1, :, :]
    boundary[:, 1:, :, :] |= diff
    boundary[:, :-1, :, :] |= diff

    diff = labels[:, :, 1:, :] != labels[:, :, :-1, :]
    boundary[:, :, 1:, :] |= diff
    boundary[:, :, :-1, :] |= diff

    diff = labels[:, :, :, 1:] != labels[:, :, :, :-1]
    boundary[:, :, :, 1:] |= diff
    boundary[:, :, :, :-1] |= diff

    return boundary.astype(np.float32)

def make_affinity_target(labels):
    selected = labels[AFFINITY_CLASS_IDXS].astype(bool)
    C, Z, Y, X = selected.shape
    aff = np.zeros((C, 3, Z, Y, X), dtype=np.float32)

    aff[:, 0, 1:, :, :] = selected[:, 1:, :, :] & selected[:, :-1, :, :]
    aff[:, 1, :, 1:, :] = selected[:, :, 1:, :] & selected[:, :, :-1, :]
    aff[:, 2, :, :, 1:] = selected[:, :, :, 1:] & selected[:, :, :, :-1]

    return aff.reshape(C * 3, Z, Y, X).astype(np.float32)

# ============================================================
# BALANCED TRAINING PATCHES
# ============================================================
def make_exact_category_sequence(n_patches, probs_dict, rng):
    names = list(probs_dict.keys())
    probs = np.array([probs_dict[k] for k in names], dtype=np.float64)
    probs = probs / probs.sum()

    raw = probs * n_patches
    counts = np.floor(raw).astype(int)
    remainder = int(n_patches - counts.sum())

    frac_order = np.argsort(-(raw - counts))
    for i in frac_order[:remainder]:
        counts[i] += 1

    seq = []
    for name, count in zip(names, counts):
        seq.extend([name] * int(count))

    rng.shuffle(seq)
    return seq

class BalancedEMTomoPatchDataset(Dataset):
    def __init__(self, tomos, samples_per_epoch, patch_size, augment, seed):
        self.tomos = list(tomos)
        self.samples_per_epoch = samples_per_epoch
        self.patch_size = patch_size
        self.augment = augment
        self.seed = seed
        self.groups = {i: [tomo] for i, tomo in enumerate(self.tomos)}
        self.group_keys = list(self.groups)
        self.set_epoch(0)

    def set_epoch(self, epoch):
        rng = np.random.default_rng(self.seed + int(epoch))
        equal = {k: 1.0 for k in self.group_keys}
        self.group_sequence = make_exact_category_sequence(self.samples_per_epoch, equal, rng)
        self.category_sequence = make_exact_category_sequence(self.samples_per_epoch, CLASS_SAMPLING, rng)

    def __len__(self):
        return self.samples_per_epoch

    def choose_center(self, tomo, choice):
        Z, Y, X = tomo.em.shape
        if choice in CHANNELS:
            pos = tomo.pos_flat[choice]
            if len(pos) > 0:
                flat = int(pos[np.random.randint(0, len(pos))])
                z = flat // (Y * X)
                rem = flat % (Y * X)
                y = rem // X
                x = rem % X
                return z, y, x

        if choice == "background":
            for _ in range(200):
                z = np.random.randint(0, Z)
                y = np.random.randint(0, Y)
                x = np.random.randint(0, X)
                if tomo.union[z, y, x] == 0:
                    return z, y, x

        z = np.random.randint(0, Z)
        y = np.random.randint(0, Y)
        x = np.random.randint(0, X)
        return z, y, x

    def choose_train_sample(self, idx):
        group = self.group_sequence[idx]
        choice = self.category_sequence[idx]
        tomos = self.groups[group]
        if choice in CHANNELS:
            eligible = [t for t in tomos if len(t.pos_flat[choice]) > 0]
            if eligible:
                tomos = eligible
        tomo = tomos[np.random.randint(0, len(tomos))]
        return tomo, self.choose_center(tomo, choice)

    def augment_patch(self, em, labels):
        if np.random.rand() < 0.5:
            em = np.flip(em, axis=0)
            labels = np.flip(labels, axis=1)

        if np.random.rand() < 0.5:
            em = np.flip(em, axis=1)
            labels = np.flip(labels, axis=2)

        if np.random.rand() < 0.5:
            em = np.flip(em, axis=2)
            labels = np.flip(labels, axis=3)

        k = np.random.randint(0, 4)
        if k > 0:
            em = np.rot90(em, k=k, axes=(1, 2))
            labels = np.rot90(labels, k=k, axes=(2, 3))

        if np.random.rand() < 0.75:
            scale = np.random.uniform(0.85, 1.15)
            shift = np.random.uniform(-0.25, 0.25)
            em = em * scale + shift

        if np.random.rand() < 0.50:
            noise = np.random.normal(0, np.random.uniform(0.02, 0.12), size=em.shape).astype(np.float32)
            em = em + noise

        if np.random.rand() < 0.20:
            z0 = np.random.randint(0, em.shape[0])
            max_slab = max(1, min(4, em.shape[0] - z0))
            slab = np.random.randint(1, max_slab + 1)
            em[z0:z0 + slab] = 0.0

        em = np.clip(em, -6.0, 6.0)

        return np.ascontiguousarray(em), np.ascontiguousarray(labels)

    def __getitem__(self, idx):
        tomo, center = self.choose_train_sample(idx)

        em = crop_pad_3d(tomo.em, center, self.patch_size, fill_value=0)
        labels = crop_pad_3d(tomo.labels, center, self.patch_size, fill_value=0)

        if self.augment:
            em, labels = self.augment_patch(em, labels)
        else:
            em = np.ascontiguousarray(em)
            labels = np.ascontiguousarray(labels)

        boundary = make_boundary_target(labels)
        affinity = make_affinity_target(labels)

        return {
            "image": torch.from_numpy(em[None].astype(np.float32)),
            "target": torch.from_numpy(labels.astype(np.float32)),
            "valid": torch.from_numpy(tomo.valid.astype(np.float32)),
            "boundary": torch.from_numpy(boundary.astype(np.float32)),
            "affinity": torch.from_numpy(affinity.astype(np.float32)),
        }

# ============================================================
# HIERARCHY-CONSTRAINED RESUNET3D + BOTTLENECK ATTENTION
# ============================================================
def group_norm(ch):
    g = min(8, ch)
    while ch % g != 0:
        g -= 1
    return nn.GroupNorm(g, ch)

class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=(1, 1, 1)):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm1 = group_norm(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = group_norm(out_ch)
        self.act = nn.SiLU(inplace=True)

        if in_ch != out_ch or stride != (1, 1, 1):
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                group_norm(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = self.act(x + residual)
        return x

class BottleneckSelfAttention3D(nn.Module):
    def __init__(self, ch, heads=4, qk_dim=96, ffn_mult=2):
        super().__init__()
        if ch % heads != 0:
            raise ValueError(f"channels={ch} must be divisible by heads={heads}")
        if qk_dim % heads != 0:
            raise ValueError(f"qk_dim={qk_dim} must be divisible by heads={heads}")

        self.ch = ch
        self.heads = heads
        self.qk_dim = qk_dim
        self.qk_head_dim = qk_dim // heads
        self.v_head_dim = ch // heads
        self.scale = self.qk_head_dim ** -0.5

        self.norm_attn = group_norm(ch)
        self.q = nn.Conv3d(ch, qk_dim, 1, bias=False)
        self.k = nn.Conv3d(ch, qk_dim, 1, bias=False)
        self.v = nn.Conv3d(ch, ch, 1, bias=False)
        self.proj = nn.Conv3d(ch, ch, 1, bias=False)

        hidden = ch * ffn_mult
        self.ffn = nn.Sequential(
            group_norm(ch),
            nn.Conv3d(ch, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden, ch, 1),
        )

    def forward(self, x):
        B, C, Z, Y, X = x.shape
        N = Z * Y * X

        residual = x
        h = self.norm_attn(x)

        q = self.q(h).view(B, self.heads, self.qk_head_dim, N).transpose(2, 3)
        k = self.k(h).view(B, self.heads, self.qk_head_dim, N)
        v = self.v(h).view(B, self.heads, self.v_head_dim, N).transpose(2, 3)

        attn = torch.matmul(q, k) * self.scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(2, 3).contiguous().view(B, C, Z, Y, X)

        x = residual + self.proj(out)
        x = x + self.ffn(x)
        return x

def match_shape(x, ref):
    _, _, Z, Y, X = x.shape
    _, _, RZ, RY, RX = ref.shape

    dz, dy, dx = RZ - Z, RY - Y, RX - X

    pad = [
        max(dx // 2, 0), max(dx - dx // 2, 0),
        max(dy // 2, 0), max(dy - dy // 2, 0),
        max(dz // 2, 0), max(dz - dz // 2, 0),
    ]

    if any(v > 0 for v in pad):
        x = F.pad(x, pad)

    _, _, Z, Y, X = x.shape
    z0 = max((Z - RZ) // 2, 0)
    y0 = max((Y - RY) // 2, 0)
    x0 = max((X - RX) // 2, 0)

    return x[:, :, z0:z0 + RZ, y0:y0 + RY, x0:x0 + RX]

class HierarchyConstrainedAttentionResUNet3D(nn.Module):
    def __init__(self, in_ch=1, base=24, n_classes=5, n_affinity_classes=4):
        super().__init__()

        self.enc0 = ResBlock3D(in_ch, base)
        self.enc1 = ResBlock3D(base, base * 2, stride=(1, 2, 2))
        self.enc2 = ResBlock3D(base * 2, base * 4, stride=(2, 2, 2))
        self.enc3 = ResBlock3D(base * 4, base * 8, stride=(2, 2, 2))
        self.bottleneck = ResBlock3D(base * 8, base * 16, stride=(2, 2, 2))

        if USE_BOTTLENECK_ATTENTION:
            self.attention = BottleneckSelfAttention3D(
                ch=base * 16,
                heads=ATTENTION_HEADS,
                qk_dim=ATTENTION_QK_DIM,
                ffn_mult=ATTENTION_FFN_MULT,
            )
        else:
            self.attention = nn.Identity()

        self.up3 = nn.ConvTranspose3d(base * 16, base * 8, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dec3 = ResBlock3D(base * 16, base * 8)

        self.up2 = nn.ConvTranspose3d(base * 8, base * 4, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dec2 = ResBlock3D(base * 8, base * 4)

        self.up1 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dec1 = ResBlock3D(base * 4, base * 2)

        self.up0 = nn.ConvTranspose3d(base * 2, base, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec0 = ResBlock3D(base * 2, base)

        self.seg_head = nn.Conv3d(base, n_classes, 1)
        self.boundary_head = nn.Conv3d(base, n_classes, 1)
        self.affinity_head = nn.Conv3d(base, n_affinity_classes * 3, 1)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.bottleneck(e3)
        b = self.attention(b)

        x = self.up3(b)
        x = match_shape(x, e3)
        x = self.dec3(torch.cat([x, e3], dim=1))

        x = self.up2(x)
        x = match_shape(x, e2)
        x = self.dec2(torch.cat([x, e2], dim=1))

        x = self.up1(x)
        x = match_shape(x, e1)
        x = self.dec1(torch.cat([x, e1], dim=1))

        x = self.up0(x)
        x = match_shape(x, e0)
        x = self.dec0(torch.cat([x, e0], dim=1))

        raw_logits = self.seg_head(x)
        raw_probs = torch.sigmoid(raw_logits)

        p_omm = raw_probs[:, OMM_IDX:OMM_IDX + 1]

        outside_omm = 1.0 - p_omm

        p_ibm = p_omm * raw_probs[:, IBM_IDX:IBM_IDX + 1]
        p_ims = p_omm * raw_probs[:, IMS_IDX:IMS_IDX + 1]

        raw_er = raw_probs[:, ER_IDX:ER_IDX + 1]
        raw_ribo = raw_probs[:, RIBO_IDX:RIBO_IDX + 1]

        p_ribo = outside_omm * raw_ribo

        # ER/Ribo exclusivity:
        # ER is outside mitochondria and suppressed wherever Ribo probability is high.
        p_er = outside_omm * raw_er * (1.0 - raw_ribo)

        seg_probs = torch.cat([p_omm, p_ibm, p_ims, p_er, p_ribo], dim=1)

        return {
            "seg_probs": seg_probs,
            "raw_probs": raw_probs,
            "raw_logits": raw_logits,
            "boundary_logits": self.boundary_head(x),
            "affinity_logits": self.affinity_head(x),
        }

# ============================================================
# LOSSES
# ============================================================
def masked_bce_probs(probs, target, valid, channel_weights, bce_pos_weights):
    eps = 1e-6
    probs = probs.clamp(eps, 1.0 - eps)

    pos_w = bce_pos_weights[None, :, None, None, None].to(probs.device)
    ch_w = channel_weights[None, :, None, None, None].to(probs.device)
    valid_w = valid[:, :, None, None, None].to(probs.device)

    loss = -(pos_w * target * torch.log(probs) + (1.0 - target) * torch.log(1.0 - probs))
    w = ch_w * valid_w

    return (loss * w).sum() / w.expand_as(loss).sum().clamp_min(eps)

def masked_dice_loss(probs, target, valid, channel_weights):
    eps = 1e-6
    dims = (2, 3, 4)

    inter = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)

    dice_loss = 1.0 - (2.0 * inter + eps) / (denom + eps)

    w = channel_weights[None, :].to(probs.device) * valid.to(probs.device)
    return (dice_loss * w).sum() / w.sum().clamp_min(eps)

def masked_bce_logits(logits, target, valid, pos_weight_value):
    eps = 1e-6

    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pos_w = torch.where(target > 0.5, torch.full_like(target, float(pos_weight_value)), torch.ones_like(target))
    loss = loss * pos_w

    w = valid[:, :, None, None, None].to(logits.device)
    return (loss * w).sum() / w.expand_as(loss).sum().clamp_min(eps)

def hierarchy_loss(raw_probs):
    p_omm = raw_probs[:, OMM_IDX:OMM_IDX + 1]
    p_ibm_raw = raw_probs[:, IBM_IDX:IBM_IDX + 1]
    p_ims_raw = raw_probs[:, IMS_IDX:IMS_IDX + 1]
    p_er_raw = raw_probs[:, ER_IDX:ER_IDX + 1]
    p_ribo_raw = raw_probs[:, RIBO_IDX:RIBO_IDX + 1]

    ibm_outside = F.relu(p_ibm_raw - p_omm).pow(2).mean()
    ims_outside = F.relu(p_ims_raw - p_omm).pow(2).mean()
    er_inside = F.relu(p_er_raw + p_omm - 1.0).pow(2).mean()
    ribo_inside = F.relu(p_ribo_raw + p_omm - 1.0).pow(2).mean()

    # ER and ribosome are biologically exclusive.
    er_ribo_overlap = F.relu(p_er_raw + p_ribo_raw - 1.0).pow(2).mean()

    return ibm_outside + ims_outside + er_inside + ribo_inside + LOSS_W_ER_RIBO_EXCLUSIVE * er_ribo_overlap

def total_loss(outputs, target, valid, boundary, affinity, channel_weights, bce_pos_weights):
    seg_probs = outputs["seg_probs"]
    raw_probs = outputs["raw_probs"]

    bce = masked_bce_probs(seg_probs, target, valid, channel_weights, bce_pos_weights)
    dice = masked_dice_loss(seg_probs, target, valid, channel_weights)

    boundary_loss = masked_bce_logits(outputs["boundary_logits"], boundary, valid, BOUNDARY_POS_WEIGHT)

    aff_valid = valid[:, AFFINITY_CLASS_IDXS].repeat_interleave(3, dim=1)
    affinity_loss = masked_bce_logits(outputs["affinity_logits"], affinity, aff_valid, AFFINITY_POS_WEIGHT)

    hier = hierarchy_loss(raw_probs)

    loss = (
        LOSS_W_BCE * bce +
        LOSS_W_DICE * dice +
        LOSS_W_BOUNDARY * boundary_loss +
        LOSS_W_AFFINITY * affinity_loss +
        LOSS_W_HIER * hier
    )

    parts = {
        "loss": float(loss.detach().cpu()),
        "bce": float(bce.detach().cpu()),
        "dice": float(dice.detach().cpu()),
        "boundary": float(boundary_loss.detach().cpu()),
        "affinity": float(affinity_loss.detach().cpu()),
        "hier": float(hier.detach().cpu()),
    }

    return loss, parts

@torch.no_grad()
def dice_per_channel(probs, target, valid):
    eps = 1e-6
    dims = (2, 3, 4)

    pred_bin = (probs > 0.5).float()
    inter = (pred_bin * target).sum(dim=dims)
    denom = pred_bin.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * inter + eps) / (denom + eps)

    dice = dice * valid

    out = []
    for ci in range(N_CLASSES):
        if valid[:, ci].sum() > 0:
            out.append(float((dice[:, ci].sum() / valid[:, ci].sum()).detach().cpu()))
        else:
            out.append(float("nan"))
    return out

def mean_valid_dice(dice_values):
    vals = np.array(dice_values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan")
    return float(vals.mean())

# ============================================================
# COMPLETE-TOMOGRAM EVALUATION
# ============================================================
def get_starts(dim, patch, stride):
    if dim <= patch:
        return [0]
    starts = list(range(0, dim - patch + 1, stride))
    if starts[-1] != dim - patch:
        starts.append(dim - patch)
    return starts

def extract_patch_start(arr, z0, y0, x0, patch_size):
    pz, py, px = patch_size
    Z, Y, X = arr.shape
    z1, y1, x1 = min(z0 + pz, Z), min(y0 + py, Y), min(x0 + px, X)
    dz, dy, dx = z1 - z0, y1 - y0, x1 - x0
    patch = np.zeros((pz, py, px), dtype=arr.dtype)
    patch[:dz, :dy, :dx] = arr[z0:z1, y0:y1, x0:x1]
    return patch, (z0, z1, y0, y1, x0, x1), (dz, dy, dx)

def make_blend_weight(patch_size):
    def one_axis(n):
        if n <= 1:
            return np.ones((n,), dtype=np.float32)
        return np.maximum(np.hanning(n).astype(np.float32), 0.05)
    wz, wy, wx = [one_axis(n) for n in patch_size]
    w = wz[:, None, None] * wy[None, :, None] * wx[None, None, :]
    return np.maximum(w / max(float(w.max()), 1e-6), 0.01).astype(np.float32)

@torch.no_grad()
def infer_complete_tomogram(model, em, device):
    Z, Y, X = em.shape
    z_starts = get_starts(Z, PATCH_SIZE[0], INFER_STRIDE[0])
    y_starts = get_starts(Y, PATCH_SIZE[1], INFER_STRIDE[1])
    x_starts = get_starts(X, PATCH_SIZE[2], INFER_STRIDE[2])
    coords = [(z, y, x) for z in z_starts for y in y_starts for x in x_starts]
    prob_sum = np.zeros((N_CLASSES, Z, Y, X), dtype=np.float32)
    weight_sum = np.zeros((Z, Y, X), dtype=np.float32)
    blend = make_blend_weight(PATCH_SIZE)
    iterator = tqdm(coords, desc="  complete-volume patches", leave=False) if tqdm is not None else coords
    model.eval()
    for z0, y0, x0 in iterator:
        patch, slc, valid_shape = extract_patch_start(em, z0, y0, x0, PATCH_SIZE)
        za, z1, ya, y1, xa, x1 = slc
        dz, dy, dx = valid_shape
        x = torch.from_numpy(patch[None, None].astype(np.float32)).to(device)
        out = model(x)["seg_probs"][0].detach().cpu().numpy().astype(np.float32)
        w = blend[:dz, :dy, :dx]
        prob_sum[:, za:z1, ya:y1, xa:x1] += out[:, :dz, :dy, :dx] * w[None]
        weight_sum[za:z1, ya:y1, xa:x1] += w
        del x
    probs = prob_sum / np.maximum(weight_sum[None], 1e-6)
    probs[ER_IDX] *= 1.0 - probs[RIBO_IDX]
    probs[IBM_IDX] = np.minimum(probs[IBM_IDX], probs[OMM_IDX])
    probs[IMS_IDX] = np.minimum(probs[IMS_IDX], probs[OMM_IDX])
    probs[ER_IDX] *= 1.0 - probs[OMM_IDX]
    probs[RIBO_IDX] *= 1.0 - probs[OMM_IDX]
    return np.clip(probs, 0.0, 1.0).astype(np.float32)

def probs_to_masks(probs):
    masks = np.stack([probs[ci] >= THRESHOLDS[ch] for ci, ch in enumerate(CHANNELS)])
    masks[IBM_IDX] &= masks[OMM_IDX]
    masks[IMS_IDX] &= masks[OMM_IDX]
    masks[ER_IDX] &= ~masks[OMM_IDX]
    masks[RIBO_IDX] &= ~masks[OMM_IDX]
    masks[ER_IDX] &= ~masks[RIBO_IDX]
    return masks

def summarize_metric_rows(rows):
    summary = []
    for ch in CHANNELS:
        ch_rows = [r for r in rows if r["channel"] == ch]
        row = {"channel": ch, "n_tomograms": len(ch_rows)}
        for metric in ("dice", "precision", "recall"):
            vals = np.array([r[metric] for r in ch_rows], dtype=np.float64)
            row[f"mean_{metric}"] = float(vals.mean()) if len(vals) else float("nan")
            row[f"sd_{metric}"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0 if len(vals) == 1 else float("nan")
        summary.append(row)
    mean_class_dice = mean_valid_dice([r["mean_dice"] for r in summary])
    return summary, mean_class_dice

@torch.no_grad()
def evaluate_complete_tomograms(model, tomos, device, split_name):
    rows = []
    for i, tomo in enumerate(tomos, 1):
        print(f"\n{split_name} tomogram {i}/{len(tomos)}: {tomo.name}")
        probs = infer_complete_tomogram(model, tomo.em, device)
        pred = probs_to_masks(probs)
        for ci, ch in enumerate(CHANNELS):
            if tomo.valid[ci] <= 0:
                continue
            target = tomo.labels[ci].astype(bool)
            p = pred[ci]
            tp = int(np.logical_and(p, target).sum())
            fp = int(np.logical_and(p, ~target).sum())
            fn = int(np.logical_and(~p, target).sum())
            rows.append({
                "split": split_name, "tomogram": tomo.name, "channel": ch,
                "dice": (2.0 * tp) / max(2 * tp + fp + fn, 1),
                "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                "tp": tp, "fp": fp, "fn": fn,
            })
        del probs, pred
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
    summary, mean_class_dice = summarize_metric_rows(rows)
    return rows, summary, mean_class_dice

def save_metric_outputs(rows, summary, prefix):
    save_log_csv(rows, OUT_DIR / f"{prefix}_per_tomogram.csv")
    save_log_csv(summary, OUT_DIR / f"{prefix}_summary.csv")
    with open(OUT_DIR / f"{prefix}.json", "w") as f:
        json.dump({"thresholds": THRESHOLDS, "per_tomogram": rows, "summary": summary}, f, indent=2)

def print_metric_summary(summary, title):
    print(f"\n{title}")
    for r in summary:
        print(
            f"  {r['channel']}: Dice={r['mean_dice']:.4f} +/- {r['sd_dice']:.4f}, "
            f"precision={r['mean_precision']:.4f} +/- {r['sd_precision']:.4f}, "
            f"recall={r['mean_recall']:.4f} +/- {r['sd_recall']:.4f}, n={r['n_tomograms']}"
        )

# ============================================================
# FIXED QUALITATIVE VISUALIZATION ON FIRST TEST TOMOGRAM
# ============================================================
def integral_image_2d(mask):
    ii = np.pad(mask.astype(np.int32), ((1, 0), (1, 0)), mode="constant")
    ii = ii.cumsum(axis=0).cumsum(axis=1)
    return ii

def rect_count(ii, y0, y1, x0, x1, H, W):
    y0 = max(0, min(H, int(y0)))
    y1 = max(0, min(H, int(y1)))
    x0 = max(0, min(W, int(x0)))
    x1 = max(0, min(W, int(x1)))
    if y1 <= y0 or x1 <= x0:
        return 0
    return int(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])

def choose_visualization_center_all_labels(viz_tomo):
    Z, Y, X = viz_tomo.em.shape
    pz, py, px = PATCH_SIZE
    z = min(max(VIZ_SLICE_INDEX, 0), Z - 1)

    labels2d = viz_tomo.labels[:, z]
    required = [ci for ci, ch in enumerate(CHANNELS) if viz_tomo.valid[ci] > 0]
    present_on_slice = [ci for ci in required if labels2d[ci].sum() > 0]

    missing_on_slice = [CHANNELS[ci] for ci in required if ci not in present_on_slice]
    if len(missing_on_slice) > 0:
        print(f"WARNING: slice {z + 1} does not contain these labels: {missing_on_slice}")
        print("Visualization will choose the best patch for labels that are present on that slice.")

    if len(present_on_slice) == 0:
        print(f"WARNING: no labels found on slice {z + 1}; using image center for visualization.")
        return (z, Y // 2, X // 2), [], []

    rng = np.random.default_rng(SEED)
    integral_maps = {ci: integral_image_2d(labels2d[ci] > 0) for ci in present_on_slice}

    candidates = []

    y_grid = list(range(0, Y, VIZ_GRID_STEP))
    x_grid = list(range(0, X, VIZ_GRID_STEP))
    if (Y - 1) not in y_grid:
        y_grid.append(Y - 1)
    if (X - 1) not in x_grid:
        x_grid.append(X - 1)

    for yy in y_grid:
        for xx in x_grid:
            candidates.append((yy, xx))

    class_centroids = []
    for ci in present_on_slice:
        ys, xs = np.where(labels2d[ci] > 0)
        if len(ys) == 0:
            continue

        cy = int(np.round(np.median(ys)))
        cx = int(np.round(np.median(xs)))
        candidates.append((cy, cx))
        class_centroids.append((cy, cx))

        n_sample = min(VIZ_RANDOM_CANDIDATES_PER_CLASS, len(ys))
        sample_idx = rng.choice(len(ys), size=n_sample, replace=False)
        for si in sample_idx:
            candidates.append((int(ys[si]), int(xs[si])))

    if len(class_centroids) > 0:
        candidates.append((
            int(np.round(np.mean([c[0] for c in class_centroids]))),
            int(np.round(np.mean([c[1] for c in class_centroids]))),
        ))

    candidates = list(dict.fromkeys(candidates))

    best = None
    best_score = None
    best_counts = None

    for cy, cx in candidates:
        y0 = int(round(cy - py // 2))
        y1 = y0 + py
        x0 = int(round(cx - px // 2))
        x1 = x0 + px

        counts = []
        for ci in present_on_slice:
            counts.append(rect_count(integral_maps[ci], y0, y1, x0, x1, Y, X))

        n_hit = sum(c > 0 for c in counts)
        min_count = min(counts) if len(counts) else 0
        log_sum = float(np.sum(np.log1p(counts)))
        score = (n_hit, min_count, log_sum)

        if best_score is None or score > best_score:
            best_score = score
            best = (cy, cx)
            best_counts = counts

    hit_channels = [CHANNELS[ci] for ci, c in zip(present_on_slice, best_counts) if c > 0]
    missed_channels = [CHANNELS[ci] for ci, c in zip(present_on_slice, best_counts) if c == 0]

    print(
        f"Visualization patch on {viz_tomo.name} slice {z + 1}: "
        f"center ZYX=({z}, {best[0]}, {best[1]}), "
        f"labels in patch={hit_channels}"
    )

    if len(missed_channels) > 0:
        print(f"WARNING: best visualization patch still misses: {missed_channels}")

    return (z, best[0], best[1]), hit_channels, missed_channels + missing_on_slice

@torch.no_grad()
def save_test_visualization(model, viz_tomo, device, epoch, out_dir, viz_center=None, viz_hit_channels=None, viz_missed_channels=None):
    if viz_tomo is None:
        return

    model.eval()

    if viz_center is None:
        viz_center, viz_hit_channels, viz_missed_channels = choose_visualization_center_all_labels(viz_tomo)

    z, cy, cx = viz_center

    em = crop_pad_3d(viz_tomo.em, viz_center, PATCH_SIZE, fill_value=0)
    labels = crop_pad_3d(viz_tomo.labels, viz_center, PATCH_SIZE, fill_value=0)

    local_z = PATCH_SIZE[0] // 2

    x = torch.from_numpy(em[None, None].astype(np.float32)).to(device)
    probs = model(x)["seg_probs"][0].detach().cpu().numpy()

    em2d = em[local_z]
    gt2d = labels[:, local_z]
    pr2d = probs[:, local_z]

    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 6, figsize=(18, 6))

    hit_text = ", ".join(viz_hit_channels or [])
    missed_text = ", ".join(viz_missed_channels or [])

    axes[0, 0].imshow(em2d, cmap="gray")
    axes[0, 0].set_title(f"{viz_tomo.name} EM\nslice {z + 1}, y={cy}, x={cx}")
    axes[1, 0].axis("off")

    for ci, ch in enumerate(CHANNELS):
        axes[0, ci + 1].imshow(gt2d[ci], cmap="gray", vmin=0, vmax=1)
        axes[0, ci + 1].set_title(f"GT {ch}")

        axes[1, ci + 1].imshow(pr2d[ci], cmap="magma", vmin=0, vmax=1)
        axes[1, ci + 1].set_title(f"Pred prob {ch}")

    for ax in axes.ravel():
        ax.axis("off")

    title = f"Epoch {epoch:04d} | {viz_tomo.name} | labels in patch: {hit_text}"
    if missed_text:
        title += f" | missed/unavailable: {missed_text}"

    fig.suptitle(title, y=0.98, fontsize=10)
    fig.tight_layout()
    fig.savefig(vis_dir / f"epoch_{epoch:04d}_test_slice_{z + 1}_y{cy}_x{cx}.png", dpi=200)
    plt.close(fig)

# ============================================================
# TRAIN / VALIDATE
# ============================================================
def run_epoch(model, loader, optimizer, device, channel_weights, bce_pos_weights, train=True, epoch=0):
    model.train() if train else model.eval()

    mode = "train" if train else "val"
    total = {"loss": 0.0, "bce": 0.0, "dice": 0.0, "boundary": 0.0, "affinity": 0.0, "hier": 0.0}
    dice_sum = np.zeros(N_CLASSES, dtype=np.float64)
    dice_count = np.zeros(N_CLASSES, dtype=np.float64)

    channel_weights = channel_weights.to(device)
    bce_pos_weights = bce_pos_weights.to(device)

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc=f"Epoch {epoch:04d} {mode}", leave=False)

    for batch_i, batch in enumerate(iterator, start=1):
        image = batch["image"].to(device)
        target = batch["target"].to(device)
        valid = batch["valid"].to(device)
        boundary = batch["boundary"].to(device)
        affinity = batch["affinity"].to(device)

        with torch.set_grad_enabled(train):
            outputs = model(image)
            loss, parts = total_loss(outputs, target, valid, boundary, affinity, channel_weights, bce_pos_weights)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        for k in total:
            total[k] += parts[k]

        d = dice_per_channel(outputs["seg_probs"].detach(), target, valid)
        for ci, val in enumerate(d):
            if not np.isnan(val):
                dice_sum[ci] += val
                dice_count[ci] += 1

        if tqdm is not None:
            iterator.set_postfix({
                "loss": f"{parts['loss']:.4f}",
                "bce": f"{parts['bce']:.4f}",
                "dice": f"{parts['dice']:.4f}",
            })
        else:
            if batch_i == 1 or batch_i % 10 == 0 or batch_i == len(loader):
                print(f"  epoch {epoch:04d} {mode} batch {batch_i}/{len(loader)} loss={parts['loss']:.4f}")

    n = max(len(loader), 1)
    avg = {k: v / n for k, v in total.items()}

    avg_dice = np.full(N_CLASSES, np.nan, dtype=np.float64)
    for ci in range(N_CLASSES):
        if dice_count[ci] > 0:
            avg_dice[ci] = dice_sum[ci] / dice_count[ci]

    mean_dice = mean_valid_dice(avg_dice)

    return avg, avg_dice, mean_dice

def save_checkpoint(path, model, optimizer, epoch, val_mean_dice, channel_weights, bce_pos_weights):
    ckpt = {
        "model_name": "HierarchyConstrainedAttentionResUNet3D",
        "epoch": epoch,
        "val_loss": float("nan"),
        "val_mean_dice": val_mean_dice,
        "best_model_metric": BEST_MODEL_METRIC,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "channels": CHANNELS,
        "affinity_channels": AFFINITY_CHANNELS,
        "patch_size": PATCH_SIZE,
        "base_ch": BASE_CH,
        "use_bottleneck_attention": USE_BOTTLENECK_ATTENTION,
        "attention_heads": ATTENTION_HEADS,
        "attention_qk_dim": ATTENTION_QK_DIM,
        "attention_ffn_mult": ATTENTION_FFN_MULT,
        "validation_mode": VALIDATION_MODE,
        "inference_stride": INFER_STRIDE,
        "thresholds": THRESHOLDS,
        "balanced_sampling": "equal tomogram allocation",
        "er_ribo_exclusive": True,
        "loss_w_er_ribo_exclusive": LOSS_W_ER_RIBO_EXCLUSIVE,
        "channel_weights": channel_weights.detach().cpu(),
        "bce_pos_weights": bce_pos_weights.detach().cpu(),
    }
    torch.save(ckpt, path)

# ============================================================
# MAIN
# ============================================================
def train():
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"device: {device}")

    train_folders, validation_folders, test_folders = discover_split_folders()
    viz_folder = test_folders[0]

    split_info = {
        "model_name": "HierarchyConstrainedAttentionResUNet3D",
        "train": [str(p) for p in train_folders],
        "validation": [str(p) for p in validation_folders],
        "test": [str(p) for p in test_folders],
        "validation_mode": VALIDATION_MODE,
        "train_sampling": "equal tomogram allocation, then class-centered sampling",
        "train_samples_per_epoch": TRAIN_SAMPLES_PER_EPOCH,
        "patch_size": list(PATCH_SIZE),
        "inference_stride": list(INFER_STRIDE),
        "thresholds": THRESHOLDS,
        "test_visualization": str(viz_folder),
        "test_visualization_trigger": "new best complete-validation mean class Dice",
        "normalization": "whole_tomogram_zscore_then_crop_patches",
    }
    split_path = OUT_DIR / "split.json"
    if RESUME_FROM_LAST and RESUME_CHECKPOINT_PATH.exists() and split_path.exists():
        with open(split_path, "r") as f:
            previous_split = json.load(f)
        for key in ("train", "validation", "test"):
            if previous_split.get(key) != split_info[key]:
                raise RuntimeError(f"Cannot resume because the {key} split changed since the checkpoint was created.")
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2)

    for split_name, folders in (("TRAIN", train_folders), ("VALIDATION", validation_folders), ("TEST", test_folders)):
        print(f"\n{split_name} TOMOGRAMS ({len(folders)}):")
        for p in folders:
            print(f"  {p.name}")

    print("\nLoading training tomograms:")
    train_tomos = [TomogramData(f) for f in train_folders]
    print("\nLoading complete validation tomograms:")
    validation_tomos = [TomogramData(f) for f in validation_folders]
    print("\nLoading independent test tomograms:")
    test_tomos = [TomogramData(f) for f in test_folders]

    train_ds = BalancedEMTomoPatchDataset(
        tomos=train_tomos, samples_per_epoch=TRAIN_SAMPLES_PER_EPOCH,
        patch_size=PATCH_SIZE, augment=True, seed=SEED + 2000,
    )
    channel_weights, bce_pos_weights, weights_dict = derive_channel_weights_from_tomos(train_tomos)
    with open(OUT_DIR / "derived_weights.json", "w") as f:
        json.dump(weights_dict, f, indent=2)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=False,
    )

    viz_tomo = test_tomos[0]
    viz_center, viz_hit_channels, viz_missed_channels = choose_visualization_center_all_labels(viz_tomo)

    model = HierarchyConstrainedAttentionResUNet3D(
        in_ch=1, base=BASE_CH, n_classes=N_CLASSES,
        n_affinity_classes=N_AFFINITY_CLASSES,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nparameters: {n_params:,}")
    print(f"training tomograms: {len(train_tomos)}")
    print(f"complete validation tomograms: {len(validation_tomos)}")
    print(f"independent test tomograms: {len(test_tomos)}")
    print(f"training patches per epoch: {len(train_ds)}")
    print(f"best model metric: {BEST_MODEL_METRIC}")

    start_epoch = 1
    logs = []
    if RESUME_FROM_LAST and RESUME_CHECKPOINT_PATH.exists():
        print(f"\nResuming from checkpoint: {RESUME_CHECKPOINT_PATH}")
        ckpt = safe_torch_load(RESUME_CHECKPOINT_PATH)
        if ckpt.get("validation_mode") != VALIDATION_MODE:
            raise RuntimeError("Resume checkpoint was not created by the complete-tomogram validation workflow.")
        model.load_state_dict(ckpt["model_state"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if NEW_LR_AFTER_RESUME is not None:
            for group in optimizer.param_groups:
                group["lr"] = float(NEW_LR_AFTER_RESUME)
            print(f"  forced optimizer LR after resume: {NEW_LR_AFTER_RESUME}")
        else:
            print(f"  restored optimizer LR: {[g.get('lr') for g in optimizer.param_groups]}")
        start_epoch = int(ckpt["epoch"]) + 1
        channel_weights = ckpt.get("channel_weights", channel_weights).float()
        bce_pos_weights = ckpt.get("bce_pos_weights", bce_pos_weights).float()
        logs = load_training_log(OUT_DIR / "training_log.csv")
        logs = [r for r in logs if int(r.get("epoch", 0)) < start_epoch]
        print(f"  checkpoint epoch: {ckpt['epoch']}; next epoch: {start_epoch}")
    else:
        print("\nStarting a fresh training run.")

    best_metric, _ = best_values_from_logs(logs)
    if (OUT_DIR / "best.pt").exists():
        best_ckpt = safe_torch_load(OUT_DIR / "best.pt")
        best_metric = max(best_metric, float(best_ckpt.get("val_mean_dice", -float("inf"))))
    print(f"Best complete-validation mean class Dice before training: {best_metric}")

    if start_epoch > EPOCHS:
        print(f"Checkpoint is already at epoch {start_epoch - 1}; proceeding to final best-checkpoint test evaluation.")
    else:
        for epoch in range(start_epoch, EPOCHS + 1):
            print(f"\nStarting epoch {epoch:04d}/{EPOCHS}")
            t0 = time.time()
            train_ds.set_epoch(epoch)
            train_avg, train_dice, train_mean_dice = run_epoch(
                model=model, loader=train_loader, optimizer=optimizer, device=device,
                channel_weights=channel_weights, bce_pos_weights=bce_pos_weights,
                train=True, epoch=epoch,
            )
            val_rows, val_summary, val_mean_dice = evaluate_complete_tomograms(
                model, validation_tomos, device, "validation"
            )
            dt = time.time() - t0
            val_text = " | ".join(f"{r['channel']}={r['mean_dice']:.4f}" for r in val_summary)
            print(
                f"Epoch {epoch:04d} complete | train_loss={train_avg['loss']:.5f} | "
                f"complete_val_mean_dice={val_mean_dice:.4f} | {val_text} | {dt:.1f}s"
            )
            row = {
                "epoch": epoch, "seconds": dt,
                "train_loss": train_avg["loss"], "train_bce": train_avg["bce"],
                "train_dice_loss": train_avg["dice"], "train_boundary": train_avg["boundary"],
                "train_affinity": train_avg["affinity"], "train_hier": train_avg["hier"],
                "train_mean_dice": train_mean_dice, "val_mean_dice": val_mean_dice,
            }
            for ci, ch in enumerate(CHANNELS):
                row[f"train_dice_{ch}"] = train_dice[ci]
                s = next(r for r in val_summary if r["channel"] == ch)
                row[f"val_dice_{ch}"] = s["mean_dice"]
                row[f"val_precision_{ch}"] = s["mean_precision"]
                row[f"val_recall_{ch}"] = s["mean_recall"]
            logs.append(row)
            save_log_csv(logs, OUT_DIR / "training_log.csv")
            save_metric_outputs(val_rows, val_summary, "validation_latest")
            save_checkpoint(
                OUT_DIR / "last.pt", model, optimizer, epoch, val_mean_dice,
                channel_weights, bce_pos_weights,
            )
            if val_mean_dice > best_metric:
                best_metric = val_mean_dice
                save_checkpoint(
                    OUT_DIR / "best.pt", model, optimizer, epoch, val_mean_dice,
                    channel_weights, bce_pos_weights,
                )
                save_metric_outputs(val_rows, val_summary, "validation_best")
                save_test_visualization(
                    model, viz_tomo, device, epoch, OUT_DIR,
                    viz_center=viz_center, viz_hit_channels=viz_hit_channels,
                    viz_missed_channels=viz_missed_channels,
                )
                print(f"  saved best.pt and fixed test visualization, val_mean_dice={val_mean_dice:.4f}")

    best_path = OUT_DIR / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"No best checkpoint found: {best_path}")
    best_ckpt = safe_torch_load(best_path)
    model.load_state_dict(best_ckpt["model_state"], strict=True)
    print(f"\nFinal test evaluation using best.pt from epoch {best_ckpt['epoch']}")
    test_rows, test_summary, test_mean_dice = evaluate_complete_tomograms(model, test_tomos, device, "test")
    save_metric_outputs(test_rows, test_summary, "test_best")
    print_metric_summary(test_summary, "INDEPENDENT COMPLETE-TOMOGRAM TEST METRICS (mean +/- SD)")
    print(f"Mean test Dice across classes: {test_mean_dice:.4f}")
    print(f"\nTraining complete. Outputs saved to: {OUT_DIR}")

def main():
    global DATA_ROOT, TRAIN_ROOT, VALIDATION_ROOT, TEST_ROOT, OUT_DIR, RESUME_CHECKPOINT_PATH, EPOCHS
    parser = argparse.ArgumentParser(description="Train a hierarchy-constrained attention ResUNet3D on EM TIFF stacks.")
    parser.add_argument("--data-root", type=Path, help="Input directory containing train, validate, test; default: manual segmentations beside this script.")
    parser.add_argument("--out-dir", type=Path, help="Output directory; default: runs beside this script.")
    parser.add_argument("--epochs", type=int, help="Final epoch number, including resumed epochs; default: 200.")
    args = parser.parse_args()
    if len(CHANNELS) != 5 or len(set(CHANNELS)) != 5:
        parser.error("CHANNELS must contain five distinct TIFF names in the original biological role order.")
    if args.epochs is not None:
        if args.epochs < 1:
            parser.error("--epochs must be positive.")
        EPOCHS = args.epochs
    if args.data_root is not None:
        DATA_ROOT = args.data_root if args.data_root.is_absolute() else SCRIPT_DIR / args.data_root
        TRAIN_ROOT, VALIDATION_ROOT, TEST_ROOT = [DATA_ROOT / s for s in ("train", "validate", "test")]
    if args.out_dir is not None:
        OUT_DIR = args.out_dir if args.out_dir.is_absolute() else SCRIPT_DIR / args.out_dir
        RESUME_CHECKPOINT_PATH = OUT_DIR / "last.pt"
    train()

if __name__ == "__main__":
    main()
