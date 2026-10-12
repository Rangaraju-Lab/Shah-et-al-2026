from pathlib import Path
import argparse
import train as training
import time

import numpy as np
import tifffile as tiff

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ============================================================
# SETTINGS
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
INFERENCE_ROOT = SCRIPT_DIR / "inference images"
RUN_DIR = training.OUT_DIR

# best.pt is selected by mean complete-held-out-tomogram validation Dice.
BEST_VAL_CHECKPOINT = RUN_DIR / "best.pt"

OUTPUT_ROOT_FOLDER_NAME = "segmentations"

CHANNELS = list(training.CHANNELS)  # Filename/channel names are configured in train.py.
N_CLASSES = len(CHANNELS)

OMM_IDX = 0
IBM_IDX = 1
IMS_IDX = 2
ER_IDX = 3
RIBO_IDX = 4

AFFINITY_CHANNELS = CHANNELS[:4]
N_AFFINITY_CLASSES = len(AFFINITY_CHANNELS)

PATCH_SIZE = training.PATCH_SIZE  # Z, Y, X
INFER_STRIDE = training.INFER_STRIDE

BASE_CH = training.BASE_CH

USE_BOTTLENECK_ATTENTION = training.USE_BOTTLENECK_ATTENTION
ATTENTION_HEADS = training.ATTENTION_HEADS
ATTENTION_QK_DIM = training.ATTENTION_QK_DIM
ATTENTION_FFN_MULT = training.ATTENTION_FFN_MULT

THRESHOLDS = dict(training.THRESHOLDS)

# Prediction TIFF:
# uint16 = probability 0-1 scaled to 0-65535.
# float32 = raw probability 0-1.
PREDICTION_DTYPE = "uint16"       # "uint16" or "float32"

# Segmentation TIFF:
# uint8 masks are saved as 0/255.
SEGMENTATION_DTYPE = "uint8"      # "uint8" or "uint16"

OVERWRITE = True

VALID_EXTENSIONS = {".tif", ".tiff", ".TIF", ".TIFF"}

# ============================================================
# BASIC UTILITIES
# ============================================================
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

def read_zyx(path):
    arr = tiff.imread(str(path))
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        arr = arr[None, :, :]

    if arr.ndim != 3:
        raise ValueError(f"Expected 2D or 3D ZYX tif, got shape {arr.shape} for {path}")

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

def find_tifs(root):
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix in VALID_EXTENSIONS)

def get_starts(dim, patch, stride):
    if dim <= patch:
        return [0]

    starts = list(range(0, dim - patch + 1, stride))
    last = dim - patch

    if starts[-1] != last:
        starts.append(last)

    return starts

def extract_patch_start(arr, z0, y0, x0, patch_size):
    pz, py, px = patch_size
    Z, Y, X = arr.shape

    z1 = min(z0 + pz, Z)
    y1 = min(y0 + py, Y)
    x1 = min(x0 + px, X)

    dz = z1 - z0
    dy = y1 - y0
    dx = x1 - x0

    patch = np.zeros((pz, py, px), dtype=arr.dtype)
    patch[:dz, :dy, :dx] = arr[z0:z1, y0:y1, x0:x1]

    return patch, (z0, z1, y0, y1, x0, x1), (dz, dy, dx)

def make_blend_weight(patch_size):
    pz, py, px = patch_size

    def one_axis(n):
        if n <= 1:
            return np.ones((n,), dtype=np.float32)
        w = np.hanning(n).astype(np.float32)
        w = np.maximum(w, 0.05)
        return w

    wz = one_axis(pz)
    wy = one_axis(py)
    wx = one_axis(px)

    w = wz[:, None, None] * wy[None, :, None] * wx[None, None, :]
    w = w / max(float(w.max()), 1e-6)
    w = np.maximum(w, 0.01).astype(np.float32)
    return w

def probs_to_output_dtype(probs):
    probs = np.clip(probs, 0.0, 1.0)

    if PREDICTION_DTYPE.lower() == "float32":
        return probs.astype(np.float32)

    if PREDICTION_DTYPE.lower() == "uint16":
        return np.round(probs * 65535.0).astype(np.uint16)

    raise ValueError("PREDICTION_DTYPE must be 'uint16' or 'float32'")

def masks_to_output_dtype(masks):
    if SEGMENTATION_DTYPE.lower() == "uint8":
        return (masks.astype(np.uint8) * 255)

    if SEGMENTATION_DTYPE.lower() == "uint16":
        return (masks.astype(np.uint16) * 65535)

    raise ValueError("SEGMENTATION_DTYPE must be 'uint8' or 'uint16'")

def save_imagej_hyperstack_tzcyx(path, arr_czyx):
    """
    Input array is CZYX.
    Saved array is TZCYX:
        T = 1
        Z = slices
        C = channels
        Y = height
        X = width

    This opens in ImageJ/Fiji as:
        Channels = 5
        Slices = Z
        Frames = 1
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if arr_czyx.ndim != 4:
        raise ValueError(f"Expected CZYX array, got shape {arr_czyx.shape}")

    # CZYX -> ZCYX -> TZCYX
    arr_tzcyx = np.transpose(arr_czyx, (1, 0, 2, 3))[None, ...]
    arr_tzcyx = np.ascontiguousarray(arr_tzcyx)

    tiff.imwrite(
        str(path),
        arr_tzcyx,
        imagej=True,
        photometric="minisblack",
        metadata={
            "axes": "TZCYX",
            "mode": "composite",
        },
    )

# ============================================================
# HIERARCHY-CONSTRAINED ATTENTION RESUNET3D
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
    def __init__(self, in_ch=1, base=24, n_classes=5, n_affinity_classes=4, use_attention=True, attention_heads=4, attention_qk_dim=96, attention_ffn_mult=2):
        super().__init__()

        self.enc0 = ResBlock3D(in_ch, base)
        self.enc1 = ResBlock3D(base, base * 2, stride=(1, 2, 2))
        self.enc2 = ResBlock3D(base * 2, base * 4, stride=(2, 2, 2))
        self.enc3 = ResBlock3D(base * 4, base * 8, stride=(2, 2, 2))
        self.bottleneck = ResBlock3D(base * 8, base * 16, stride=(2, 2, 2))

        if use_attention:
            self.attention = BottleneckSelfAttention3D(
                ch=base * 16,
                heads=attention_heads,
                qk_dim=attention_qk_dim,
                ffn_mult=attention_ffn_mult,
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

        # ER/Ribo exclusivity.
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
# MODEL LOADING
# ============================================================
def build_model_from_checkpoint(ckpt, device):
    base = int(ckpt.get("base_ch", BASE_CH))
    use_attention = bool(ckpt.get("use_bottleneck_attention", USE_BOTTLENECK_ATTENTION))
    attention_heads = int(ckpt.get("attention_heads", ATTENTION_HEADS))
    attention_qk_dim = int(ckpt.get("attention_qk_dim", ATTENTION_QK_DIM))
    attention_ffn_mult = int(ckpt.get("attention_ffn_mult", ATTENTION_FFN_MULT))

    model = HierarchyConstrainedAttentionResUNet3D(
        in_ch=1,
        base=base,
        n_classes=N_CLASSES,
        n_affinity_classes=N_AFFINITY_CLASSES,
        use_attention=use_attention,
        attention_heads=attention_heads,
        attention_qk_dim=attention_qk_dim,
        attention_ffn_mult=attention_ffn_mult,
    ).to(device)

    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    return model

# ============================================================
# INFERENCE
# ============================================================
@torch.no_grad()
def infer_volume(model, em_norm, device, patch_size, stride):
    Z, Y, X = em_norm.shape
    pz, py, px = patch_size
    sz, sy, sx = stride

    z_starts = get_starts(Z, pz, sz)
    y_starts = get_starts(Y, py, sy)
    x_starts = get_starts(X, px, sx)

    total_patches = len(z_starts) * len(y_starts) * len(x_starts)

    print(f"  volume shape ZYX: {em_norm.shape}")
    print(f"  patch size: {patch_size}, stride: {stride}")
    print(f"  patches: {total_patches}")

    prob_sum = np.zeros((N_CLASSES, Z, Y, X), dtype=np.float32)
    weight_sum = np.zeros((Z, Y, X), dtype=np.float32)

    blend = make_blend_weight(patch_size)

    coords = []
    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                coords.append((z0, y0, x0))

    iterator = coords
    if tqdm is not None:
        iterator = tqdm(coords, desc="  inference patches", leave=False)

    for z0, y0, x0 in iterator:
        patch, slc, valid_shape = extract_patch_start(em_norm, z0, y0, x0, patch_size)
        z0, z1, y0, y1, x0, x1 = slc
        dz, dy, dx = valid_shape

        x = torch.from_numpy(patch[None, None].astype(np.float32)).to(device)

        out = model(x)["seg_probs"][0].detach().cpu().numpy().astype(np.float32)

        out_crop = out[:, :dz, :dy, :dx]
        w_crop = blend[:dz, :dy, :dx]

        prob_sum[:, z0:z1, y0:y1, x0:x1] += out_crop * w_crop[None, :, :, :]
        weight_sum[z0:z1, y0:y1, x0:x1] += w_crop

        del x

    probs = prob_sum / np.maximum(weight_sum[None, :, :, :], 1e-6)

    # Hard cleanup after blended inference.
    probs[ER_IDX] = probs[ER_IDX] * (1.0 - probs[RIBO_IDX])
    probs[IBM_IDX] = np.minimum(probs[IBM_IDX], probs[OMM_IDX])
    probs[IMS_IDX] = np.minimum(probs[IMS_IDX], probs[OMM_IDX])
    probs[ER_IDX] = probs[ER_IDX] * (1.0 - probs[OMM_IDX])
    probs[RIBO_IDX] = probs[RIBO_IDX] * (1.0 - probs[OMM_IDX])

    probs = np.clip(probs, 0.0, 1.0).astype(np.float32)

    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    return probs

def probs_to_masks(probs, thresholds):
    masks = np.zeros_like(probs, dtype=bool)

    for ci, ch in enumerate(CHANNELS):
        masks[ci] = probs[ci] >= float(thresholds[ch])

    # Hard binary cleanup.
    masks[IBM_IDX] = masks[IBM_IDX] & masks[OMM_IDX]
    masks[IMS_IDX] = masks[IMS_IDX] & masks[OMM_IDX]

    masks[ER_IDX] = masks[ER_IDX] & (~masks[OMM_IDX])
    masks[RIBO_IDX] = masks[RIBO_IDX] & (~masks[OMM_IDX])

    masks[ER_IDX] = masks[ER_IDX] & (~masks[RIBO_IDX])

    return masks

def save_outputs(input_path, probs, thresholds):
    out_dir = INFERENCE_ROOT / OUTPUT_ROOT_FOLDER_NAME / input_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir / f"{input_path.stem}_pred.tif"
    seg_path = out_dir / f"{input_path.stem}_seg.tif"

    if pred_path.exists() and seg_path.exists() and not OVERWRITE:
        print(f"  skipping, outputs exist: {out_dir}")
        return pred_path, seg_path, "skipped_exists"

    pred_out = probs_to_output_dtype(probs)      # CZYX
    masks = probs_to_masks(probs, thresholds)   # CZYX bool
    seg_out = masks_to_output_dtype(masks)      # CZYX

    save_imagej_hyperstack_tzcyx(pred_path, pred_out)
    save_imagej_hyperstack_tzcyx(seg_path, seg_out)

    return pred_path, seg_path, "done"

# ============================================================
# MAIN
# ============================================================
def main():
    global INFERENCE_ROOT, BEST_VAL_CHECKPOINT
    parser = argparse.ArgumentParser(description="Segment the EM TIFF stacks in inference images.")
    parser.add_argument("--checkpoint", type=Path, help="Trained checkpoint; default: runs/best.pt beside the scripts.")
    args = parser.parse_args()
    if args.checkpoint is not None:
        BEST_VAL_CHECKPOINT = args.checkpoint if args.checkpoint.is_absolute() else SCRIPT_DIR / args.checkpoint
    if not INFERENCE_ROOT.exists():
        raise FileNotFoundError(f"Could not find inference folder: {INFERENCE_ROOT}")

    if not BEST_VAL_CHECKPOINT.exists():
        raise FileNotFoundError(f"Could not find best validation checkpoint: {BEST_VAL_CHECKPOINT}")

    device = get_device()
    print(f"device: {device}")
    print(f"inference root: {INFERENCE_ROOT}")
    print(f"checkpoint: {BEST_VAL_CHECKPOINT}")

    ckpt = safe_torch_load(BEST_VAL_CHECKPOINT)

    if "model_state" not in ckpt:
        raise KeyError(f"Checkpoint does not contain model_state: {BEST_VAL_CHECKPOINT}")

    ckpt_channels = list(ckpt.get("channels", CHANNELS))
    ckpt_affinity_channels = list(ckpt.get("affinity_channels", AFFINITY_CHANNELS))
    if ckpt_channels != CHANNELS:
        raise RuntimeError(f"Checkpoint channels {ckpt_channels} do not match inference channels {CHANNELS}")
    if ckpt_affinity_channels != AFFINITY_CHANNELS:
        raise RuntimeError(
            f"Checkpoint affinity channels {ckpt_affinity_channels} do not match inference affinity channels {AFFINITY_CHANNELS}"
        )

    print(f"checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"checkpoint val_loss: {ckpt.get('val_loss', 'unknown')}")
    print(f"checkpoint val_mean_dice: {ckpt.get('val_mean_dice', 'unknown')}")
    print(f"checkpoint validation_mode: {ckpt.get('validation_mode', 'unknown')}")

    patch_size = tuple(int(v) for v in ckpt.get("patch_size", PATCH_SIZE))
    stride = tuple(int(v) for v in ckpt.get("inference_stride", INFER_STRIDE))
    thresholds = {ch: float(ckpt.get("thresholds", THRESHOLDS).get(ch, THRESHOLDS[ch])) for ch in CHANNELS}
    print(f"patch size: {patch_size}")
    print(f"inference stride: {stride}")
    print(f"thresholds: {thresholds}")

    model = build_model_from_checkpoint(ckpt, device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"parameters: {n_params:,}")

    tif_files = find_tifs(INFERENCE_ROOT)
    if len(tif_files) == 0:
        print(f"No tif files found under: {INFERENCE_ROOT}")
        return

    print(f"\nFound {len(tif_files)} tif file(s).")

    for file_i, tif_path in enumerate(tif_files, start=1):
        print("\n============================================================")
        print(f"[{file_i}/{len(tif_files)}] {tif_path}")

        t0 = time.time()

        try:
            em_raw = read_zyx(tif_path)
            em_norm = normalize_em_zscore(em_raw)

            probs = infer_volume(
                model=model,
                em_norm=em_norm,
                device=device,
                patch_size=patch_size,
                stride=stride,
            )

            pred_path, seg_path, status = save_outputs(tif_path, probs, thresholds)

            dt = time.time() - t0

            print(f"  status: {status}")
            print(f"  saved prediction: {pred_path}")
            print(f"  saved segmentation: {seg_path}")
            print(f"  done in {dt:.1f}s")

            del em_raw, em_norm, probs

            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            dt = time.time() - t0
            print(f"  FAILED after {dt:.1f}s: {repr(e)}")

    print("\nDone.")

if __name__ == "__main__":
    main()
