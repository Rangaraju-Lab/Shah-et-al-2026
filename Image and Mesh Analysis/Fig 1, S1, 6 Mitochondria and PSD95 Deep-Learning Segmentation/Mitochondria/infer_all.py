from pathlib import Path
import random

import numpy as np
import tifffile as tiff

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.ndimage import gaussian_filter


# ============================================================
# SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent

# Folder containing individual time-series stacks.
# Each .tif is treated as one separate T,Y,X time series.
INPUT_DIR = ROOT / "InferenceTifs"

# Output folder. Masks are saved with the same filename as input.
OUTPUT_MASK_DIR = INPUT_DIR / "InferenceTifMasks"

CHECKPOINT_PATH = ROOT / "Outputs" / "checkpoints" / "best.pt"

# Must match training.
SEG_BASE_CHANNELS = 24
AVG_BASE_CHANNELS = 16

LOCAL_Z_SIGMA = 12.0
EDGE_SIGMA = 1.0
Z_CLIP = 6.0

THRESHOLD = 0.5

GATE_MIN_SCALE = 0.10
GATE_MAX_SCALE = 2.00

SEED = 1


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

    if arr.ndim == 4:
        # Handles simple T,C,Y,X or C,T,Y,X cases by taking channel 0.
        if arr.shape[1] <= 4:
            arr = arr[:, 0, :, :]
        elif arr.shape[0] <= 4:
            arr = arr[0, :, :, :]
        else:
            raise ValueError(f"Ambiguous 4D stack shape {arr.shape} from {path}")

    if arr.ndim != 3:
        raise ValueError(f"Expected T,Y,X stack. Got {arr.shape} from {path}")

    return arr.astype(np.float32)


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


def mask_to_uint8(mask):
    return (mask > 0).astype(np.uint8) * 255


# ============================================================
# FEATURE PRECOMPUTE
# ============================================================

def make_current_features(stack):
    n_frames, h, w = stack.shape

    current_features = np.zeros((n_frames, 3, h, w), dtype=np.float32)

    print("  Computing current-frame features...")
    print("  Channels: global_z, local_z, gradient")

    for t_idx in range(n_frames):
        print(f"    frame {t_idx + 1}/{n_frames}")

        img = stack[t_idx]

        gz = global_zscore(img)
        lz = local_zscore(img, sigma=LOCAL_Z_SIGMA)
        grad = gradient_map(lz, sigma=EDGE_SIGMA)

        current_features[t_idx, 0] = gz
        current_features[t_idx, 1] = lz
        current_features[t_idx, 2] = grad

    return current_features


def make_average_features_for_single_timeseries(current_features):
    """
    Each input TIFF is one complete time series.
    Therefore this returns one average-feature map for the whole stack.
    """
    n_frames, _, h, w = current_features.shape

    average_features = np.zeros((4, h, w), dtype=np.float32)

    gz_accum = np.zeros((h, w), dtype=np.float32)
    lz_accum = np.zeros((h, w), dtype=np.float32)
    grad_accum = np.zeros((h, w), dtype=np.float32)
    cand_accum = np.zeros((h, w), dtype=np.float32)

    for tt in range(n_frames):
        gz = current_features[tt, 0]
        lz = current_features[tt, 1]
        grad = current_features[tt, 2]
        cand = (lz > 0.5).astype(np.float32)

        gz_accum += gz
        lz_accum += lz
        grad_accum += grad
        cand_accum += cand

    denom = max(n_frames, 1)

    average_features[0] = gz_accum / denom
    average_features[1] = lz_accum / denom
    average_features[2] = grad_accum / denom
    average_features[3] = cand_accum / denom

    average_features[0:3] = np.clip(average_features[0:3], -Z_CLIP, Z_CLIP)
    average_features[3] = np.clip(average_features[3], 0, 1)

    return average_features.astype(np.float32)


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
# CHECKPOINT
# ============================================================

def load_model(checkpoint_path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = AvgGuidedSegmentationModel().to(device)

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")

    if isinstance(ckpt, dict) and "settings" in ckpt:
        print("Checkpoint settings:")
        for k, v in ckpt["settings"].items():
            if k in [
                "seg_base_channels",
                "avg_base_channels",
                "local_z_sigma",
                "edge_sigma",
                "z_clip",
                "gate_min_scale",
                "gate_max_scale",
            ]:
                print(f"  {k}: {v}")

    return model


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def infer_one_timeseries(model, stack, device):
    n_frames, h, w = stack.shape

    current_features = make_current_features(stack)
    avg_features = make_average_features_for_single_timeseries(current_features)

    avg_input = torch.from_numpy(avg_features[None, ...].astype(np.float32)).to(device)

    mask_stack = np.zeros((n_frames, h, w), dtype=np.uint8)

    print("  Running inference...")

    for tt in range(n_frames):
        print(f"    frame {tt + 1}/{n_frames}")

        current_np = current_features[tt]
        current = torch.from_numpy(current_np[None, ...].astype(np.float32)).to(device)

        seg_logits, avg_gate_logits, avg_gate = model(current, avg_input)

        pred = torch.sigmoid(seg_logits)[0, 0].detach().cpu().numpy().astype(np.float32)

        mask_stack[tt] = mask_to_uint8(pred >= THRESHOLD)

        if device.type == "mps":
            torch.mps.empty_cache()

    return mask_stack


def list_input_tifs(input_dir, output_dir):
    tif_files = []

    for pattern in ["*.tif", "*.tiff"]:
        tif_files.extend(sorted(input_dir.glob(pattern)))

    tif_files = [
        p for p in tif_files
        if p.is_file()
        and p.parent.resolve() != output_dir.resolve()
        and not p.name.startswith(".")
    ]

    return sorted(tif_files)


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_MASK_DIR.mkdir(exist_ok=True)

    tif_files = list_input_tifs(INPUT_DIR, OUTPUT_MASK_DIR)

    if len(tif_files) == 0:
        raise FileNotFoundError(f"No .tif/.tiff files found in {INPUT_DIR}")

    print(f"Found {len(tif_files)} input TIFF files.")
    print(f"Output mask folder: {OUTPUT_MASK_DIR}")

    model = load_model(CHECKPOINT_PATH, DEVICE)

    for file_idx, tif_path in enumerate(tif_files, start=1):
        print("")
        print(f"[{file_idx}/{len(tif_files)}] Processing: {tif_path.name}")

        stack = read_tif_stack(tif_path)
        print(f"  Stack shape: {stack.shape}")

        mask_stack = infer_one_timeseries(
            model=model,
            stack=stack,
            device=DEVICE,
        )

        out_path = OUTPUT_MASK_DIR / tif_path.name

        tiff.imwrite(
            str(out_path),
            mask_stack.astype(np.uint8),
            imagej=True,
        )

        print(f"  Saved mask: {out_path}")

    print("")
    print("Done.")


if __name__ == "__main__":
    main()
