# Fig S4 — Hierarchy-constrained attention ResUNet3D

A 3D residual U-Net with bottleneck self-attention for segmenting electron microscopy tomograms into five biological channels. This package provides the complete trainer, dependency setup, and a small synthetic architecture test. Training uses hierarchy-gated segmentation, auxiliary boundary and affinity losses, and complete-volume validation.

## Architecture schematic

![Architecture schematic](assets/model_architecture.png)

Add your schematic as `assets/model_architecture.png`. This is the only unfinished item in the package.

## Quick start

Merge the matching `Fig S4` folder from the data distribution into this folder. The TIFF data will then sit under `manual segmentations`, beside `train.py`. Existing tomogram folder names can be kept as they are.

From a terminal in `Fig S4`, create the environment:

```bash
python setup_env.py
```

Then run the trainer on Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe train.py
```

On macOS/Linux:

```bash
.venv/bin/python train.py
```

Use Python 3.11 for setup; Python 3.10 or newer is accepted if supported by the installed dependencies. On systems using the `python3` launcher, run `python3 setup_env.py`. Setup creates a local `.venv`, installs `requirements.txt`, and ensures the input folders exist. It does not change existing TIFFs. No environment activation, data download configuration, or folder renaming is needed.

To test the network without data, run the following before training:

```powershell
.\.venv\Scripts\python.exe smoke_test.py
```

On macOS/Linux, use `.venv/bin/python smoke_test.py`. This runs the network at its default base width on a smaller synthetic CPU input, checks output shapes and hierarchy constraints, and performs a backward pass and optimizer step. It does not write checkpoints or assess segmentation accuracy.

## Input folders

The supplied input folders are empty and ready to merge with the data distribution:

```text
Fig S4/
├── README.md
├── requirements.txt
├── setup_env.py
├── train.py
├── infer.py
├── smoke_test.py
├── assets/
│   └── model_architecture.png   # your schematic
└── manual segmentations/
    ├── train/
    │   └── any tomogram name/
    │       ├── EM_stack.tif
    │       ├── OMM.tif
    │       ├── IBM.tif
    │       ├── IMS.tif
    │       ├── ER.tif
    │       └── Ribo.tif
    ├── validate/
    │   └── any tomogram name/
    │       └── matching EM and segmentation TIFFs
    └── test/
        └── any tomogram name/
            └── matching EM and segmentation TIFFs
```

Use one subfolder per tomogram in each split. Nested subfolders also work; discovery searches for the configured EM filename. Tomogram names are unrestricted: no year, Control, TTX, or numbered sample name is required. All discovered tomograms are loaded. Each split needs at least one tomogram. `.gitkeep` files retain the empty folders in GitHub and are ignored by the data loader.

Each tomogram must have one grayscale EM stack and matching segmentation stacks. All stacks must have matching `(Z, Y, X)` dimensions and alignment. Masks are interpreted as foreground wherever their values exceed zero, so both 0/1 and 0/255 masks work. `.tif`, `.tiff`, `.TIF`, and `.TIFF` extensions are supported. Provide one file per configured name. RGB or time/channel hyperstacks are not the expected input.

The first four segmentation files are required. The fifth, ribosome file is optional; if it is missing, that channel is masked out of supervised losses and metrics for that tomogram. A present all-zero mask means annotated absence. The loader removes ER target voxels overlapping annotated ribosomes. Keep the same biological tomogram in only one split; copied or renamed duplicates are not detected by content. The trainer checks overlapping resolved folder paths and matching image/mask dimensions, without checking tomogram names.

## Changing image and segmentation filenames

At the top of `train.py`, change only these settings to match your own TIFFs:

```python
EM_NAME = "EM_stack"
CHANNELS = ["OMM", "IBM", "IMS", "ER", "Ribo"]
```

Names are TIFF filename stems, without the extension. For example, if the input EM image is `image.tif` and the corresponding masks are named `mitochondria.tif`, `inner_volume.tif`, `intermembrane_space.tif`, `endoplasmic_reticulum.tif`, and `ribosomes.tif`, use:

```python
EM_NAME = "image"
CHANNELS = ["mitochondria", "inner_volume", "intermembrane_space", "endoplasmic_reticulum", "ribosomes"]
```

The configured names also appear in metrics and plot labels. Sampling keys, threshold keys, auxiliary channels, and optional ribosome handling follow these names automatically; no additional naming edits are needed.

Preserve the five-channel count and biological role order below. Renaming files is supported; changing the number or meaning of the channels would require changing the architecture or hierarchy logic.

| Position | Default name | Biological role |
| --- | --- | --- |
| 1 | OMM | Whole mitochondrial volume parent mask, not only a membrane surface |
| 2 | IBM | Inner-boundary-enclosed volume, including matrix and cristae lumen |
| 3 | IMS | Intermembrane space and cristae lumen |
| 4 | ER | Endoplasmic reticulum outside mitochondria |
| 5 | Ribo | Punctate ribosomes outside mitochondria |

## Network

The default network has 13,856,782 trainable parameters. It takes one EM channel and produces five segmentation probability channels, five boundary logits, and twelve affinity logits.

| Stage | Channels | Operation |
| --- | --- | --- |
| Encoder 0 | 24 | Residual 3D block, stride `(1,1,1)` |
| Encoder 1 | 48 | Residual 3D block, stride `(1,2,2)` |
| Encoder 2 | 96 | Residual 3D block, stride `(2,2,2)` |
| Encoder 3 | 192 | Residual 3D block, stride `(2,2,2)` |
| Bottleneck | 384 | Residual 3D block, stride `(2,2,2)`, then spatial self-attention |
| Decoder | 192 → 96 → 48 → 24 | Transposed convolutions, encoder skip concatenation, residual blocks |
| Segmentation head | 5 | `1×1×1` convolution, sigmoid, hierarchy gating |
| Boundary head | 5 | `1×1×1` convolution |
| Affinity head | 12 | Z/Y/X neighbor affinities for the first four channels |

Residual blocks use two padded `3×3×3` convolutions, GroupNorm, SiLU, and a projected shortcut when stride or channel count changes. Attention uses four heads, 96 query/key channels, 384 value channels, softmax over spatial positions, a residual output projection, and a residual pointwise feed-forward block of width 768. At the default input size, the bottleneck contains 1,024 spatial tokens. Decoder upsampling reverses the encoder strides and matches tensor shapes before concatenation.

For raw sigmoid probabilities `r`, segmentation gating is:

```text
p_OMM  = r_OMM
p_IBM  = p_OMM × r_IBM
p_IMS  = p_OMM × r_IMS
p_Ribo = (1 − p_OMM) × r_Ribo
p_ER   = (1 − p_OMM) × r_ER × (1 − r_Ribo)
```

These formulas refer to biological roles, regardless of the configured TIFF names. IBM and IMS are each constrained by OMM. Final thresholded masks explicitly exclude mitochondria from ER/ribosomes and ribosomes from ER. Boundary and affinity heads supervise the shared network; segmentation evaluation uses the segmentation probabilities.

## Training and evaluation

Defaults: seed 42, patch size `(32,256,256)` ZYX, 256 patches per epoch, batch size 1, 200 total epochs, AdamW learning rate `2e-4`, weight decay `1e-4`, and gradient clipping at 1.0.

Patches are allocated evenly across training tomograms, up to integer rounding, with the existing class-centered sampling proportions: OMM 0.16, IBM 0.12, IMS 0.12, ER 0.18, Ribo 0.28, background 0.10, and random 0.04. Training-derived channel and positive-class weights use training data only.

EM is normalized by whole-tomogram z-score and clipped before cropping. Augmentation includes flips, in-plane quarter rotations, intensity scale/shift, noise, and occasional zeroed Z slabs. Boundary and affinity targets are calculated from augmented masks.

The objective is `BCE + Dice + 0.15 × boundary BCE + 0.15 × affinity BCE + 0.20 × hierarchy penalty`. The hierarchy penalty includes an ER/ribosome overlap term with weight 0.20. Missing ribosome annotations are masked out of supervised losses.

Every epoch evaluates all complete validation tomograms with overlapping patches, stride `(16,128,128)`, and Hann-derived blending. The implementation additionally applies probability containment/exclusion gates after blending, then thresholds each channel at 0.50. The best checkpoint is selected using mean complete-validation Dice across classes, with each class averaged over its annotated tomograms.

Whenever a new best checkpoint is saved, the trainer renders the same fixed patch from the first test tomogram in sorted path order. This display does not affect optimization or checkpoint selection. Final testing evaluates all test tomograms using the best validation-selected checkpoint. Avoid tuning the model using test visualizations if you want an independent test result.

## Outputs and resuming

Outputs are written to `runs` beside the trainer:

| Output | Contents |
| --- | --- |
| `split.json` | Split paths, evaluation settings, visualization source |
| `derived_weights.json` | Training-derived label frequencies and loss weights |
| `training_log.csv` | Epoch time, losses, training patch Dice, validation Dice/precision/recall |
| `last.pt` | Latest model, optimizer, epoch, and settings |
| `best.pt` | Checkpoint with highest validation mean class Dice |
| `validation_latest_*` | Latest complete-validation metrics |
| `validation_best_*` | Complete-validation metrics for the best checkpoint |
| `test_best_*` | Final independent-test metrics |
| `visualizations/` | Fixed test patch whenever a new best checkpoint is saved |

Metric sets include CSV tables and JSON. Class summaries report mean and sample standard deviation over annotated tomograms. The trainer does not export prediction TIFF stacks. Checkpoint `val_loss` is stored as NaN and is unused; checkpoint selection uses Dice. Epoch time covers training and validation, excluding later checkpoint/visualization writes and final testing.

Rerunning `train.py` automatically resumes from `runs/last.pt`, restoring model and optimizer states. Split paths must match the saved split record. The workflow does not restore all random generator states, so resumed runs need not exactly match uninterrupted trajectories.

For a fresh run or different total epoch count:

```powershell
.\.venv\Scripts\python.exe train.py --out-dir runs_review --epochs 10
```

`--epochs` means the final epoch number, including already completed epochs. Use a new output directory after changing data, label names, or settings so old checkpoints are not mixed with a new experiment.

For data located elsewhere:

```powershell
.\.venv\Scripts\python.exe train.py --data-root "D:\datasets\manual segmentations"
```

Relative data and output paths are resolved beside `train.py`, regardless of the terminal's working directory. Remaining training settings are in the `SETTINGS` section of that script.

## Inference on new EM images

Place grayscale EM TIFF stacks directly in `inference images`, beside `train.py` and `infer.py`. Input filenames are unrestricted; no masks are required. Train the model first, then run on Windows:

```powershell
.\.venv\Scripts\python.exe infer.py
```

On macOS/Linux:

```bash
.venv/bin/python infer.py
```

Inference loads `runs/best.pt`, uses the trained checkpoint's patch size, overlap stride, thresholds, base width, and attention settings, and takes channel names from the settings in `train.py`. Keep those names consistent with the checkpoint used for inference. For a different run directory:

```powershell
.\.venv\Scripts\python.exe infer.py --checkpoint runs_review/best.pt
```

Each input gets its own output folder, named after the input filename without its TIFF extension:

```text
Fig S4/
├── train.py
├── infer.py
└── inference images/
    ├── sample A.tif
    ├── sample B.tiff
    └── segmentations/
        ├── sample A/
        │   ├── sample A_pred.tif
        │   └── sample A_seg.tif
        └── sample B/
            ├── sample B_pred.tif
            └── sample B_seg.tif
```

`*_pred.tif` contains the five probability channels, stored by default as uint16 values from 0 to 65535. `*_seg.tif` contains the five binary segmentation channels as uint8 values 0/255. Both are ImageJ/Fiji-compatible hyperstacks with five channels, the input's Z slices, and one time frame. Channel order follows `CHANNELS` in `train.py`. Image dimensions are preserved.

The attached inference implementation's whole-volume normalization, overlapping-patch blending, hierarchy cleanup, and TIFF exports are retained. `PREDICTION_DTYPE`, `SEGMENTATION_DTYPE`, and `OVERWRITE` are configurable in the settings at the top of `infer.py`; existing outputs are overwritten by default. Use distinct input filename stems, so `sample.tif` and `sample.tiff` do not write to the same output folder. Only TIFFs directly in `inference images` are processed, so saved segmentation files are never treated as inputs.

The package includes empty `inference images` and `segmentations` folders. Setup also creates them if needed. Neither original EM stacks nor generated segmentation TIFFs are included in the code distribution.

## Hardware and dependencies

Device selection is automatic: NVIDIA CUDA, then Apple MPS, then CPU. A GPU is recommended for full-size training. All tomograms, masks, and foreground index arrays are loaded into RAM at startup; full-volume evaluation also allocates prediction accumulators. The small CPU smoke test does not establish whether full training fits on your machine.

`requirements.txt` contains dependency ranges rather than a frozen publication environment. If an NVIDIA machine requires a specific CUDA build of PyTorch, install that build in the virtual environment before running setup. An existing Python environment can also be used:

```bash
python -m pip install -r requirements.txt
python smoke_test.py
python train.py
```

To record the installed environment on Windows:

```powershell
.\.venv\Scripts\python.exe -m pip freeze > requirements-resolved.txt
```

For Python reuse without starting training:

```python
from train import HierarchyConstrainedAttentionResUNet3D
model = HierarchyConstrainedAttentionResUNet3D()
```


