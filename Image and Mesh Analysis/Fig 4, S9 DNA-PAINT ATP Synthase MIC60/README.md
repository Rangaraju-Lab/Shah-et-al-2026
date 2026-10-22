# Fig 4, S9 — DNA-PAINT ATP Synthase MIC60

Analysis of ATP synthase and MIC60 DNA-PAINT localizations. The primary Fig 4 workflow runs **SMLM clustering → juxtaspinal measurements → nonjuxtaspinal measurements**. The S9 section provides NeNA precision, localizations per cluster, molecules through acquisition frames, and measurements per mitochondrion.

This package starts from drift-corrected, masked Picasso localization files and the corresponding annotations. Raw movie localization, drift correction, channel registration, and manual ROI/mask creation are upstream steps; the matching data distribution supplies the analysis inputs.

## Setup

Use Python 3.12; Python 3.11 or newer compatible with the dependencies is also supported. From a terminal in this folder:

```bash
python setup_env.py
```

On macOS/Linux, use `python3 setup_env.py` if your Python launcher is named `python3`. Setup creates `.venv`, installs `requirements.txt`, and creates the input/output folders while preserving existing data. Internet access is required for dependency installation. Picasso is installed as the Python package **`picassosr==0.11.3`**, imported in scripts as `picasso`. No separate Picasso GUI installation or GPU is needed for these analyses.

Merge the matching project folder from the separate data repository into this folder. With the filenames and calibration described below, no script edits are needed.

## Fig 4: primary measurements

Run the default workflow on Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe run_analysis.py
```

On macOS/Linux:

```bash
.venv/bin/python run_analysis.py
```

The default runs only these three stages, in this order:

| Order | Script | Purpose |
| --- | --- | --- |
| 00 | `scripts/00_smlm_cluster.py` | Compute NeNA independently for each field-of-view/probe file, run the 3D SMLM clusterer, and save clustered localizations plus cluster centers |
| 01 | `scripts/01_measure_juxtaspinal.py` | Position dendrite-oriented boxes from synapse coordinates; count cluster centers and measure mitochondrial area in each box |
| 02 | `scripts/02_measure_nonjuxtaspinal.py` | Position equivalent boxes from supplied nonjuxtaspinal ROIs; count cluster centers and measure mitochondrial area |

The scripts use paths relative to their own locations, so the runner can also be launched using its absolute path from another working directory. The runner streams progress and saves a transcript for each stage under `Run logs`. It stops on a process failure or reported per-file errors/missing required inputs.

To use cluster centers already merged from the data repository, run the two measurements directly through the runner:

```powershell
.\.venv\Scripts\python.exe run_analysis.py --only juxta
.\.venv\Scripts\python.exe run_analysis.py --only nonjuxta
```

To regenerate clustering alone:

```powershell
.\.venv\Scripts\python.exe run_analysis.py --only cluster
```

### Per-field-of-view, per-probe clustering radii

Each input HDF5 file represents **one field of view and one probe**. The script loads that file, calculates its own NeNA from the localizations, and sets:

$$r_x = r_y = r_{xy} = 2\,\mathrm{NeNA}_{xy}, \qquad r_z = 2\,r_{xy} = 4\,\mathrm{NeNA}_{xy}.$$

For example, a file with NeNA of 10 nm uses XY radius 20 nm and Z radius 40 nm. Another file or probe receives radii from its own NeNA. There is no pooling across fields, probes, or experimental conditions, and the clustering stage does not read the separate S9 NeNA CSV.

Picasso returns NeNA in camera pixels. XY/Z radii passed to the clusterer are in camera pixels; the metadata `Pixelsize` converts them to nanometers. Input localization `x` and `y` are in camera pixels, while `z` is in nanometers. Picasso converts Z to camera pixels internally before the ellipsoidal neighborhood search, using the supplied pixel size. The script retains `NENA_RADIUS_MULTIPLIER = 2.0`, `Z_RADIUS_FACTOR = 2.0`, `MIN_LOCS = 10`, `FRAME_ANALYSIS = True`, and `REQUIRE_3D = True`.

The output metadata records each file's NeNA, pixel size, XY radius, and Z radius in pixels/nanometers. Both clustered localizations and cluster centers retain the input filename, without adding a suffix; their separate folders distinguish the products. One center row represents one accepted cluster and is treated as one molecule by downstream scripts.

### Fig 4 box geometry and quantification

Dendrite line/polyline ROIs are interpolated at 0.05 µm spacing. Each synapse row or nonjuxtaspinal ROI centroid is mapped to the nearest interpolated dendrite point. A **2 µm box along the dendrite × 4 µm perpendicular to it** is centered there, with its orientation taken from the local dendrite tangent. A nearest point is assigned without a maximum-distance cutoff. Boxes are evaluated independently and may overlap.

For the juxtaspinal analysis, every finite XY row in the supplied `Synapse Localizations` file is one anchor and receives a sequential `spine` index. Supply the intended synapse representatives; the script does not consolidate these rows into synapses. Nonjuxtaspinal locations come from the supplied ROI ZIP; they are not automatically selected by excluding synapses. Polygon-like ROIs use their area centroid where supported; other supported types use the mean of their coordinates or bounding-box center. Each selected point is then snapped to its closest dendrite location.

Both analyses count all probe cluster centers inside each box. Counts are **not additionally restricted to mitochondrial-mask pixels**, and these measurements use XY coordinates without a Z-selection window. `mito_area_um2` is the foreground area of the original binary mitochondrial mask inside the same oriented box. The CSVs contain counts and area; they do not add a density column. A count/area ratio can be calculated downstream when the area is positive.

The original calibration is retained: X = **98.44999694824219 nm/pixel**, Y = **97.9000015258789 nm/pixel**. Fig 4 geometry and area use their arithmetic mean, **98.17499923706055 nm/pixel**, with pixel area equal to the square of that mean. This fixed image calibration is independent of the HDF5 metadata pixel size used for NeNA/clustering. The distributed images, localization coordinates, and ROIs must refer to the same registered image grid.

### Fig 4 output columns

A separate CSV is saved for each cell, named with its full `Date Condition DishN CellN` prefix. Condition remains in the filename; it is not a separate column in the supplied Fig 4 tables.

| Table | Columns |
| --- | --- |
| `Measurements/juxtaspinal/<cell prefix>.csv` | `Date`, `Dish`, `Cell`, `dendrite`, `spine`, `probe01`, `probe02`, `mito_area_um2` |
| `Measurements/nonjuxtaspinal/<cell prefix>.csv` | `Date`, `Dish`, `Cell`, `dendrite`, `non_juxta`, `probe01`, `probe02`, `mito_area_um2` |

**The original Fig 4 output-column names differ from the input probe labels:**

| Biological target | Input file probe | Fig 4 CSV count column |
| --- | --- | --- |
| ATP synthase | `probe00` | `probe01` |
| MIC60 | `probe01` | `probe02` |
| Synapse anchors | `probe02` in `Synapse Localizations` | Used to position juxtaspinal boxes; not the Fig 4 `probe02` count column |

These names are preserved. Dendrite IDs start at 1 in ROI-file order. The juxtaspinal script uses lexicographic ROI sorting; the nonjuxtaspinal/per-mito scripts use natural sorting. Zero-padded names such as `Dendrite01.roi`, `Dendrite02.roi` keep IDs consistent across these sorting conventions. Nonjuxtaspinal ROI numbers are taken from names such as `NonJuxta1.roi`, a trailing number, or the fallback sorted order.

### Fig 4 visualizations

Per-cell ImageJ/Fiji-compatible **five-channel uint16 TIFFs** are saved in `Visualizations/juxtaspinal` and `Visualizations/nonjuxtaspinal` with their original labels and LUTs:

| Channel | Contents | LUT |
| --- | --- | --- |
| 1 | Synapse anchors, or nonjuxtaspinal ROI centroids | Yellow hot |
| 2 | Input `probe00` cluster centers: ATP synthase | Cyan hot |
| 3 | Input `probe01` cluster centers: MIC60 | Magenta hot |
| 4 | Mitochondrial-mask outline | Green |
| 5 | Dendrite-oriented box outlines | White |

Images are rendered at 10× the original pixel sampling, approximately 9.8175 nm/output pixel. Molecule rendering retains Gaussian sigma 1.2 output pixels, percentile normalization 98.5, brightness multiplier 1.0, and gamma 0.65. Mitochondrial outlines use the supplied resized/blurred/Otsu-derived display mask. These display operations do not change the original coordinates or binary mask used for measurements. Large fields of view produce large rendered arrays and TIFFs.

## S9: supporting analyses

Run S9 after clustering has generated `Undrift Masked SMLM` and `Undrift Masked SMLM Centers`:

```powershell
.\.venv\Scripts\python.exe run_analysis.py --section s9
```

To run Fig 4 and then all S9 stages in one command:

```powershell
.\.venv\Scripts\python.exe run_analysis.py --section all
```

On macOS/Linux, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`. S9 does not rerun clustering automatically. Individual stages are available through `--only`:

| Order | Runner selection | Script | Output |
| --- | --- | --- | --- |
| 03 | `--only nena` | `03_measure_NeNA.py` | `Measurements/Undrift Masked NeNA.csv` |
| 04 | `--only locs-per-cluster` | `04_QC_locs_per_cluster.py` | `Measurements/locs_per_cluster/locs_per_cluster.csv` |
| 05 | `--only frames` | `05_QC_cluster_density_frames.py` | `Measurements/molecules_through_frames/molecules_through_frames.csv` |
| 06 | `--only per-mito` | `06_measure_probe_density_per_mito.py` | `Measurements/molecules_per_mito/<cell prefix>.csv` |

### NeNA precision

NeNA is recomputed independently from the original `Undrift Masked` input file for each probe and field of view, converted from camera pixels to nm, and reported to 3 decimals. Columns are `Date`, `Condition`, `Dish`, `Cell`, `Probe`, and `NeNA_nm`. Condition names containing spaces are accepted consistently with the other metadata parsers.

### Localizations per cluster

Clustered localization rows are counted by nonnegative integer `group` ID. Invalid/nonfinite groups and negative noise labels are excluded. Columns are `Date`, `Condition`, `Dish`, `Cell`, `cluster_id`, `ATP_synthase_locs`, and `Mic60_locs`. The output takes the union of group IDs from the two probes and uses zero where an ID is absent from one probe. **IDs are generated independently for each probe:** equal IDs in the same CSV row do not establish a paired or colocalized molecule.

### Molecules through acquisition frames

Each valid cluster-center row is assigned to exactly one **1,200-frame bin** using its center `frame` value, which represents the cluster's mean frame. Bins are start-inclusive and end-exclusive, for example `0-1200` and `1200-2400`. The acquisition length is taken from `Frames` metadata or inferred from observed frames; the larger length of the two probes is used. Empty bins are retained. Columns are `Date`, `Condition`, `Dish`, `Cell`, `frame_bin`, `ATP_synthase`, and `Mic60`. These are molecule counts, without area normalization or a new spatial clustering step.

### Per-mitochondrion measurements

The 2D mitochondrial mask is labeled with 8-connected components, retaining components of at least one pixel. Center coordinates are rounded to the nearest image pixel; a center contributes only to the mitochondrial label at that pixel. Nonfinite coordinates and coordinates mapping outside the image do not contribute. Each component is assigned to a dendrite by a vote over its skeleton pixels, or all its mask pixels if no skeleton pixels are present. Ties use the smaller mean distance and then the smaller dendrite ID.

Skeleton length sums each horizontal, vertical, and diagonal connection between neighboring skeleton pixels once, using the anisotropic X/Y calibration and diagonal length. Skeleton pixels outside mitochondrial labels are excluded. This stage uses the supplied skeleton TIFF; it does not create or re-skeletonize it.

Columns are `Date`, `Condition`, `Dish`, `Cell`, `dendrite`, `mito`, `mito_area_um2`, `skeleton_length_um`, `probe00`, and `probe01`. Here the count columns match the **input** labels: `probe00` = ATP synthase and `probe01` = MIC60. Area uses the product of X and Y pixel sizes, while Fig 4 uses the square of the mean pixel size. The per-mito CSV contains counts, area, and length; it does not add a density column. Files are written with 6 decimal places for floating values.

## Data folders and filenames

The following folders are included empty for merging with the matching data distribution. Put files directly inside them; discovery is not recursive.

```text
Fig 4, S9 DNA-PAINT ATP Synthase MIC60/
├── README.md
├── requirements.txt
├── setup_env.py
├── run_analysis.py
├── source_files.json
├── scripts/
│   ├── 00_smlm_cluster.py
│   ├── 01_measure_juxtaspinal.py
│   ├── 02_measure_nonjuxtaspinal.py
│   ├── 03_measure_NeNA.py
│   ├── 04_QC_locs_per_cluster.py
│   ├── 05_QC_cluster_density_frames.py
│   └── 06_measure_probe_density_per_mito.py
├── Undrift Masked/
├── Undrift Masked SMLM/
├── Undrift Masked SMLM Centers/
├── Synapse Localizations/
├── Dendrites/
├── Mito Masks/
├── NonJuxta ROI/
├── Mito Skels/
├── Measurements/
├── Visualizations/
└── Run logs/
```

A common cell prefix is `2026-01-08 Control Dish1 Cell1`; a multiword condition such as `Homeostatic Scaling` is also supported. Match the complete prefix, including condition spelling/case, across every input. Use lowercase `.hdf5`, `.yaml`, `.zip`, `.tif` or `.tiff` extensions for the complete workflow and lowercase `probe00`, `probe01`, `probe02` tokens. HDF5 filenames must end in the clean probe token; files with extra suffixes after that token are not selected for clustering.

| Folder | Example filename | Required for |
| --- | --- | --- |
| `Undrift Masked` | `2026-01-08 Control Dish1 Cell1 probe00.hdf5` and matching `probe01.hdf5` | Clustering; S9 NeNA |
| `Undrift Masked SMLM` | Same probe filenames, produced by clustering | S9 localizations per cluster |
| `Undrift Masked SMLM Centers` | Same probe filenames, produced by clustering | Fig 4 measurements; S9 frames and per-mito |
| `Synapse Localizations` | `2026-01-08 Control Dish1 Cell1 probe02.hdf5` | Juxtaspinal anchors |
| `Dendrites` | `2026-01-08 Control Dish1 Cell1.zip` | Both Fig 4 measurements; S9 per-mito |
| `Mito Masks` | `2026-01-08 Control Dish1 Cell1.tif` | Both Fig 4 measurements; S9 per-mito |
| `NonJuxta ROI` | `2026-01-08 Control Dish1 Cell1.zip` | Nonjuxtaspinal anchor ROIs |
| `Mito Skels` | `2026-01-08 Control Dish1 Cell1.tif` | S9 per-mito only |

Keep each Picasso HDF5 file with its same-stem YAML sidecar when present; the pinned Picasso version also supports embedded HDF5 metadata. At minimum, supply `Pixelsize` in nm/pixel and the localization columns required by the calculation. For NeNA/clustering these include `x`, `y`, `z`, `frame`, `lpx`, and `lpy`; use a normal Picasso localization table with its original fields retained. `Frames` records total acquisition frames for frame-binning outputs. Clustering generates metadata and centers automatically; those output folders can instead be merged from an existing matching analysis.

Mito masks and skeletons must be 2D images, not RGB images or Z stacks. Positive pixels are foreground; zero is background. Mito masks and skeletons must have the same shape. XY coordinates in all HDF5 inputs and all ImageJ ROIs must already be registered to that original image grid, in image pixels. The scripts do not align channels, rescale localization coordinates to a different image grid, or generate missing ROI/mask annotations.

Save dendrite lines or segmented polylines from Fiji ROI Manager as a ZIP of `.roi` files. Basic straight-line/polyline ROIs with coordinates on the original image grid are supported; preserve those coordinates when exporting. Save nonjuxtaspinal selections as a separate ROI Manager ZIP for each cell. The distributed data should include these annotations; no manual annotations are required when using that complete data distribution.

## Reruns and configuration

The supplied defaults process all matched files, with `OVERWRITE = True`, dry-run disabled, and first-file-only modes disabled. Clustering and NeNA use up to 8 workers; juxtaspinal, nonjuxtaspinal, localization-count, and per-mito stages use up to 4. Frame binning runs sequentially. Large localization tables and 10× image renderings can use substantial RAM; worker settings at the top of each script can be reduced without changing analysis parameters.

All numerical settings, masks, ROI geometry, measurements, output labels, and LUTs retain their supplied definitions. Clustering parameters and per-file radii are recorded in HDF5/YAML metadata. Optional configuration for a different experiment is at the top of the relevant script; the documented data distribution and default calibration require no edits. Rerun downstream measurements after regenerating cluster centers or changing annotations. Different analyses should use separate project copies to keep inputs and generated products together.

The `.gitignore` retains code and empty folders while excluding merged data, generated tables/TIFFs, environment files, and logs from Git commits. `source_files.json` maps each supplied script to its ordered packaged filename.
