# Fig 3, S5, S6, S7, S8, S10, S11 — Tomogram Mesh Analysis

An ordered workflow for reconstructing tomogram segmentation meshes, remeshing and trimming them, calculating mitochondrial matrix and membrane surfaces, mapping curvature, finding crista junctions, measuring cristae/ER/ribosomes, and saving figure screenshots. The analysis uses the supplied numerical settings and rendering styles.

## Install the software

Two separate environments are required:

| Environment | Purpose | Installation |
| --- | --- | --- |
| Python 3.11 or 3.12 | Image processing, mesh operations, measurements, screenshots, and launching the workflow | Install Python, then run the setup command below |
| Blender with BlendGAMer | Voxel remeshing and GAMer curvature calculation | Install the [MCell/CellBlender bundle](https://mcell.org/download.html), or Blender with a matching [BlendGAMer installation](https://gamer.readthedocs.io/en/latest/install.html) |

Use a Blender/BlendGAMer combination built for the same platform and Blender Python version. The supplied launchers include support for older and newer STL import/export APIs, and the curvature worker recognizes both legacy add-ons and Blender extension module paths. The original scripts include CellBlender installations using Blender 5.1 and 2.93; that does not make an arbitrary BlendGAMer binary interchangeable between those versions.

Open the installed Blender once to finish any operating-system prompts. In Blender's **Edit → Preferences → Add-ons**, enable **BlendGAMer** if it is not already available. Its `gamer.compute_curvatures` operator and `colormap.curveToData` interface are required. Installing a Python package in the project's `.venv` does not install an add-on into Blender. MCell simulations and CellBlender model files are not used by this analysis.

From a terminal in this package's top folder, run:

```bash
python setup_env.py
```

On macOS/Linux, use `python3 setup_env.py` if that is your Python launcher. Setup creates `.venv`, installs `requirements.txt`, locates Blender, loads the curvature add-on, and records the executable locally. Automatic discovery includes standard Blender/CellBlender installations, matching bundles under Downloads, a package-local `tools` folder, and the system PATH. For an installation elsewhere, pass its executable or macOS `.app` bundle using `setup_env.py --blender`; no script edit is needed. An explicit `BLENDER_EXE` environment variable is also supported.

Blender is installed separately because its executable and compiled add-ons depend on the operating system. The [official CellBlender installation guide](https://mcell.org/mcell4_documentation/installation.html) covers platform-specific installation steps.

## Merge the tomogram data

Merge the matching project folder from the separate data repository into this package. The `Control` and `TTX` folders are included empty and retained by `.gitkeep`. Input data are not included in the code distribution.

```text
Fig 3, S5, S6, S7, S8, S10, S11 Tomogram Mesh Analysis/
├── README.md
├── requirements.txt
├── setup_env.py
├── run_pipeline.py
├── settings.py
├── source_files.json
├── scripts/
│   ├── 00_scale_segmentations.py
│   ├── 01_mesh_segmentations.py
│   ├── 02_run_parallel_blender_remesh.py
│   ├── 02b_blender_remesh_worker.py
│   ├── 03_trim_extrusions.py
│   ├── 04_make_matrix_and_membranes.py
│   ├── 05_run_parallel_blendgamer_curvature.py
│   ├── 05b_blendgamer_curvature_worker.py
│   ├── 06_make_surfaces.py
│   ├── 07_make_CJ.py
│   ├── 08_measure_cristae_props.py
│   ├── 08b_measure_cristae_basic.py
│   ├── 09_measure_ER_props.py
│   ├── 10_measure_ribo_props.py
│   ├── 11_batch_visualize_cristae_screenshot_2um.py
│   ├── 12_batch_visualize_curvature_screenshot_2um.py
│   ├── 13_batch_visualize_ERMCS_screenshot_2um.py
│   ├── 13b_visualize_ERMCS_screenshot_2um_control1_2.py
│   ├── project_paths.py
│   └── configure_blender.py
├── Control/
│   └── 2022-10-08 Control 1/
│       ├── OMM.tif
│       ├── IBM.tif
│       ├── IMS.tif
│       ├── ER.tif
│       └── Ribo.tif
└── TTX/
    └── 2025-12-09 TTX 1/
        └── corresponding segmentation TIFFs
```

The dated sample folders in the tree illustrate the required naming convention; they are supplied by the data distribution, not generated as empty samples by setup. Put each tomogram directly under its condition folder. Keep the sample name in the form **`YYYY-MM-DD Control N`** or **`YYYY-MM-DD TTX N`**, where `N` is an integer. Measurement tables and junction coordinates extract `Date`, `Condition`, and `Mito` from this name. Do not place an additional directory level between the condition and the tomogram.

Use aligned, single-channel, three-dimensional **`(Z, Y, X)`** TIFF stacks with matching dimensions. For the complete workflow, provide five **uint8 masks with foreground 255 and background 0**, named `OMM`, `IBM`, `IMS`, `ER`, and `Ribo`. `.tif` and `.tiff` are accepted; use the exact case of the stems shown. Supply one extension per stem. OMM is required for centering. Boolean, float, and 16-bit TIFFs are not meshed by stage 01, which selects uint8 stacks and uses isolevel 127. Do not put grayscale uint8 EM images in the segmentation folder: stage 01 attempts to mesh every uint8 TIFF in `scaled`.

OMM represents the filled mitochondrial volume. IBM represents the volume enclosed by the inner boundary, including matrix and cristae lumen. IMS includes intermembrane space and cristae lumen. Their volume definitions matter: the matrix mesh is computed as `IBM_vol − IMS_vol`.

The complete figure workflow expects ER and ribosome inputs. Individual mitochondrial stages can run without those inputs, but unavailable surfaces will leave missing measurements or prevent the corresponding screenshot. An empty segmentation cannot produce a mesh; missing and empty objects are handled by the supplied stage routines, rather than replaced with invented geometry.

## Run the workflow

### Windows PowerShell

```powershell
.\.venv\Scripts\python.exe run_pipeline.py
```

### macOS/Linux

```bash
.venv/bin/python run_pipeline.py
```

This runs primary stages **00 through 13 in order**, using separate Python processes. Blender workers are launched automatically by stages 02 and 05; do not run `02b` or `05b` with regular Python. Outputs are saved beside the input data or at the project root as listed below. The runner streams progress and writes one transcript per stage under `pipeline logs`.

Run only reconstruction and quantitative analysis, without screenshots:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --to-stage 10
```

After quantitative analysis, render the three standard screenshot sets:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --from-stage 11 --to-stage 13
```

Rerun a particular stage, or restart from an intermediate result:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --only 06
.\.venv\Scripts\python.exe run_pipeline.py --from-stage 07 --to-stage 10
```

For macOS/Linux, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`. Stage numbers retain leading zeros. Starting at a later stage assumes that its input files already exist; the runner does not regenerate earlier intermediates automatically. A stage stops the runner if its process fails or it writes a new nonempty failure table. The transcript and the stage's CSV identify the affected sample and error. After correcting that input or installation problem, restart at the failed stage.

## Stage order and methods

Mesh coordinates are in **micrometers**. Marching cubes retains TIFF axis order: mesh coordinate 0 corresponds to TIFF Z, coordinate 1 to TIFF Y, and coordinate 2 to TIFF X. Thus the slab-thickness direction is mesh axis 0. Coordinates labeled `x`, `y`, `z` in the junction CSV are the stored mesh coordinates; do not silently swap them into conventional image XYZ.

| Stage | Method and retained defaults | Main products |
| --- | --- | --- |
| 00 | Set Z depth to 60 slices; center on the largest OMM connected component; zero pixels outside a 1 µm radius in the image plane, preserving the canvas dimensions | `scaled/*.tif` or `.tiff` |
| 01 | Marching cubes at 2 nm spacing, isolevel 127, step 1; extend first/last slices by 5 slices each and pad with a 1-voxel black border; restore the original coordinate origin | `scaled/meshes/OMM.stl`, `IBM.stl`, `IMS.stl`, `ER.stl`, `Ribo.stl` |
| 02 | Blender voxel remesh, followed by triangulation: 4 nm voxels for OMM/IBM/IMS/ER and 2 nm for ribosomes | `*_remeshed.stl` |
| 03 | Intersect meshes with a thickness box spanning axis 0 from 0 to 0.12 µm, using a 0.01 µm margin on the other axes; fuse vertices at 6 decimal places; use Manifold boolean intersection | `OMM_vol.stl`, `IBM_vol.stl`, `IMS_vol.stl`, `ER_vol.stl`, `Ribo_vol.stl` |
| 04 | Manifold boolean difference `IBM_vol − IMS_vol`; retain components with at least 100 vertices | `Matrix_vol.stl` |
| 05 | BlendGAMer MDSB curvature, K1, smoothing iteration 1; preserve world-space vertex coordinates with values | `smooth_k1` pickle payload |
| 06 | Remove faces near the crop box (0.005 µm threshold), separate matrix faces by proximity to temporary IBM vertices (0.008 µm), remap K1 by coordinates (tolerance `1e-6` µm), and retain high-curvature vertices with K1 strictly greater than 70 µm⁻¹ | OMM/IBM/CM/ER surfaces, ER contact surfaces, CM curvature array and high-K1 surface |
| 07 | Find shared CM/IBM edges after rounding coordinates to 9 decimals; retain edge components of size ≥15; split candidates using proximity 0.01 µm and a graph path of ≥10 edges; retain junctions assigned to a CM component with ≥500 faces | Per-tomogram `CJ.csv` and combined `CJ_all.csv` |
| 08 | Measure OMM volume, OMM area, crista membrane area, high-K1 area, and total IMM area | `mesh_summary.csv` |
| 09 | Measure OMM, ER, and ER–mitochondria contact surface areas | `ERMCS_mesh_summary.csv` |
| 10 | Measure total ribosome mesh volume and connected components; count retained components with volume strictly greater than 2,000 nm³ | `Ribo_mesh_summary.csv` |
| 11 | Fixed-FOV cristae screenshots | `cristae snaps/*.tif` |
| 12 | Fixed-FOV crista curvature screenshots | `curvature snaps/*.tif` |
| 13 | Fixed-FOV ER–mitochondria contact screenshots with ribosomes when available | `ERMCS snaps/*.tif` |

Stage 00 changes the depth to 60 slices rather than inferring physical calibration from TIFF metadata. For depths greater than 60 it takes the central slices; for smaller depths it interpolates along Z. OMM uses the binary resampling branch. The supplied filename detector lowercases stems but compares its segmentation keywords case-sensitively, so the other named masks follow the image interpolation/dtype-preservation branch. This supplied behavior is retained. Subsequent marching cubes uses isolevel 127 on the scaled uint8 values. The 60-slice depth and 2 nm mesh spacing jointly define the 0.12 µm slab used later.

For ER contacts, stage 06 approximates distance from ER vertices to OMM **vertices plus triangle centers**, rather than evaluating exact triangle-to-triangle distance. An ER face is classified as contact if **any** of its vertices is within 0.02 µm of those OMM reference points. The complementary ER faces form `nonERMCS_surf.stl`. If either subset is empty, its STL is not created by the supplied routine.

The high-K1 surface retains only faces whose three vertices all exceed the curvature threshold; it is not a continuous isocurvature contour. Stage 06 reports a failure when no vertices exceed the fixed threshold. Measurement routines retain their existing treatment of absent files, including NaN area/volume entries and zero ribosome output when `Ribo_vol.stl` is absent. Those outputs must be interpreted alongside the per-stage processing tables.

## Generated files and units

All per-tomogram mesh intermediates are written in `<condition>/<tomogram>/scaled/meshes`. Surface outputs include `OMM_surf.stl`, `IBM_surf.stl`, `CM_surf.stl`, `ER_surf.stl`, `ERMCS_surf.stl`, `nonERMCS_surf.stl`, and `CM_surf_highk1.stl`. `CM_surf_smooth_k1.pkl` contains values aligned to the reloaded `CM_surf.stl` vertices. Keep the curvature payloads with their corresponding meshes; they cannot be interchanged between samples or reordered meshes.

| Root-level table | Measurement columns and units |
| --- | --- |
| `CJ_all.csv` | `Date`, `Condition`, `Mito`, `Folder`, `Junction_ID`, and stored mesh coordinates `x`, `y`, `z` in µm |
| `mesh_summary.csv` | `OMM_vol_um3` in µm³; `OMM_surf_um2`, `CM_surf_um2`, `CM_highk1_um2`, `IMM_surf_um2` in µm² |
| `ERMCS_mesh_summary.csv` | `OMM_surf_um2`, `ER_surf_um2`, `ERMCS_surf_um2` in µm² |
| `Ribo_mesh_summary.csv` | `Ribo_vol_nm3`, `Retained_Ribo_components`, `Smallest_Ribo_vol_nm3`, `Average_Smallest_Ribo_vol_nm3`, `n_ribosomes` |

All three measurement summaries include `Date`, `Condition`, and `Mito`. Values are rounded to 3 decimals by the original measurement routines. IMM area is the sum of the separately rounded CM and IBM areas. Mitochondrial volume uses the signed `trimesh` volume directly; ribosome volume uses its absolute magnitude and converts µm³ to nm³. `Ribo_vol_nm3` includes the **entire** ribosome mesh; the 2,000 nm³ component threshold affects the retained count and smallest retained component, not the total volume. `Average_Smallest_Ribo_vol_nm3` is the mean of positive per-sample smallest retained volumes across the complete processed dataset, including both conditions. `n_ribosomes` is the rounded ratio of total ribosome volume to that shared average; it is a volume-based estimate, distinct from `Retained_Ribo_components`. If no positive smallest volumes exist, the average and estimated counts are zero.

When no junctions are found, each sample still receives a header-only `CJ.csv`; the supplied combined-table writer creates an empty, headerless `CJ_all.csv`.

Stage processing/failure CSVs are saved under `scripts`, except screenshot manifests/failures, which are saved in their respective screenshot folders. These are operational outputs from the analysis, not extra measurement tables.

## Screenshots

The standard renderers save **1920 × 1440 (4:3)** TIFF screenshots with a **2 µm horizontal field of view**, viewing the `yz` plane from the positive normal direction. A parallel camera uses the in-plane principal axis to orient CM horizontally. Default centering uses the combined visible structures; the default orientation uses CM. The mesh `yz` plane corresponds to the original image's Y/X plane because mesh axis 0 stores the tomogram thickness coordinate.

Cristae are yellow `(0.9686, 0.7686, 0.2627)`, ER contact/noncontact surfaces use blue `(0.0, 0.447, 0.698)`, and ribosomes use magenta `(0.780, 0.082, 0.522)`. Cristae/curvature renderers use OMM opacity 0.5 and IBM opacity 0.25; the ER-contact renderer uses 0.25 for both, CM/contact opacity 1.0, noncontact ER opacity 0.3, and ribosome opacity 1.0. Curvature display retains its `YlOrRd` colormap, 0–80 µm⁻¹ scale, and smoothing sigma 0.1. Screenshots use the supplied transparent-background export. Scene construction, lighting, colors, camera logic, and TIFF export follow the supplied renderers.

Filenames are sequential labels such as `Control 1.tif`, assigned within each condition after sorting samples by date and mitochondrial number. The manifest maps labels back to source paths. Existing screenshots are overwritten by default. The default pixel dimensions and the actual aspect-ratio validation are 4:3; some original error text still says 16:9, but 16:9 sizes are not accepted by that validation.

To change a rendering option, call the stage directly; for example:

```powershell
.\.venv\Scripts\python.exe scripts/11_batch_visualize_cristae_screenshot_2um.py --plane yz --horizontal-fov-um 2.0 --width 1920 --height 1440
```

Use `--help` on each renderer for its supported options. A functioning VTK/OpenGL rendering backend is required even for off-screen screenshots. On a headless Linux system, use a VTK build with an available EGL/OSMesa backend or a working X display; install the appropriate graphics libraries for that environment. Numerical stages 00–10 do not require the screenshot backend.

## Optional supplied scripts

Neither optional script is executed by the default pipeline:

- **`08b_measure_cristae_basic.py`** is the supplied `08_measure_cristae_props copy` variant. It writes only OMM volume, CM area, and high-K1 area. Run with `run_pipeline.py --only 08b`. It writes the same `mesh_summary.csv` and processing filenames as stage 08, so it replaces the fuller table if run afterward.
- **`13b_visualize_ERMCS_screenshot_2um_control1_2.py`** renders the first two sorted Control samples together. Run with `run_pipeline.py --only 13b`. It requires two Control folders and saves `ERMCS snaps/Control 1 and 2.tif` with its own manifest and failure CSV. It uses the supplied combined scene, not a newly aligned or registered mesh arrangement.

## Execution settings and reruns

`settings.py` controls Python worker count, concurrent Blender instances, and an optional Blender path. Both worker defaults remain **8**. Each process can hold substantial geometry in memory; reducing those counts changes concurrency without changing numerical thresholds. A machine with less memory can run the same stages with fewer workers.

Numerical parameters remain in the settings section of each numbered script. With the documented sample layout and installed dependencies, no source edits are required. Existing outputs are generally regenerated because the supplied skip-existing settings are false. Restarting an intermediate stage reuses earlier files as they are; rerun downstream stages after changing upstream inputs. Keep different analyses in separate project copies to avoid mixing intermediate meshes, curvature arrays, and summaries.

The `.gitignore` retains code and empty data folders while excluding merged TIFFs, generated meshes/tables/screenshots, local environments, and machine-specific Blender configuration. `source_files.json` maps every supplied script to its packaged filename.
