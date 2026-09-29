# Mitochondria remodel near synapses to boost metabolic capacity and fuel plasticity

Analysis code for **Shah et al., _Mitochondria remodel near synapses to boost metabolic capacity and fuel plasticity_**, bioRxiv (2026). [Read the current preprint (v3)](https://www.biorxiv.org/content/10.1101/2025.08.27.672715v3).

This repository organizes image processing, 3D reconstruction, quantitative analysis, and figure generation by their corresponding manuscript figures. It spans live-cell confocal imaging, connectomics, mitochondrial and spine ATP measurements, electron tomography, 3D DNA-PAINT, and mitochondrial protein kinase A inhibition experiments.

## Find the analysis for a figure

| Figure | Analysis and measurements |
| --- | --- |
| **Fig. 1** | Confocal analysis of juxtaspinal mitochondrial area and connectomics analysis of mitochondrial volume near synapses. |
| **Fig. 2** | Mitochondrial ATP and spine ATP measurements. |
| **Fig. 3** | Electron-tomogram analysis of the inner mitochondrial membrane (IMM), cristae membranes (CM), cristae junctions (CJ), and high-curvature (high-k1) cristae membrane regions. |
| **Fig. 4** | 3D DNA-PAINT analysis of MIC60 and ATP synthase. |
| **Fig. 5** | Mitochondria-ER contact site (MERCS) and ribosome cluster analysis using the electron tomograms analyzed in Fig. 3. |
| **Fig. 6** | Confocal mitochondrial morphology, mitochondrial ATP, and spine size analyses in mitochondrial PKI experiments. |
| **Fig. S4** | Deep-learning pipeline for segmentation of the mitochondrial intermembrane space (IMS), outer mitochondrial membrane (OMM), inner boundary membrane (IBM), endoplasmic reticulum (ER), and ribosomes in electron tomograms. |

## Repository structure

### `Image and Mesh Analysis/`

Image-processing and 3D analysis pipelines for confocal microscopy, connectomics, electron tomography, and 3D DNA-PAINT. These scripts cover segmentation, quantitative measurements, and mesh generation and analysis for the structures shown in the manuscript. The Fig. S4 segmentation pipeline produces labels used in the electron-tomogram analyses; the Fig. 3 and Fig. 5 analyses draw on the same tomograms.

### `Statistics and Plots/`

Scripts for the quantitative analyses, statistical tests, and plots associated with the manuscript figures and supplementary figures. Figure-specific subfolders provide the entry point for locating the code behind each graph.

## Citation

If you use code from this repository, please cite:

Shah, M., et al. (2026). *Mitochondria remodel near synapses to boost metabolic capacity and fuel plasticity*. bioRxiv. https://doi.org/10.1101/2025.08.27.672715

