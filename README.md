# Mitochondria remodel near synapses to boost metabolic capacity and fuel plasticity

This repository contains all analysis scripts and data-processing pipelines used in:

**Shah et. al. 2026**  
*Mitochondria structurally remodel near synapses to fuel the sustained energy demands of plasticity.*  
bioRxiv. [https://www.biorxiv.org/content/10.1101/2025.08.27.672715v1](https://www.biorxiv.org/content/10.1101/2025.08.27.672715v1)

Please cite this paper when using any of these analysis scripts or data-processing pipelines.

---

## Repository Structure

The repository contains two top-level folders:

### 1. Image and Mesh Analysis
Analysis pipelines for confocal and electron tomography (EM) data.

- **Fig 1, 7, S5**  
  Scripts for analysis of confocal microscopy data, including mitochondrial and spine segmentation, measurements for mito length, local mito area, high-width mito regions, and related measurements.

- **Fig 2, S2**  
  Deep-learning segmentation pipeline for mitochondrial intermembrane space (IMS) in electron tomograms (adaptable to other ultrastructures).  
  Includes scripts for generating meshes from segmentations, used for the 3D reconstructions shown in Fig 3, S3, 4, and 6.

- **Fig 3, S3, 4, 6**  
  Complete 3D mesh analysis and visualization tools for mitochondrial inner membrane density, cristae membrane density, cristae curvature, cristae junction analysis, ER–mitochondria contact sites (ERMCS), and ribosomal clusters.

### 2. Statistics and Plots
Scripts to reproduce all quantitative analyses and statistical figures presented in **Shah et al., 2025**.  
Each subfolder corresponds to individual figures and supplements.

