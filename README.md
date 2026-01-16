# AI4Arctic Sea Ice

<p align="center">
  <img src="assets/ai4arctic-sea-ice.png" alt="AI4Arctic Sea Ice" width="500"/>
</p>

## Overview

This repository explores **sea-ice concentration estimation** from **Sentinel-1 SAR imagery** using the **AI4Arctic dataset**.  
The project addresses a **weakly supervised learning** problem, where sea-ice labels are defined at the polygon (patch) level rather than at the pixel level.

## Data

- **Dataset**: AI4Arctic (Danish Meteorological Institute / Technical University of Denmark)
- **Inputs**:
  - Sentinel-1 SAR images (HH, HV polarizations)
- **Labels**:
  - Ice-chart polygons (SIGRID3 codes)
  - Sea-ice concentration and ice-type attributes

## Objective

The objective is to infer **pixel-level sea-ice information** from **patch-level ice-chart annotations**, in order to better characterize the spatial distribution of ice and water at high resolution.

## Methods

- Deep learning models (CNN-based)
- Patch-based training strategy
- Weakly supervised semantic segmentation

## Academic Context

This project is carried out as part of a **Deep Learning for Computer Vision lab** within the  
**Mastère Spécialisé Intelligence Artificielle Multimodale**,  
Télécom Paris.

## Authors

- Julien Gimenez  
- Reda Elwaradi  
- Mehdi Ait Hamma  
- Stephane Hordoir  

## Status

Work in progress
