# EMC HTLM Summer Lapenta Internship

This repository contains code, configuration files, and analysis tools developed during the 2025 NOAA Lapenta Internship. The project focuses on the development and evaluation of a Hybrid Tangent Linear Model (HTLM) within the FV3-JEDI data assimilation framework. The goal is to assess the skill and utility of HTLMs for ensemble-based sensitivity and hybrid variational methods.

## Repository Structure

```
├── YAML/                         # YAML configuration files for TLM/HTLM experiments
├── Namelists/                   # GFS namelist (.nml) input files
├── Scripts/                     # Shell scripts to run experiments and plotting
├── Python/                      # Python scripts for postprocessing and visualization
├── Notebooks/                   # Jupyter notebooks for exploratory data analysis
└── README.md                    # This file
```

## Key Components

- HTLM Generation: YAML files to generate HTLM perturbations across multiple ensemble members.
- TLM Comparison: Scripts to compute correlations and RMSE between TLM, HTLM, and HTLM-no-TLM forecasts.
- Merged Coefficient Analysis: Tools to merge MPI-split HTLM coefficients and visualize structure by vertical level.
- Workflow Automation: Shell scripts to automate 80-member runs for HTLM and TLM.
- Notebooks: Interactive summaries and diagnostics of coefficient structure and HTLM performance.

## Requirements

- Python 3.x
- Python packages: xarray, matplotlib, numpy, cartopy, scipy
- HPC system access (e.g., NOAA RDHPCS, NCAR Derecho)
- FV3-JEDI system and appropriate builds

## Usage

To generate HTLM coefficients:

```
cd Scripts/
bash run_80_htlms.sh
```

To merge and plot coefficient maps:

```
bash run_merge_htlm_coefficients_from_mpi_processes.sh
bash run_htlm_map_plotting_in_parallel.sh
```

To analyze TLM vs HTLM skill:

```
python Python/tlm_htlm_correlation_and_rmse_comparison_by_vertical_level.py
```

## Project Context

This work was conducted as part of the NOAA Lapenta Internship Program. It is a collaboration between the Environmental Modeling Center (EMC) and the Joint Effort for Data assimilation Integration (JEDI) team. The project investigates the development and application of HTLMs for use in hybrid 4D-EnVar systems.

## Contact

Joey Rotondo  
University of Washington  
Atmospheric and Climate Science PhD Student  
jrotondo@uw.edu

## License & Public Access

This project is released under the United States Government Work designation.

This material is based upon work supported by the National Oceanic and Atmospheric Administration (NOAA). As a work of the U.S. Government, this repository is in the public domain within the United States. Users are free to use, distribute, or modify this code and associated content, with proper acknowledgment.
"""
