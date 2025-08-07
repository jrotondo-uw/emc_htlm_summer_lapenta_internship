#!/bin/bash
set -euo pipefail

# Load required modules (adjust if necessary)
source /home/dholdawa/jedi_modules/modules_int_spack-stack-1.9.1
module load python/3.10.8

# Activate your Python virtual environment
source "$HOME/jedi_pyenv/bin/activate"

# Full path to the Python script
PY_SCRIPT="/work/noaa/da/jrotondo/JediCode/hybrid_tlm/fv3-jedi-htlm/py_scripts/tlm_htlm_one_hour_map_comparison.py"

echo "Running TLM vs HTLM ensemble map comparison..."

# Execute the Python script
python "$PY_SCRIPT"

echo "Map comparison plots generated successfully."
