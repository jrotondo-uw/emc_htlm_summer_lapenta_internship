#!/bin/bash

# === Settings ===
SCRIPT="/work/noaa/da/jrotondo/JediCode/hybrid_tlm/fv3-jedi-htlm/plot_coeffs_map.py"
DATA_DIR="/work/noaa/da/jrotondo/JediCode/hybrid_tlm/jedi-bundle/build/fv3-jedi/test/Data/test_breakdown/one_hour/merged/"
PYTHON_VENV="$HOME/jedi_pyenv/bin/python"
MAX_JOBS=6

# === Function to check job count ===
function wait_for_jobs {
  while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
    sleep 1
  done
}

# === Loop over files ===
for file in "$DATA_DIR"/*.nc; do
  wait_for_jobs
  echo "Launching: $file"

  bash -c "
    source /home/dholdawa/jedi_modules/modules_int_spack-stack-1.9.1
    module load python/3.11.7
    $PYTHON_VENV $SCRIPT $file
  " &
done

wait
echo "Bestie, all jobs done!"
