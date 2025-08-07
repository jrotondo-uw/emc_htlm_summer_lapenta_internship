#!/bin/bash
# run_regrid_all.sh

INPUT_DIR="/work/noaa/da/jrotondo/JediCode/hybrid_tlm/jedi-bundle/build/fv3-jedi/test/Data/six_hour_test/3_lvls/"
OUTPUT_DIR="${INPUT_DIR}/merged"

mkdir -p $OUTPUT_DIR

# Find unique timestamps
timestamps=$(ls ${INPUT_DIR}/hybrid_linear_model_coeffs_*.nc | \
  sed -E 's|.*hybrid_linear_model_coeffs_([0-9TZ]+)_.*|\1|' | sort -u)

for ts in $timestamps; do
  echo "🔄 Merging files for timestamp $ts"
  python regrid_htlm.py $ts $INPUT_DIR $OUTPUT_DIR
done
