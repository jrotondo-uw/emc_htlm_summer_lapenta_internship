#!/bin/bash

# Configurable paths
TEMPLATE_YAML="/work/noaa/da/jrotondo/JediCode/hybrid_tlm/fv3-jedi-htlm/fv3_htlm.yaml"
EXECUTABLE="/work/noaa/da/jrotondo/JediCode/hybrid_tlm/jedi-bundle/build/bin/fv3jedi_tlm_forecast.x"
BASE_WORKDIR="/work/noaa/da/jrotondo/JediCode/hybrid_tlm/fv3-jedi-htlm"

for i in $(seq 2 80); do
  MEM_NUM=$(printf "%03d" $i)      # Always mem001 to mem080
  MEM="mem${MEM_NUM}"             # mem001, mem002, ..., mem080

  WORKDIR="${BASE_WORKDIR}/run_${MEM}"
  YAML_FILE="${WORKDIR}/fv3_htlm_${MEM}.yaml"

  mkdir -p "$WORKDIR"

  # Replace mem002 (template input path) and mem2 (output filename) with correct memNNN
  sed \
    -e "s|mem002|${MEM}|g" \
    -e "s|htlm_forecast_cubed_sphere_mem2_six_hours_no_tlm.nc|htlm_forecast_cubed_sphere_${MEM}_six_hours_no_tlm.nc|g" \
    "$TEMPLATE_YAML" > "$YAML_FILE"

  echo "Running HTLM forecast for ${MEM}..."

  srun -n 216 "$EXECUTABLE" "$YAML_FILE"
done
