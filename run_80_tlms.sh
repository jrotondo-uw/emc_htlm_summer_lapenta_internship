#!/bin/bash

# Paths
TEMPLATE_YAML="/work/noaa/da/jrotondo/JediCode/hybrid_tlm/fv3-jedi-htlm/tlm_forecast.yaml"
EXECUTABLE="/work/noaa/da/jrotondo/JediCode/hybrid_tlm/jedi-bundle/build/bin/fv3jedi_tlm_forecast.x"
BASE_WORKDIR="/work/noaa/da/jrotondo/JediCode/hybrid_tlm/fv3-jedi-htlm"

for i in $(seq -w 2 80); do
  MEM="mem$(printf "%03d" $i)"  # mem002, mem003, ..., mem080
  WORKDIR="${BASE_WORKDIR}/run_${MEM}"
  YAML_FILE="${WORKDIR}/tlm_forecast_${MEM}.yaml"

  mkdir -p "$WORKDIR"

  # Generate YAML with correct member and output filename
  sed \
    -e "s|mem002|${MEM}|g" \
    -e "s|tlm_forecast_cubed_sphere_mem2_six_hours.nc|tlm_forecast_cubed_sphere_${MEM}_six_hours.nc|g" \
    "$TEMPLATE_YAML" > "$YAML_FILE"

  echo "Running forecast for ${MEM}..."

  # Run the job
  srun -n 216 "$EXECUTABLE" "$YAML_FILE"
done
