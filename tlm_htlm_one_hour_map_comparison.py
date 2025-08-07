#import xarray as xr
#import numpy as np
#import matplotlib.pyplot as plt
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
#from scipy.interpolate import griddata
#import os
#
## === Plotting function ===
#def plot_cubed_sphere_field(var, lat, lon, level_idx=39, title="", outname=None, vmin=None, vmax=None):
#    data = var.isel(pfull=level_idx).values
#    lat_vals = lat.values
#    lon_vals = lon.values
#
#    lon_flat = lon_vals.reshape(-1)
#    lat_flat = lat_vals.reshape(-1)
#    data_flat = data.reshape(-1)
#
#    lon_ext = np.concatenate([lon_flat, lon_flat - 360])
#    lat_ext = np.concatenate([lat_flat, lat_flat])
#    data_ext = np.concatenate([data_flat, data_flat])
#
#    lon_grid, lat_grid = np.meshgrid(np.linspace(-180, 180, 720), np.linspace(-90, 90, 361))
#    field_interp = griddata(
#        (lon_ext, lat_ext), data_ext,
#        (lon_grid, lat_grid),
#        method='linear'
#    )
#
#    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})
#    pcm = ax.pcolormesh(lon_grid, lat_grid, field_interp,
#                        cmap="RdBu_r", shading="auto", transform=ccrs.PlateCarree(),
#                        vmin=vmin, vmax=vmax)
#    ax.set_title(title)
#    ax.coastlines()
#    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
#    fig.colorbar(pcm, ax=ax, orientation="vertical", label="m/s")
#
#    if outname:
#        plt.savefig(outname, dpi=300)
#        print(f"✅ Saved: {outname}")
#        plt.close()
#    else:
#        plt.show()
#
## === File paths ===
#tlm_dir = '/work/noaa/da/jrotondo/JediCode/hybrid_tlm/fv3-jedi-htlm/tlm_output'
#htlm_dir = '/work/noaa/da/jrotondo/JediCode/hybrid_tlm/fv3-jedi-htlm/htlm_output'
#truth_file = '/work/noaa/da/jrotondo/HybridTlm/Training/C48/Data/diff_from_control/diff_mem002_cubed_sphere_grid_atmf004.nc'
#fig_dir = '/work/noaa/da/jrotondo/JediCode/hybrid_tlm/fv3-jedi-htlm/figs'
#
#os.makedirs(fig_dir, exist_ok=True)
#
## === Load truth delta ===
#ds_truth = xr.open_dataset(truth_file)
#ugrd_true = ds_truth['ugrd'].astype(np.float64).isel(time=0)
#ugrd_true = xr.where(ugrd_true < 1e20, ugrd_true, np.nan)
#lat = ds_truth['lat']
#lon = ds_truth['lon']
#
## === Load and average ensembles ===
#def load_ensemble_mean(varname, base_dir, prefix):
#    members = []
#    for i in range(2, 81):
#        mem_id = f"mem{str(i).zfill(3)}"
#        filename = f"{prefix}_forecast_cubed_sphere_{mem_id}_one_hour.nc"
#        full_path = os.path.join(base_dir, filename)
#        if not os.path.exists(full_path):
#            print(f"⚠️ WARNING: Missing {full_path}, skipping...")
#            continue
#        ds = xr.open_dataset(full_path)
#        ds = ds.assign_coords(pfull=ds_truth['pfull'])  # Align vertical levels
#        members.append(ds[varname].astype(np.float64).isel(time=0).expand_dims(ensemble=[i]))
#    if not members:
#        raise RuntimeError(f"No {prefix.upper()} members were loaded!")
#    return xr.concat(members, dim='ensemble').mean(dim='ensemble')
#
## Load ensemble means
#ugrd_tlm_mean = load_ensemble_mean('ugrd', tlm_dir, 'tlm')
#ugrd_htlm_mean = load_ensemble_mean('ugrd', htlm_dir, 'htlm')
#
## === Compute dy fields ===
#dy_true = ugrd_true
#dy_tlm = ugrd_tlm_mean
#dy_htlm = ugrd_htlm_mean
#
#dy_diff_tlm = dy_tlm - dy_true
#dy_diff_htlm = dy_htlm - dy_true
#
## === Plot for each level ===
#levels = [39, 79, 119]  # Python 0-indexed for levels 40, 80, 120
#for level_idx in levels:
#    level_num = level_idx + 1
#
#    # Extract fields at level
#    true_vals = dy_true.isel(pfull=level_idx)
#    tlm_vals = dy_tlm.isel(pfull=level_idx)
#    htlm_vals = dy_htlm.isel(pfull=level_idx)
#
#    diff_tlm_vals = dy_diff_tlm.isel(pfull=level_idx)
#    diff_htlm_vals = dy_diff_htlm.isel(pfull=level_idx)
#
#    # Compute common vmin/vmax
#    all_dy_vals = xr.concat([true_vals, tlm_vals, htlm_vals], dim='comparison')
#    dy_min = float(all_dy_vals.min())
#    dy_max = float(all_dy_vals.max())
#    dy_bound = max(abs(dy_min), abs(dy_max))
#
#    all_diff_vals = xr.concat([diff_tlm_vals, diff_htlm_vals], dim='comparison')
#    diff_min = float(all_diff_vals.min())
#    diff_max = float(all_diff_vals.max())
#    diff_bound = max(abs(diff_min), abs(diff_max))
#
#    # Plot Δy
#    plot_cubed_sphere_field(dy_true, lat, lon, level_idx,
#        title=f"Δy True (U Wind, Level {level_num})",
#        outname=f"{fig_dir}/dy_true_L{level_num}.png",
#        vmin=-dy_bound, vmax=dy_bound)
#
#    plot_cubed_sphere_field(dy_tlm, lat, lon, level_idx,
#        title=f"Δy TLM Mean (U Wind, Level {level_num})",
#        outname=f"{fig_dir}/dy_tlm_L{level_num}.png",
#        vmin=-dy_bound, vmax=dy_bound)
#
#    plot_cubed_sphere_field(dy_htlm, lat, lon, level_idx,
#        title=f"Δy HTLM Mean (U Wind, Level {level_num})",
#        outname=f"{fig_dir}/dy_htlm_L{level_num}.png",
#        vmin=-dy_bound, vmax=dy_bound)
#
#    # Plot Δy differences (TLM - True, HTLM - True)
#    plot_cubed_sphere_field(dy_diff_tlm, lat, lon, level_idx,
#        title=f"TLM Mean - True (U Wind, Level {level_num})",
#        outname=f"{fig_dir}/dy_diff_tlm_L{level_num}.png",
#        vmin=-diff_bound, vmax=diff_bound)
#
#    plot_cubed_sphere_field(dy_diff_htlm, lat, lon, level_idx,
#        title=f"HTLM Mean - True (U Wind, Level {level_num})",
#        outname=f"{fig_dir}/dy_diff_htlm_L{level_num}.png",
#        vmin=-diff_bound, vmax=diff_bound)

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
import os

# === Toggle MSLP overlay ===
plot_mslp = True  # Set to False to disable MSLP contour overlays

# === Toggle layout style ===
stacked_layout = True  # True = 2 top, 1 bottom; False = 3 horizontal

# === Helper function: interpolate to PlateCarree ===
def interp_to_platecarree(var, lat, lon):
    var_vals = var.values
    lat_vals = lat.values
    lon_vals = lon.values

    lon_flat = lon_vals.reshape(-1)
    lat_flat = lat_vals.reshape(-1)
    var_flat = var_vals.reshape(-1)

    lon_ext = np.concatenate([lon_flat, lon_flat - 360])
    lat_ext = np.concatenate([lat_flat, lat_flat])
    var_ext = np.concatenate([var_flat, var_flat])

    lon_grid, lat_grid = np.meshgrid(np.linspace(-180, 180, 720), np.linspace(-90, 90, 361))
    field_interp = griddata((lon_ext, lat_ext), var_ext, (lon_grid, lat_grid), method='linear')
    return lon_grid, lat_grid, field_interp

# === File paths ===
tlm_dir = '/work/noaa/da/jrotondo/JediCode/hybrid_tlm/fv3-jedi-htlm/tlm_output'
htlm_dir = '/work/noaa/da/jrotondo/JediCode/hybrid_tlm/fv3-jedi-htlm/htlm_output'
truth_file = '/work/noaa/da/jrotondo/HybridTlm/Training/C48/Data/diff_from_control/diff_mem002_cubed_sphere_grid_atmf009.nc'
fig_out = '/work/noaa/da/jrotondo/JediCode/hybrid_tlm/fv3-jedi-htlm/figs/dy_6_hr_panel_level118_for_powerpoint_with_pressure.png'

file_control = '/work/noaa/da/jrotondo/HybridTlm/Training/C48/Data/ens/control/mem001/cubed_sphere_grid_atmf009.nc'

# === Load truth Δy ===
ds_truth = xr.open_dataset(truth_file)
ugrd_true = ds_truth['ugrd'].astype(np.float64).isel(time=0)
ugrd_true = xr.where(ugrd_true < 1e20, ugrd_true, np.nan)
lat = ds_truth['lat']
lon = ds_truth['lon']

# === Load ensemble means ===
def load_ensemble_mean(varname, base_dir, prefix):
    members = []
    for i in range(2, 81):
        mem_id = f"mem{str(i).zfill(3)}"
        filename = f"{prefix}_forecast_cubed_sphere_{mem_id}_six_hours.nc"
        full_path = os.path.join(base_dir, filename)
        if not os.path.exists(full_path):
            continue
        ds = xr.open_dataset(full_path)
        ds = ds.assign_coords(pfull=ds_truth['pfull'])
        members.append(ds[varname].astype(np.float64).isel(time=0).expand_dims(ensemble=[i]))
    return xr.concat(members, dim='ensemble').mean(dim='ensemble')

ugrd_tlm_mean = load_ensemble_mean('ugrd', tlm_dir, 'tlm')
ugrd_htlm_mean = load_ensemble_mean('ugrd', htlm_dir, 'htlm')

# === Extract level 118
level_idx = 118
dy_true = ugrd_true.isel(pfull=level_idx)
dy_tlm = ugrd_tlm_mean.isel(pfull=level_idx)
dy_htlm = ugrd_htlm_mean.isel(pfull=level_idx)

# === Interpolate fields
lon_grid, lat_grid, field_true = interp_to_platecarree(dy_true, lat, lon)
_, _, field_tlm = interp_to_platecarree(dy_tlm, lat, lon)
_, _, field_htlm = interp_to_platecarree(dy_htlm, lat, lon)

# === Load and process MSLP if enabled ===
if plot_mslp:
    ds_control = xr.open_dataset(file_control)
    pressfc = ds_control['pressfc'].isel(time=0)
    hgtsfc = ds_control['hgtsfc'].isel(time=0)  # in meters

    # Use surface level temperature (e.g., level 0 in pfull)
    tmp = ds_control['tmp'].isel(time=0, pfull=-1)  # near-surface temperature

    # Mask invalid values
    pressfc = xr.where(pressfc < 1e20, pressfc, np.nan)
    hgtsfc = xr.where(hgtsfc < 1e20, hgtsfc, np.nan)
    tmp = xr.where(tmp < 1e20, tmp, np.nan)

    # Constants
    Rd = 287.05  # J/(kg·K)
    g = 9.80665  # m/s²

    # Compute MSLP
    mslp = pressfc * np.exp(g * hgtsfc / (Rd * tmp))

    # Interpolation to global grid
    lat_mslp = ds_control['lat']  # [tile, y, x]
    lon_mslp = ds_control['lon']  # [tile, y, x]

    lon_flat = lon_mslp.values.reshape(-1)
    lat_flat = lat_mslp.values.reshape(-1)
    mslp_flat = mslp.values.reshape(-1)

    lon_ext = np.concatenate([lon_flat, lon_flat - 360])
    lat_ext = np.concatenate([lat_flat, lat_flat])
    mslp_ext = np.concatenate([mslp_flat, mslp_flat])

    lon_grid, lat_grid = np.meshgrid(np.linspace(-180, 180, 720), np.linspace(-90, 90, 361))
    mslp_interp = griddata((lon_ext, lat_ext), mslp_ext, (lon_grid, lat_grid), method='linear')

# === Plot panel
vmin = min(np.nanmin(field_true), np.nanmin(field_tlm), np.nanmin(field_htlm))
vmax = max(np.nanmax(field_true), np.nanmax(field_tlm), np.nanmax(field_htlm))
bound = max(0.8 * abs(vmin), 0.8 * abs(vmax))

titles = ["Δy TLM Mean", "Δy HTLM Mean", "Δy True (U Wind)"]
fields = [field_tlm, field_htlm, field_true]

if stacked_layout:
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.75], hspace=0)

    # Top: 2 equal-width axes
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())

    # Bottom: 1 full-width axis with same height
    ax3 = fig.add_subplot(gs[1, :], projection=ccrs.PlateCarree())

    axs = [ax1, ax2, ax3]
else:
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={'projection': ccrs.PlateCarree()})

# Plot loop
for ax, field, title in zip(axs, fields, titles):
    pcm = ax.pcolormesh(lon_grid, lat_grid, field, cmap='RdBu_r', vmin=-bound, vmax=bound, shading='auto')
    ax.set_title(title)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    
    # Add MSLP contours if enabled
    if plot_mslp:
        try:
            contours = ax.contour(lon_grid, lat_grid, mslp_interp / 100, levels=20, colors='black', linewidths=0.6)
            ax.clabel(contours, inline=True, fontsize=6, fmt='%1.0f')
        except Exception as e:
            print(f"Could not plot MSLP contours: {e}")

# Adjust layout and add colorbar
fig.subplots_adjust(right=0.86, wspace=0.05)
cbar_ax = fig.add_axes([0.88, 0.2, 0.015, 0.6])
fig.colorbar(pcm, cax=cbar_ax, label='m/s')

# Save and close
plt.savefig(fig_out, dpi=300, bbox_inches='tight')
plt.close()