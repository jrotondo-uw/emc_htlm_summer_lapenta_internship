import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
import sys
import os

print("Bestie is using Python from:", sys.executable)

# === Get inputs ===
if len(sys.argv) < 2:
    print("Usage: python plot_coeffs_map.py <NetCDF file path>")
    sys.exit(1)

filepath = sys.argv[1]
basename = os.path.basename(filepath).replace(".nc", "")
varname = "cloud_liquid_water"  # Change this to the variable you want to plot
# Instead of [0, 1, 2], use last three
modes_to_plot = [0, 1, 2]
levels_to_plot = ['mean', 'lowest', 'weather']

print(f"Bestie is loading: {filepath}")

# === Load Dataset ===
ds = xr.open_dataset(filepath)
var_dims = ds[varname].dims
nz_dim = [d for d in var_dims if d.startswith("nz_")][0]
nv_dim = [d for d in var_dims if d.startswith("nv_")][0]

lat_1d = ds["lat"].values
lon_1d = ds["lon"].values

# === Duplicate points across 360° seam ===
print("Bestie is duplicating longitudes for seam wrapping...")
lon_extended = np.concatenate([lon_1d, lon_1d - 360])
lat_extended = np.concatenate([lat_1d, lat_1d])

# === Create Regular Grid ===
lon_target = np.linspace(-180, 180, 720)
lat_target = np.linspace(-90, 90, 361)
lon_grid, lat_grid = np.meshgrid(lon_target, lat_target)

# === Ensure subfolder for standalone plots exists ===
first_mode_dir = "first_mode"
os.makedirs(first_mode_dir, exist_ok=True)

# === Create folder for all other plots ===
test_figs_dir = "test_figs"
os.makedirs(test_figs_dir, exist_ok=True)

print("=== Scanning for max magnitude across all modes...")

max_magnitude = -np.inf
max_mode = None
max_interp_field = None
max_level_type = None

mode_magnitudes = []

level_type_scan = "weather"  # Can change this

for mode in range(ds.dims[nv_dim]):
    if level_type_scan == 'mean':
        field_base = ds[varname].isel({nv_dim: mode}).mean(dim=nz_dim).values
    elif level_type_scan == 'lowest':
        field_base = ds[varname].isel({nv_dim: mode, nz_dim: -1}).values
    elif level_type_scan == 'weather':
        field_base = ds[varname].isel({nv_dim: mode, nz_dim: 110}).values
    else:
        raise ValueError("Unknown level_type")

    values_extended = np.concatenate([field_base, field_base])

    interp_field = griddata(
        points=(lon_extended, lat_extended),
        values=values_extended,
        xi=(lon_grid, lat_grid),
        method='linear'
    )

    # === Identify NaN locations ===
    nan_mask = np.isnan(interp_field)
    num_nans = np.sum(nan_mask)

    if num_nans > 0:
        print(f"⚠️ {num_nans} NaNs found in interpolated field for mode {mode} at {level_type_scan} level")

        nan_indices = np.argwhere(nan_mask)
        i_vals = nan_indices[:, 0]
        j_vals = nan_indices[:, 1]
        lat_vals = lat_grid[i_vals, j_vals]
        lon_vals = lon_grid[i_vals, j_vals]

        # Save to .npz file
        nan_outfile = f"{basename}_nan_locations_mode{mode}_{level_type_scan}.npz"
        np.savez(nan_outfile, i=i_vals, j=j_vals, lat=lat_vals, lon=lon_vals)
        print(f"📁 Saved NaN (i, j, lat, lon) to: {nan_outfile}")

        # Show first few
        for idx in range(min(10, len(i_vals))):
            print(f"  → NaN at i={i_vals[idx]}, j={j_vals[idx]} → (lat={lat_vals[idx]:.2f}, lon={lon_vals[idx]:.2f})")
    else:
        print(f"✅ No NaNs found in mode {mode} ({level_type_scan} level)")

    max_val = np.nanmax(np.abs(interp_field))
    print(f"Mode {mode}: max |value| = {max_val:.3f}")
    mode_magnitudes.append(max_val)

    if max_val > max_magnitude:
        max_magnitude = max_val
        max_mode = mode
        max_interp_field = interp_field
        max_level_type = level_type_scan

# === Plot max |value| vs mode index ===
plt.figure(figsize=(8, 4))
plt.plot(range(len(mode_magnitudes)), mode_magnitudes, marker='o')
plt.title(f"Max |{varname}| vs Mode Index ({level_type_scan} level)")
plt.xlabel("Mode index (nv)")
plt.ylabel("Max |value|")
plt.grid(True)
plt.tight_layout()

basename = os.path.basename(filepath).replace(".nc", "")
plot_name = f"{basename}_mode_magnitude_scan_clw_{level_type_scan}.png"
plot_path = os.path.join(test_figs_dir, plot_name)
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"📈 Saved mode magnitude scan plot: {plot_path}")

print(f"\n🔥 Mode {max_mode} has the largest magnitude value: {max_magnitude:.3f} ({max_level_type} level)")

# === Loop over level types ===
for level_type in levels_to_plot:
    print(f"Now plotting: {level_type} levels")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10),
                             subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)},
                             constrained_layout=True)

    pcms = []

    for i, k_level in enumerate(modes_to_plot):
        print(f"Bestie is processing mode {k_level}...")

        # === Get the base field ===
        if level_type == 'mean':
            field_base = ds[varname].isel({nv_dim: k_level}).mean(dim=nz_dim).values
        elif level_type == 'lowest':
            field_base = ds[varname].isel({nv_dim: k_level, nz_dim: -1}).values
        elif level_type == 'weather':
            field_base = ds[varname].isel({nv_dim: k_level, nz_dim: 110}).values
        else:
            raise ValueError("Unknown level_type")

        values_extended = np.concatenate([field_base, field_base])

        interp_field = griddata(
            points=(lon_extended, lat_extended),
            values=values_extended,
            xi=(lon_grid, lat_grid),
            method='linear'
        )

        vmax = np.nanmax(np.abs(interp_field))

        ax = axes[i]
        pcm = ax.pcolormesh(
            lon_grid, lat_grid, interp_field,
            cmap="coolwarm", shading="auto", transform=ccrs.PlateCarree(),
            vmin=-0.8*vmax, vmax=0.8*vmax
        )

        # === Find and plot max magnitude (abs value) locations safely ===
        abs_interp = np.abs(interp_field)
        
        # Mask out NaNs
        valid_mask = np.isfinite(abs_interp)
        valid_values = abs_interp[valid_mask]
        
        if valid_values.size == 0:
            print("Warning: All values are NaN, skipping star plot.")
        else:
            max_val = np.nanmax(valid_values)

            max_locs = np.where(abs_interp == max_val)

            # Filter out junk near poles or corners (optional)
            max_lat_candidates = lat_grid[max_locs]
            max_lon_candidates = lon_grid[max_locs]

            # Example: keep only those between -80 and 80 latitude
            valid_idx = (max_lat_candidates > -80) & (max_lat_candidates < 80)

            max_lats = max_lat_candidates[valid_idx]
            max_lons = max_lon_candidates[valid_idx]

            # Plot only if valid points remain
            if max_lats.size > 0:
                ax.plot(max_lons, max_lats, marker='*', color='black', markersize=12, alpha = 0,
                        linestyle='None', transform=ccrs.PlateCarree(), label='Max Magnitude')
                ax.legend(loc='lower left')
            else:
                print("Note: All max values were near poles/corners and filtered.")

        if level_type == 'mean':
            title_suffix = "Mean over Vertical Levels"
        elif level_type == 'lowest':
            title_suffix = "Lowest Vertical Level"
        elif level_type == 'weather':
            title_suffix = "Weather Level (nz=110)"

        ax.set_title(f"{varname} | Coeff {k_level} | {title_suffix}")
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        pcms.append(pcm)

        # === Save standalone plot for mode 0 ===
        if k_level == 0:
            fig1, ax1 = plt.subplots(1, 1, figsize=(10, 4),
                                     subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
            pcm1 = ax1.pcolormesh(
                lon_grid, lat_grid, interp_field,
                cmap="coolwarm", shading="auto", transform=ccrs.PlateCarree(),
                vmin=-0.8 * vmax, vmax=0.8 * vmax
            )
            ax1.coastlines()
            ax1.add_feature(cfeature.BORDERS, linewidth=0.5)
            #ax1.set_title(f"{varname} | Coeff 0 | {title_suffix}")
            ax1.set_title(f"First-Order CLW HTLM Coefficient Mode | {title_suffix}")
            cbar = plt.colorbar(pcm1, ax=ax1, orientation="vertical", shrink=0.75, pad=0.02)
            #cbar.set_label(f"{varname} Coeff (Mode 0)")
            cbar.set_label("HTLM Coefficient Value")

            basename = os.path.basename(filepath).replace(".nc", "")
            suffix = {
                'mean': 'mean',
                'lowest': 'lowest',
                'weather': 'weather_level'
            }[level_type]
            standalone_name = f"{basename}_first_mode_clw_new_vmax_{suffix}.png"
            standalone_path = os.path.join(first_mode_dir, standalone_name)
            plt.savefig(standalone_path, dpi=300)
            plt.close(fig1)
            print(f"Bestie saved standalone mode 0 plot: {standalone_path}")

    # === Colorbar ===
    colorbar_label = {
        'mean': f"{varname} (mean over levels)",
        'lowest': f"{varname} (lowest level)",
        'weather': f"{varname} (weather level: nz=110)"
    }[level_type]

    fig.colorbar(pcms[0], ax=axes.ravel().tolist(), orientation="vertical",
                 label=colorbar_label, shrink=0.75, pad=0.02)

    # === Save the full 3-mode figure ===
    basename = os.path.basename(filepath).replace(".nc", "")
    suffix = {
        'mean': 'mean',
        'lowest': 'lowest',
        'weather': 'weather_level'
    }[level_type]
    outname = f"{basename}_coeff_plot_{suffix}_clw_new_vmax.png"
    outpath = os.path.join(test_figs_dir, outname)
    plt.savefig(outpath, dpi=300)
    print(f"Bestie saved: {outpath}")

# == CODE FOR WIND SPEED ==
#import xarray as xr
#import numpy as np
#import matplotlib.pyplot as plt
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
#from scipy.interpolate import griddata
#import sys
#import os
#
#print("Bestie is using Python from:", sys.executable)
#
## === Get inputs ===
#if len(sys.argv) < 2:
#    print("Usage: python plot_coeffs_map.py <NetCDF file path>")
#    sys.exit(1)
#
#filepath = sys.argv[1]
#east_var = "eastward_wind"
#north_var = "northward_wind"
#modes_to_plot = [0, 1, 2]
#levels_to_plot = ['mean', 'lowest', 'weather']
#
#print(f"Bestie is loading: {filepath}")
#
## === Load Dataset ===
#ds = xr.open_dataset(filepath)
#
## === Get dimensions from each variable ===
#east_dims = ds[east_var].dims
#north_dims = ds[north_var].dims
#
#nz_dim_east = [d for d in east_dims if d.startswith("nz_")][0]
#nv_dim_east = [d for d in east_dims if d.startswith("nv_")][0]
#
#nz_dim_north = [d for d in north_dims if d.startswith("nz_")][0]
#nv_dim_north = [d for d in north_dims if d.startswith("nv_")][0]
#
#lat_1d = ds["lat"].values
#lon_1d = ds["lon"].values
#
## === Duplicate points across 360° seam ===
#print("Bestie is duplicating longitudes for seam wrapping...")
#lon_extended = np.concatenate([lon_1d, lon_1d - 360])
#lat_extended = np.concatenate([lat_1d, lat_1d])
#
## === Create Regular Grid ===
#lon_target = np.linspace(-180, 180, 720)
#lat_target = np.linspace(-90, 90, 361)
#lon_grid, lat_grid = np.meshgrid(lon_target, lat_target)
#
## === Ensure subfolder for standalone plots exists ===
#first_mode_dir = "first_mode"
#os.makedirs(first_mode_dir, exist_ok=True)
#
## === Loop over level types ===
#for level_type in levels_to_plot:
#    print(f"Now plotting: {level_type} levels")
#
#    fig, axes = plt.subplots(3, 1, figsize=(14, 12),
#                             subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)},
#                             constrained_layout=True)
#
#    pcms = []
#
#    for i, k_level in enumerate(modes_to_plot):
#        print(f"Bestie is processing mode {k_level}...")
#
#        # === Get u and v components ===
#        if level_type == 'mean':
#            u = ds[east_var].isel({nv_dim_east: k_level}).mean(dim=nz_dim_east).values
#            v = ds[north_var].isel({nv_dim_north: k_level}).mean(dim=nz_dim_north).values
#        elif level_type == 'lowest':
#            u = ds[east_var].isel({nv_dim_east: k_level, nz_dim_east: -1}).values
#            v = ds[north_var].isel({nv_dim_north: k_level, nz_dim_north: -1}).values
#        elif level_type == 'weather':
#            u = ds[east_var].isel({nv_dim_east: k_level, nz_dim_east: 110}).values
#            v = ds[north_var].isel({nv_dim_north: k_level, nz_dim_north: 110}).values
#        else:
#            raise ValueError("Unknown level_type")
#
#        # === Compute wind speed ===
#        field_base = np.sqrt(u**2 + v**2)
#        values_extended = np.concatenate([field_base, field_base])
#
#        # === Interpolate to regular grid ===
#        interp_field = griddata(
#            points=(lon_extended, lat_extended),
#            values=values_extended,
#            xi=(lon_grid, lat_grid),
#            method='linear'
#        )
#
#        vmax = np.nanmax(np.abs(interp_field))
#
#        ax = axes[i]
#        pcm = ax.pcolormesh(
#            lon_grid, lat_grid, interp_field,
#            cmap="coolwarm", shading="auto", transform=ccrs.PlateCarree(),
#            vmin=-vmax, vmax=vmax
#        )
#
#        # === Title suffix ===
#        if level_type == 'mean':
#            title_suffix = "Mean over Vertical Levels"
#        elif level_type == 'lowest':
#            title_suffix = "Lowest Vertical Level"
#        elif level_type == 'weather':
#            title_suffix = "Weather Level (nz=110)"
#
#        ax.set_title(f"Wind Speed | Coeff {k_level} | {title_suffix}", fontsize=18)
#        ax.coastlines()
#        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
#        ax.tick_params(labelsize=14)
#        pcms.append(pcm)
#
#        # === Save standalone plot for mode 0 ===
#        if k_level == 0:
#            fig1, ax1 = plt.subplots(1, 1, figsize=(12, 5),
#                                     subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
#            pcm1 = ax1.pcolormesh(
#                lon_grid, lat_grid, interp_field,
#                cmap="coolwarm", shading="auto", transform=ccrs.PlateCarree(),
#                vmin=-0.8 * vmax, vmax=0.8 * vmax
#            )
#            ax1.coastlines()
#            ax1.add_feature(cfeature.BORDERS, linewidth=0.5)
#            ax1.set_title(f"First-Order Wind Speed HTLM Coefficient Mode | {title_suffix}", fontsize=20)
#            cbar = plt.colorbar(pcm1, ax=ax1, orientation="vertical", shrink=0.75, pad=0.02)
#            cbar.set_label("HTLM Coefficient Value", fontsize=16)
#            cbar.ax.tick_params(labelsize=14)
#            ax1.tick_params(labelsize=14)
#
#            basename = os.path.basename(filepath).replace(".nc", "")
#            suffix = {
#                'mean': 'mean',
#                'lowest': 'lowest',
#                'weather': 'weather_level'
#            }[level_type]
#            standalone_name = f"{basename}_first_mode_{suffix}.png"
#            standalone_path = os.path.join(first_mode_dir, standalone_name)
#            plt.savefig(standalone_path, dpi=300)
#            plt.close(fig1)
#            print(f"Bestie saved standalone mode 0 plot: {standalone_path}")
#
#    # === Colorbar ===
#    colorbar_label = {
#        'mean': "Wind Speed (mean over levels)",
#        'lowest': "Wind Speed (lowest level)",
#        'weather': "Wind Speed (weather level: nz=110)"
#    }[level_type]
#
#    cbar = fig.colorbar(pcms[0], ax=axes.ravel().tolist(), orientation="vertical",
#                        label=colorbar_label, shrink=0.75, pad=0.02)
#    cbar.ax.tick_params(labelsize=14)
#    cbar.set_label(colorbar_label, fontsize=16)
#
#    # === Save the full 3-mode figure ===
#    basename = os.path.basename(filepath).replace(".nc", "")
#    suffix = {
#        'mean': 'mean',
#        'lowest': 'lowest',
#        'weather': 'weather_level'
#    }[level_type]
#    outname = f"{basename}_coeff_plot_{suffix}_wspd.png"
#    plt.savefig(outname, dpi=300)
#    print(f"Bestie saved: {outname}")
