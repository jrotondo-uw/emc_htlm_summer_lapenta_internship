import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

# Paths
tlm_dir = '/work/noaa/da/jrotondo/JediCode/hybrid_tlm/fv3-jedi-htlm/tlm_output'
htlm_dir = '/work/noaa/da/jrotondo/JediCode/hybrid_tlm/fv3-jedi-htlm/htlm_output'
htlm_no_tlm_dir = '/work/noaa/da/jrotondo/JediCode/hybrid_tlm/fv3-jedi-htlm/htlm_output'
truth_file = '/work/noaa/da/jrotondo/HybridTlm/Training/C48/Data/diff_from_control/diff_mem002_cubed_sphere_grid_atmf009.nc'
output_dir = '/work/noaa/da/jrotondo/JediCode/hybrid_tlm/fv3-jedi-htlm/figs'

os.makedirs(output_dir, exist_ok=True)

# Load truth
ds_truth = xr.open_dataset(truth_file)

def load_ensemble_averages(varname, base_dir, prefix):
    members = []
    for i in range(2, 81):
        mem_id = f"mem{str(i).zfill(3)}"
        filename = f"{prefix}_forecast_cubed_sphere_{mem_id}_six_hours.nc"
        full_path = os.path.join(base_dir, filename)
        if not os.path.exists(full_path):
            print(f"WARNING: Missing {full_path}, skipping...")
            continue
        ds = xr.open_dataset(full_path)
        ds = ds.assign_coords(pfull=ds_truth['pfull'])
        members.append(ds[varname].astype(np.float64).isel(time=0).expand_dims(ensemble=[i]))
    return xr.concat(members, dim='ensemble')

def load_ensemble_averages_no_tlm(varname, base_dir, prefix):
    members = []
    for i in range(2, 81):
        mem_id = f"mem{str(i).zfill(3)}"
        filename = f"{prefix}_forecast_cubed_sphere_{mem_id}_six_hours_no_tlm.nc"
        full_path = os.path.join(base_dir, filename)
        if not os.path.exists(full_path):
            print(f"WARNING: Missing {full_path}, skipping...")
            continue
        ds = xr.open_dataset(full_path)
        ds = ds.assign_coords(pfull=ds_truth['pfull'])
        members.append(ds[varname].astype(np.float64).isel(time=0).expand_dims(ensemble=[i]))
    return xr.concat(members, dim='ensemble')

def load_ensemble_wind_speed(base_dir, prefix):
    members = []
    for i in range(2, 81):
        mem_id = f"mem{str(i).zfill(3)}"
        filename = f"{prefix}_forecast_cubed_sphere_{mem_id}_six_hours.nc"
        full_path = os.path.join(base_dir, filename)
        if not os.path.exists(full_path):
            print(f"WARNING: Missing {full_path}, skipping...")
            continue
        ds = xr.open_dataset(full_path)
        ds = ds.assign_coords(pfull=ds_truth['pfull'])
        ugrd = ds['ugrd'].astype(np.float64).isel(time=0)
        vgrd = ds['vgrd'].astype(np.float64).isel(time=0)
        wspd = np.sqrt(ugrd**2 + vgrd**2)
        members.append(wspd.expand_dims(ensemble=[i]))
    return xr.concat(members, dim='ensemble')

def load_ensemble_wind_speed_no_tlm(base_dir, prefix):
    members = []
    for i in range(2, 81):
        mem_id = f"mem{str(i).zfill(3)}"
        filename = f"{prefix}_forecast_cubed_sphere_{mem_id}_six_hours_no_tlm.nc"
        full_path = os.path.join(base_dir, filename)
        if not os.path.exists(full_path):
            print(f"WARNING: Missing {full_path}, skipping...")
            continue
        ds = xr.open_dataset(full_path)
        ds = ds.assign_coords(pfull=ds_truth['pfull'])
        ugrd = ds['ugrd'].astype(np.float64).isel(time=0)
        vgrd = ds['vgrd'].astype(np.float64).isel(time=0)
        wspd = np.sqrt(ugrd**2 + vgrd**2)
        members.append(wspd.expand_dims(ensemble=[i]))
    return xr.concat(members, dim='ensemble')

#def compute_rms_corr_ensemble(varname, label):
#    if varname == 'wspd':
#        ugrd = ds_truth['ugrd'].astype(np.float64).isel(time=0)
#        vgrd = ds_truth['vgrd'].astype(np.float64).isel(time=0)
#        v_truth = np.sqrt(ugrd**2 + vgrd**2)
#        v_tlm_ens = load_ensemble_wind_speed(tlm_dir, 'tlm')
#        v_htlm_ens = load_ensemble_wind_speed(htlm_dir, 'htlm')
#    else:
#        v_truth = ds_truth[varname].astype(np.float64).isel(time=0)
#        v_tlm_ens = load_ensemble_averages(varname, tlm_dir, 'tlm')
#        v_htlm_ens = load_ensemble_averages(varname, htlm_dir, 'htlm')  
#
#    assert v_tlm_ens.shape[1:] == v_truth.shape, f"Shape mismatch for {varname}"
#
#    v_tlm_ens, v_truth = xr.align(v_tlm_ens, v_truth)
#    v_htlm_ens, _ = xr.align(v_htlm_ens, v_truth)
#
#    v_tlm_mean = v_tlm_ens.mean(dim='ensemble')
#    v_tlm_std = v_tlm_ens.std(dim='ensemble')
#    v_htlm_mean = v_htlm_ens.mean(dim='ensemble')
#    v_htlm_std = v_htlm_ens.std(dim='ensemble')
#
#    rms_tlm = np.sqrt(((v_tlm_mean - v_truth) ** 2).mean(dim=('grid_yt', 'grid_xt')))
#    rms_htlm = np.sqrt(((v_htlm_mean - v_truth) ** 2).mean(dim=('grid_yt', 'grid_xt')))
#
#    spread_tlm = v_tlm_std.mean(dim=('grid_yt', 'grid_xt'))
#    spread_htlm = v_htlm_std.mean(dim=('grid_yt', 'grid_xt'))
#
#    corr_tlm = xr.corr(v_tlm_mean, v_truth, dim=('grid_yt', 'grid_xt'))
#    corr_htlm = xr.corr(v_htlm_mean, v_truth, dim=('grid_yt', 'grid_xt'))
#
#    rms_tlm_mean = rms_tlm.mean(dim='tile')
#    rms_htlm_mean = rms_htlm.mean(dim='tile')
#    spread_tlm_mean = spread_tlm.mean(dim='tile')
#    spread_htlm_mean = spread_htlm.mean(dim='tile')
#    corr_tlm_mean = corr_tlm.mean(dim='tile')
#    corr_htlm_mean = corr_htlm.mean(dim='tile')
#
#    levels = np.arange(1, len(rms_tlm['pfull']) + 1)
#    fig, ax1 = plt.subplots(figsize=(8, 10))
#
#    # TLM (red)
#    ax1.plot(rms_tlm_mean, levels, 'r-', label='RMS - TLM')
#    ax1.fill_betweenx(levels,
#                    rms_tlm_mean - spread_tlm_mean,
#                    rms_tlm_mean + spread_tlm_mean,
#                    color='red', alpha=0.2, label='Spread - TLM')
#
#    # HTLM (orange or magenta)
#    ax1.plot(rms_htlm_mean, levels, 'darkorange', linestyle='--', label='RMS - HTLM')
#    ax1.fill_betweenx(levels,
#                    rms_htlm_mean - spread_htlm_mean,
#                    rms_htlm_mean + spread_htlm_mean,
#                    color='orange', alpha=0.2, label='Spread - HTLM')
#
#    ax1.set_xlabel('RMS ± Spread', color='red')
#    ax1.tick_params(axis='x', colors='red')
#    ax1.set_ylabel('Vertical Level (1 = Top)')
#    ax1.invert_yaxis()
#    ax1.grid(True)
#
#    ax2 = ax1.twiny()
#    ax2.plot(corr_tlm_mean, levels, 'b-', label='Corr - TLM')
#    ax2.plot(corr_htlm_mean, levels, 'b--', label='Corr - HTLM')
#    ax2.set_xlabel('Correlation', color='blue')
#    ax2.tick_params(axis='x', colors='blue')
#    ax2.set_xlim(0, 1)
#
#    plt.title(f'Ensemble TLM vs HTLM vs Truth: {label}')
#    handles1, labels1 = ax1.get_legend_handles_labels()
#    handles2, labels2 = ax2.get_legend_handles_labels()
#    ax1.legend(handles1 + handles2, labels1 + labels2,
#            loc='upper right', fontsize='small', frameon=False)
#    plt.tight_layout()
#
#    # Save plot
#    plot_filename = os.path.join(output_dir, f"tlm_htlm_comparison_six_hour_{varname}.png")
#    plt.savefig(plot_filename, dpi=300)
#    plt.close()
#    print(f"Saved plot to: {plot_filename}")

def compute_rms_corr_ensemble(varname, label):
    if varname == 'wspd':
        ugrd = ds_truth['ugrd'].astype(np.float64).isel(time=0)
        vgrd = ds_truth['vgrd'].astype(np.float64).isel(time=0)
        v_truth = np.sqrt(ugrd**2 + vgrd**2)
        v_tlm_ens = load_ensemble_wind_speed(tlm_dir, 'tlm')
        v_htlm_ens = load_ensemble_wind_speed(htlm_dir, 'htlm')
        v_htlm_no_tlm_ens = load_ensemble_wind_speed_no_tlm(htlm_no_tlm_dir, 'htlm')
    else:
        v_truth = ds_truth[varname].astype(np.float64).isel(time=0)
        v_tlm_ens = load_ensemble_averages(varname, tlm_dir, 'tlm')
        v_htlm_ens = load_ensemble_averages(varname, htlm_dir, 'htlm')
        v_htlm_no_tlm_ens = load_ensemble_averages_no_tlm(varname, htlm_no_tlm_dir, 'htlm') 

    v_tlm_ens, v_truth = xr.align(v_tlm_ens, v_truth)
    v_htlm_ens, _ = xr.align(v_htlm_ens, v_truth)
    v_htlm_no_tlm_ens, _ = xr.align(v_htlm_no_tlm_ens, v_truth)

    v_tlm_mean = v_tlm_ens.mean(dim='ensemble')
    v_htlm_mean = v_htlm_ens.mean(dim='ensemble')
    v_htlm_no_tlm_mean = v_htlm_no_tlm_ens.mean(dim='ensemble')

    v_tlm_std = v_tlm_ens.std(dim='ensemble')
    v_htlm_std = v_htlm_ens.std(dim='ensemble')
    v_htlm_no_tlm_std = v_htlm_no_tlm_ens.std(dim='ensemble')

    rmse_tlm = np.sqrt(((v_tlm_mean - v_truth) ** 2).mean(dim=('grid_yt', 'grid_xt')))
    rmse_htlm = np.sqrt(((v_htlm_mean - v_truth) ** 2).mean(dim=('grid_yt', 'grid_xt')))
    rmse_htlm_no_tlm = np.sqrt(((v_htlm_no_tlm_mean - v_truth) ** 2).mean(dim=('grid_yt', 'grid_xt')))

    spread_tlm = v_tlm_std.mean(dim=('grid_yt', 'grid_xt'))
    spread_htlm = v_htlm_std.mean(dim=('grid_yt', 'grid_xt'))
    spread_htlm_no_tlm = v_htlm_no_tlm_std.mean(dim=('grid_yt', 'grid_xt'))

    corr_tlm = xr.corr(v_tlm_mean, v_truth, dim=('grid_yt', 'grid_xt'))
    corr_htlm = xr.corr(v_htlm_mean, v_truth, dim=('grid_yt', 'grid_xt'))

    rmse_tlm_mean = rmse_tlm.mean(dim='tile')
    rmse_htlm_mean = rmse_htlm.mean(dim='tile')
    spread_tlm_mean = spread_tlm.mean(dim='tile')
    spread_htlm_mean = spread_htlm.mean(dim='tile')
    corr_tlm_mean = corr_tlm.mean(dim='tile')
    corr_htlm_mean = corr_htlm.mean(dim='tile')

    levels = np.arange(1, len(rmse_tlm['pfull']) + 1)

    # --- Plot RMSE ± Spread ---
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.plot(rmse_tlm_mean, levels, 'r-', label='RMSE Mean - TLM')
    ax.fill_betweenx(levels, rmse_tlm_mean - spread_tlm_mean,
                     rmse_tlm_mean + spread_tlm_mean, color='red', alpha=0.2, label='Spread (±1σ) - TLM')
    
    ax.plot(rmse_htlm_mean, levels, 'darkorange', linestyle='--', label='RMSE Mean - HTLM')
    ax.fill_betweenx(levels, rmse_htlm_mean - spread_htlm_mean,
                     rmse_htlm_mean + spread_htlm_mean, color='orange', alpha=0.2, label='Spread (±1σ) - HTLM')

    ax.plot(rmse_htlm_no_tlm.mean(dim='tile'), levels, 'g-.', label='RMSE Mean - HTLM-no-TLM')
    ax.fill_betweenx(levels,
                     rmse_htlm_no_tlm.mean(dim='tile') - spread_htlm_no_tlm.mean(dim='tile'),
                     rmse_htlm_no_tlm.mean(dim='tile') + spread_htlm_no_tlm.mean(dim='tile'),
                     color='green', alpha=0.2, label='Spread (±1σ) - HTLM-no-TLM')

    ax.set_xlabel('RMSE ± Spread (±1σ)')
    ax.set_ylabel('Vertical Level (1 = Top)')
    ax.invert_yaxis()
    ax.grid(True)
    ax.legend(loc='upper right', fontsize='small', frameon=False)
    plt.title(f'RMSE: TLM vs HTLM – {label}')
    plt.tight_layout()
    rmse_path = os.path.join(output_dir, f"rmse_comparison_{varname}.png")
    plt.savefig(rmse_path, dpi=300)
    plt.close()
    print(f"Saved RMSE plot: {rmse_path}")

#    # --- Compute correlation spread across ensemble members ---
#    def member_corr(ens, truth):
#        return xr.concat([
#            xr.corr(ens.isel(ensemble=i), truth, dim=('grid_yt', 'grid_xt')).expand_dims(ensemble=[i])
#            for i in range(ens.sizes['ensemble'])
#        ], dim='ensemble')
#
#    corr_tlm_all = member_corr(v_tlm_ens, v_truth)
#    corr_htlm_all = member_corr(v_htlm_ens, v_truth)
#
#    corr_tlm_mean = corr_tlm_all.mean(dim='ensemble').mean(dim='tile')
#    corr_htlm_mean = corr_htlm_all.mean(dim='ensemble').mean(dim='tile')
#    corr_tlm_std = corr_tlm_all.std(dim='ensemble').mean(dim='tile')
#    corr_htlm_std = corr_htlm_all.std(dim='ensemble').mean(dim='tile')
#
#    # --- Plot Correlation ± Spread ---
#    fig, ax = plt.subplots(figsize=(8, 10))
#    ax.plot(corr_tlm_mean, levels, 'b-', label='Corr - TLM')
#    ax.fill_betweenx(levels,
#                     corr_tlm_mean - corr_tlm_std,
#                     corr_tlm_mean + corr_tlm_std,
#                     color='blue', alpha=0.2, label='Spread - TLM')
#
#    ax.plot(corr_htlm_mean, levels, 'b--', label='Corr - HTLM')
#    ax.fill_betweenx(levels,
#                     corr_htlm_mean - corr_htlm_std,
#                     corr_htlm_mean + corr_htlm_std,
#                     color='skyblue', alpha=0.2, label='Spread - HTLM')
#
#    ax.set_xlabel('Correlation ± Spread')
#    ax.set_ylabel('Vertical Level (1 = Top)')
#    ax.set_xlim(0, 1)
#    ax.invert_yaxis()
#    ax.grid(True)
#    ax.legend(loc='lower right', fontsize='small', frameon=False)
#    plt.title(f'Correlation: TLM vs HTLM – {label}')
#    plt.tight_layout()
#    corr_path = os.path.join(output_dir, f"correlation_comparison_{varname}.png")
#    plt.savefig(corr_path, dpi=300)
#    plt.close()
#    print(f"Saved Correlation plot: {corr_path}")

         # --- Compute mean correlation from ensemble mean ---
    corr_tlm_mean = xr.corr(v_tlm_mean, v_truth, dim=('grid_yt', 'grid_xt')).mean(dim='tile')
    corr_htlm_mean = xr.corr(v_htlm_mean, v_truth, dim=('grid_yt', 'grid_xt')).mean(dim='tile')

    # --- Compute correlation spread across ensemble members ---
    def member_corr(ens, truth):
        return xr.concat([
            xr.corr(ens.isel(ensemble=i), truth, dim=('grid_yt', 'grid_xt')).expand_dims(ensemble=[i])
            for i in range(ens.sizes['ensemble'])
        ], dim='ensemble')

    corr_tlm_all = member_corr(v_tlm_ens, v_truth)
    corr_htlm_all = member_corr(v_htlm_ens, v_truth)

    corr_tlm_std = corr_tlm_all.std(dim='ensemble').mean(dim='tile')
    corr_htlm_std = corr_htlm_all.std(dim='ensemble').mean(dim='tile')

    corr_htlm_no_tlm_mean = xr.corr(v_htlm_no_tlm_mean, v_truth, dim=('grid_yt', 'grid_xt')).mean(dim='tile')
    corr_htlm_no_tlm_all = member_corr(v_htlm_no_tlm_ens, v_truth)
    corr_htlm_no_tlm_std = corr_htlm_no_tlm_all.std(dim='ensemble').mean(dim='tile')

    # --- Plot Correlation ± Spread ---
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.plot(corr_tlm_mean, levels, 'b-', label='Corr Mean - TLM')
    ax.fill_betweenx(levels,
                     corr_tlm_mean - corr_tlm_std,
                     corr_tlm_mean + corr_tlm_std,
                     color='blue', alpha=0.2, label='Spread (±1σ) - TLM')

    ax.plot(corr_htlm_mean, levels, 'b--', label='Corr Mean - HTLM')
    ax.fill_betweenx(levels,
                     corr_htlm_mean - corr_htlm_std,
                     corr_htlm_mean + corr_htlm_std,
                     color='skyblue', alpha=0.2, label='Spread (±1σ) - HTLM')

    ax.plot(corr_htlm_no_tlm_mean, levels, 'g-.', label='Corr Mean - HTLM-no-TLM')
    ax.fill_betweenx(levels,
                     corr_htlm_no_tlm_mean - corr_htlm_no_tlm_std,
                     corr_htlm_no_tlm_mean + corr_htlm_no_tlm_std,
                     color='green', alpha=0.2, label='Spread (±1σ) - HTLM-no-TLM')

    ax.set_xlabel('Correlation ± Spread (±1σ)')
    ax.set_ylabel('Vertical Level (1 = Top)')
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.grid(True)
    ax.legend(loc='lower right', fontsize='small', frameon=False)
    plt.title(f'Correlation: TLM vs HTLM – {label}')
    plt.tight_layout()
    corr_path = os.path.join(output_dir, f"correlation_comparison_{varname}_mean_first.png")
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f"Saved Correlation plot: {corr_path}")

mpl.rcParams.update({
    'font.size': 16,           # Increase base font size
    'axes.titlesize': 18,      # Title font size
    'axes.labelsize': 16,      # Axis label size
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})

# Run for multiple variables
compute_rms_corr_ensemble('ugrd', 'U Wind')
compute_rms_corr_ensemble('vgrd', 'V Wind')
compute_rms_corr_ensemble('tmp', 'Temperature')
compute_rms_corr_ensemble('wspd', 'Wind Speed')
compute_rms_corr_ensemble('dpres', 'Pressure Difference')
compute_rms_corr_ensemble('spfh', 'Specific Humidity')