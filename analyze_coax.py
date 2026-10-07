"""
Per-station mixing metrics + REV2 vs REV3 comparison figures.
Own implementation, per HANDOFF_injector.md Sec 4 metric list.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyvista as pv

RHO_LOX = 1141.0
RHO_RP1 = 810.0
OF_DESIGN = 2.5
IDEAL_T = 0.36036036  # RP1 volumetric fraction at design O/F (Q_RP1/(Q_RP1+Q_LOX))

# local-T band corresponding to O/F within +/-25% of design
def T_for_OF(OF):
    return RHO_LOX / (RHO_LOX + OF * RHO_RP1)
T_LO = T_for_OF(OF_DESIGN * 1.25)   # OF=3.125 -> smaller T
T_HI = T_for_OF(OF_DESIGN * 0.75)   # OF=1.875 -> larger T

STATIONS_MM = [5, 10, 20, 40, 60, 90, 120, 150, 180]

def analyze_case(case_dir, tracer_time):
    foam_file = f'{case_dir}/case.foam'
    open(foam_file, 'a').close()
    reader = pv.OpenFOAMReader(foam_file)
    reader.set_active_time_value(float(tracer_time))
    mesh = reader.read()['internalMesh']
    print(f"  [{case_dir}] cells={mesh.n_cells} fields={mesh.array_names}")

    rows = []
    for y_mm in STATIONS_MM:
        y = y_mm / 1000.0
        sl = mesh.slice(normal=(0, 1, 0), origin=(0, y, 0))
        if sl.n_cells == 0:
            rows.append(dict(y_mm=y_mm, empty=True))
            continue
        sl = sl.compute_cell_sizes(length=False, area=True, volume=False)
        T = np.asarray(sl['T']).ravel()
        U = np.asarray(sl['U'])
        Uy = U[:, 1]
        A = np.asarray(sl['Area'])

        # --- net flux-weighted RP1 fraction (mass/volume conservation check) ---
        flux = Uy * A                      # signed volumetric flux per cell
        fwd = np.clip(flux, 0, None)       # forward-flow only, for flux-weighted stats
        total_fwd = fwd.sum()
        flux_weighted_T = (T * fwd).sum() / total_fwd if total_fwd > 1e-15 else np.nan
        net_flux_weighted_T = (T * flux).sum() / flux.sum() if abs(flux.sum()) > 1e-15 else np.nan

        # --- segregation intensity: area-weighted and forward-flux-weighted ---
        w_area = A / A.sum()
        mean_area_T = np.sum(w_area * T)
        var_area_T = np.sum(w_area * (T - mean_area_T) ** 2)
        Is_area = var_area_T / (mean_area_T * (1 - mean_area_T)) if 0 < mean_area_T < 1 else np.nan

        if total_fwd > 1e-15:
            w_flux = fwd / total_fwd
            mean_flux_T = np.sum(w_flux * T)
            var_flux_T = np.sum(w_flux * (T - mean_flux_T) ** 2)
            Is_flux = var_flux_T / (mean_flux_T * (1 - mean_flux_T)) if 0 < mean_flux_T < 1 else np.nan
        else:
            Is_flux = np.nan

        # --- Rupe mixing efficiency (cell-level adaptation) ---
        rp1_flux_i = fwd * T
        lox_flux_i = fwd * (1 - T)
        tot_rp1 = rp1_flux_i.sum()
        tot_lox = lox_flux_i.sum()
        if tot_rp1 > 1e-15 and tot_lox > 1e-15:
            Em = 1.0 - 0.5 * np.sum(np.abs(rp1_flux_i / tot_rp1 - lox_flux_i / tot_lox))
        else:
            Em = np.nan

        # --- fraction of forward mass flux within +/-25% of design O/F ---
        in_band = (T >= T_LO) & (T <= T_HI)
        frac_in_band = fwd[in_band].sum() / total_fwd if total_fwd > 1e-15 else np.nan

        # --- recirculation area fraction (reverse axial flow) ---
        recirc_frac = A[Uy < 0].sum() / A.sum()

        rows.append(dict(
            y_mm=y_mm, empty=False,
            flux_weighted_T=flux_weighted_T, net_flux_weighted_T=net_flux_weighted_T,
            Is_area=Is_area, Is_flux=Is_flux, Em=Em,
            frac_in_band=frac_in_band, recirc_frac=recirc_frac,
        ))
        print(f"    y={y_mm:4d}mm  T_flux={flux_weighted_T:.4f}  Is_area={Is_area:.4f}  "
              f"Is_flux={Is_flux:.4f}  Em={Em:.4f}  frac_OF_band={frac_in_band:.4f}  "
              f"recirc={recirc_frac:.4f}")

    return rows, mesh


def main():
    out_dir = sys.argv[1]
    case_specs = {}
    for arg in sys.argv[2:]:
        label, path = arg.split('=', 1)
        case_specs[label] = path
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    meshes = {}
    for label, spec in case_specs.items():
        case_dir, tracer_time = spec.split('@')
        print(f"=== {label}: {case_dir} @ t={tracer_time} ===")
        rows, mesh = analyze_case(case_dir, tracer_time)
        results[label] = rows
        meshes[label] = mesh

    # ---- comparison figure: metrics vs distance ----
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))
    colors = {'REV2': 'tab:red', 'REV3': 'tab:blue'}
    for label, rows in results.items():
        ys = [r['y_mm'] for r in rows if not r['empty']]
        c = colors.get(label, None)
        axes[0,0].plot(ys, [r['flux_weighted_T'] for r in rows if not r['empty']], 'o-', label=label, color=c)
        axes[0,1].plot(ys, [r['Is_flux'] for r in rows if not r['empty']], 'o-', label=label, color=c)
        axes[0,2].plot(ys, [r['Em'] for r in rows if not r['empty']], 'o-', label=label, color=c)
        axes[1,0].plot(ys, [r['frac_in_band'] for r in rows if not r['empty']], 'o-', label=label, color=c)
        axes[1,1].plot(ys, [r['recirc_frac'] for r in rows if not r['empty']], 'o-', label=label, color=c)
        axes[1,2].plot(ys, [r['Is_area'] for r in rows if not r['empty']], 'o-', label=label, color=c)

    axes[0,0].axhline(IDEAL_T, color='gray', ls='--', lw=1)
    axes[0,0].set_title('Flux-weighted RP-1 fraction (ideal=%.3f)' % IDEAL_T)
    axes[0,1].set_title('Segregation intensity (flux-weighted, 0=mixed)')
    axes[0,1].set_ylim(0, 1.05)
    axes[0,2].set_title('Rupe mixing efficiency Em (1=perfect)')
    axes[0,2].set_ylim(0, 1.05)
    axes[1,0].set_title('Fraction of flux within +/-25% of design O/F')
    axes[1,0].set_ylim(0, 1.05)
    axes[1,1].set_title('Recirculation area fraction')
    axes[1,2].set_title('Segregation intensity (area-weighted)')
    axes[1,2].set_ylim(0, 1.05)
    for ax in axes.flat:
        ax.set_xlabel('distance downstream of injector face (mm)')
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    fig_path = f'{out_dir}/comparison_metrics.png'
    plt.savefig(fig_path, dpi=130)
    print("saved", fig_path)

    # ---- contour figures at 10/40/90mm for each case ----
    for label, mesh in meshes.items():
        fig2, axs = plt.subplots(1, 3, figsize=(15, 5))
        for ax, y_mm in zip(axs, [10, 40, 90]):
            sl = mesh.slice(normal=(0, 1, 0), origin=(0, y_mm/1000.0, 0))
            if sl.n_cells == 0:
                continue
            sl = sl.cell_data_to_point_data()
            pts = sl.points
            T = np.asarray(sl['T']).ravel() if 'T' in sl.array_names else np.zeros(sl.n_points)
            sc = ax.tricontourf(pts[:,0]*1000, pts[:,2]*1000, T, levels=20, cmap='coolwarm', vmin=0, vmax=1)
            ax.set_aspect('equal')
            ax.set_title(f'{label} y={y_mm}mm')
        plt.colorbar(sc, ax=axs, shrink=0.8, label='RP-1 fraction T')
        plt.savefig(f'{out_dir}/{label}_cross_sections.png', dpi=120)
        print(f"saved {out_dir}/{label}_cross_sections.png")

    # ---- longitudinal slice (z=0) for each case ----
    for label, mesh in meshes.items():
        sl = mesh.slice(normal=(0, 0, 1), origin=(0, 0, 0))
        if sl.n_cells == 0:
            continue
        plotter = pv.Plotter(off_screen=True, window_size=(1400, 700))
        plotter.add_mesh(sl, scalars='T', cmap='coolwarm', clim=[0, 1], show_edges=False)
        plotter.view_yx()
        plotter.add_title(f'{label} RP-1 fraction, longitudinal (z=0)')
        plotter.screenshot(f'{out_dir}/{label}_longitudinal.png')
        print(f"saved {out_dir}/{label}_longitudinal.png")

    # ---- summary table ----
    with open(f'{out_dir}/summary.txt', 'w') as f:
        for label, rows in results.items():
            f.write(f"=== {label} ===\n")
            for r in rows:
                if r['empty']:
                    continue
                f.write(f"y={r['y_mm']:4d}mm  T_flux={r['flux_weighted_T']:.4f}  "
                        f"Is_flux={r['Is_flux']:.4f}  Em={r['Em']:.4f}  "
                        f"frac_OF_band={r['frac_in_band']:.4f}  recirc={r['recirc_frac']:.4f}\n")
    print("wrote summary.txt")


if __name__ == '__main__':
    main()
