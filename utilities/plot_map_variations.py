#!/usr/bin/env python
"""Correlation functions of the AbacusAurora "varMAP" halo-finder parameter
variations, compared against the fiducial MAP parameters, plus a cache of the
raw correlation functions for later finite-difference derivative estimation.

Standalone script version of the commented-out draft in cell 10 of
notebooks/analysis.ipynb. Every AbacusAurora_varMAP_c000_ph350_m0* directory shares
the same fiducial cosmology (H0=67.36, omega_b=0.02237, omega_cdm=0.12,
n_s=0.9649, w0=-1, wa=0, Omega_K=0 -- this is cosmology c000, identical to the
"emulatorslim0p5" c112 fiducial used by plot_cosmo_variations.py); each
m-index instead perturbs the three MAP halo-finder parameters (Map_Aaccrete,
Map_Amerge, Map_bmerge_proper) away from their fiducial values (confirmed by
grepping abacus.par across every AbacusAurora_varMAP_c000_ph350_m0* dir).
There is no plain fiducial-MAP run at c000/ph350 in the varMAP set itself, so
the fiducial comparison curve comes from
AbacusAurora_emulatorslim0p5_c112_ph311, which has the identical cosmology and
identical fiducial Map_Aaccrete/Map_Amerge/Map_bmerge_proper values, just a
different phase (IC realization).

Two families of runs are read:
  - single-parameter sweeps (m041-m052): one parameter varied at a time, 4
    step sizes each, used for first/second-derivative (diagonal Hessian)
    estimates.
  - two-parameter cross grids (m060-m071): 2x2 grids in each pair of
    parameters (Amerge x bmerge_proper, Aaccrete x Amerge, Aaccrete x
    bmerge_proper), used for cross-derivative (off-diagonal Hessian) finite
    differences.

Rather than reuse ``utilities.compare_correlation_functions`` (which
recomputes each maplog's correlation function internally and returns only a
Figure), this script calls ``utilities.get_map_xi`` directly once per
(run, redshift), keeps the raw (r, xi, poisson) triples, and:
  1. saves them to ``analysis_data/map_variations_xi_n<abundance>.npz`` +
     ``.json`` (metadata: swept parameter values, labels, redshifts) so a
     later derivative-estimation script -- or a re-plot of this same script --
     can load them straight off disk without re-reading maplogs or re-running
     pair counts. If a cache already on disk covers every requested run and
     redshift, it's loaded instead of recomputing (pass --recompute to force
     a fresh computation regardless);
  2. plots them itself (one row per parameter/pair, one column per redshift,
     gray dashed fiducial reference) in the same visual style as
     plot_cosmo_variations.py, saving two figures:
     ``Images/compare_correlation_functions_n<abundance>_map_variations.png``
     (raw r^2*xi) and
     ``Images/compare_correlation_functions_n<abundance>_map_variations_ratio.png``
     (each curve divided by the fiducial at that redshift).

z=0.2 is not requested by default: FIDUCIAL_DIR has FinalRedshift=0.5, so it
has no timeslice near z=0.2 (its nearest step is the same one used for
z=0.5), and a "z=0.2" fiducial curve would silently just be a duplicate of the
z=0.5 one. Pass --redshifts explicitly (e.g. to include 0.2) if a downstream
use of the cache needs it. z=4 (InitialRedshift=99 for both FIDUCIAL_DIR and
the varMAP runs) is comfortably in range and is requested by default.

Run directly, or via the accompanying ``run_map_variations.pbs``::

    python ~/map_analysis/utilities/plot_map_variations.py --nworkers 8 --nthreads 26
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

from utilities import read_maplog_dir, get_map_xi, map_step_for_redshift, _xi_plot_values
from abacus.param import InputFile

AURORA_BASE = Path("/flare/Abacus/AbacusAurora")
FIDUCIAL_DIR = AURORA_BASE / "AbacusAurora_emulatorslim0p5_c112_ph311"
CACHE_DIR = Path("analysis_data")

MAP_PARAMS = ('Map_Aaccrete', 'Map_Amerge', 'Map_bmerge_proper')

# row label -> (m-indices in plotting order, [swept params in that row])
# Single-parameter rows are ordered low -> high value (checked against
# abacus.par); cross-term rows are just the four corners of that pair's 2x2
# grid, in whatever order they were run (labels are built from abacus.par
# values, not assumed order, so this doesn't matter for correctness).
RUN_GROUPS = {
    'Map_Amerge':                       (['m041', 'm042', 'm043', 'm044'], ['Map_Amerge']),
    'Map_bmerge_proper':                (['m045', 'm046', 'm047', 'm048'], ['Map_bmerge_proper']),
    'Map_Aaccrete':                     (['m049', 'm050', 'm051', 'm052'], ['Map_Aaccrete']),
    'Map_Amerge x Map_bmerge_proper':   (['m060', 'm061', 'm062', 'm063'], ['Map_Amerge', 'Map_bmerge_proper']),
    'Map_Aaccrete x Map_Amerge':        (['m064', 'm065', 'm066', 'm067'], ['Map_Aaccrete', 'Map_Amerge']),
    'Map_Aaccrete x Map_bmerge_proper': (['m068', 'm069', 'm070', 'm071'], ['Map_Aaccrete', 'Map_bmerge_proper']),
}


def map_dir(m, cosmo='c000', phase=350):
    d = AURORA_BASE / f"AbacusAurora_varMAP_{cosmo}_ph{phase}_{m}"
    if not d.is_dir():
        raise FileNotFoundError(f"no {d} directory found")
    return d


def read_par(mdir):
    return InputFile(fn=str(Path(mdir) / "abacus.par"))


def build_runs():
    """Resolve every m-index in RUN_GROUPS to its directory, MAP parameter
    values, and a legend label (value + % delta from fiducial for each param
    swept in its row). Returns (runs, fiducial_par):
      runs[m_index] = {'dir', 'values' (all 3 Map_* params), 'label'}
    """
    fiducial_par = read_par(FIDUCIAL_DIR)
    runs = {}
    for row_name, (ms, swept_params) in RUN_GROUPS.items():
        for m in ms:
            if m in runs:
                continue
            d = map_dir(m)
            par = read_par(d)
            values = {p: par[p] for p in MAP_PARAMS}
            parts = []
            for p in swept_params:
                fid_val = fiducial_par[p]
                val = values[p]
                pct = 100.0 * (val - fid_val) / fid_val
                parts.append(f'{p}={val:.5g} ({pct:+.1f}%)')
            runs[m] = {'dir': d, 'values': values, 'label': ', '.join(parts)}

    print('--- resolved runs ---')
    for row_name, (ms, _) in RUN_GROUPS.items():
        print(f'{row_name}:')
        for m in ms:
            print(f'  {runs[m]["label"]:40s} {runs[m]["dir"].name}')

    return runs, fiducial_par


def compute_all_xi(runs, redshifts, r_edges, nworkers, nthreads, abundance, max_p):
    """get_map_xi for the fiducial + every run in ``runs``, one entry per
    (run, redshift). Reads each unique maplog directory exactly once.

    Returns {'fiducial': result, <m_index>: result, ...} where
    result = {'xi': (nz, nbins), 'poisson': (nz, nbins), 'z_actual': (nz,)}.
    """
    maplog_cache = {}

    def get_maplog(d):
        key = str(d)
        if key not in maplog_cache:
            print(f'Reading maplog: {d}')
            maplog_cache[key] = read_maplog_dir(key, nworkers=nworkers)
        return maplog_cache[key]

    def compute_for(d):
        maplog = get_maplog(d)
        xi_list, poisson_list, z_actual = [], [], []
        for z_req in redshifts:
            ts, zm = map_step_for_redshift(maplog, z_req)
            xi, poisson = get_map_xi(maplog, ts, r_edges, abundance=abundance,
                                     max_p=max_p, nthreads=nthreads)
            xi_list.append(xi)
            poisson_list.append(poisson)
            z_actual.append(zm)
        return {'xi': np.array(xi_list), 'poisson': np.array(poisson_list),
                'z_actual': np.array(z_actual)}

    results = {'fiducial': compute_for(FIDUCIAL_DIR)}
    for m, info in runs.items():
        results[m] = compute_for(info['dir'])
    return results


def save_cache(path_prefix, r, redshifts, runs, results, fiducial_par):
    """Save raw (r, xi, poisson) for every run + metadata, retrievable for
    later finite-difference derivative estimation without re-running any
    pair counts. Numeric arrays go in <path_prefix>.npz (keyed by
    '<run_id>_xi' / '<run_id>_poisson' / '<run_id>_z'); everything else
    (labels, swept parameter values, row grouping) goes in
    <path_prefix>.json.
    """
    npz_path = path_prefix.with_suffix('.npz')
    json_path = path_prefix.with_suffix('.json')

    arrays = {'r': r, 'redshifts_requested': np.asarray(redshifts, dtype=float)}
    for run_id, res in results.items():
        arrays[f'{run_id}_xi'] = res['xi']
        arrays[f'{run_id}_poisson'] = res['poisson']
        arrays[f'{run_id}_z'] = res['z_actual']
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **arrays)

    meta = {
        'fiducial_dir': str(FIDUCIAL_DIR),
        'fiducial_values': {p: fiducial_par[p] for p in MAP_PARAMS},
        'rows': {row_name: {'run_ids': ms, 'swept_params': swept}
                 for row_name, (ms, swept) in RUN_GROUPS.items()},
        'runs': {run_id: {'dir': str(info['dir']), 'values': info['values'],
                           'label': info['label']}
                 for run_id, info in runs.items()},
    }
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'Saved correlation function cache -> {npz_path}, {json_path}')


def load_cache(path_prefix, run_ids, redshifts):
    """Try to satisfy (run_ids, redshifts) from a cache previously written by
    ``save_cache``. Returns (r, results) -- same shape compute_all_xi would
    return, with columns picked out to match ``redshifts`` -- or None if the
    cache is missing, or doesn't cover every requested run/redshift (in which
    case the caller should fall back to recomputing from the maplogs).
    """
    npz_path = path_prefix.with_suffix('.npz')
    if not npz_path.exists():
        return None
    data = np.load(npz_path)

    cached_z = data['redshifts_requested']
    col_idx = []
    for z in redshifts:
        matches = np.where(np.isclose(cached_z, z))[0]
        if len(matches) == 0:
            print(f'Cache {npz_path} has no z={z} column; recomputing from maplogs.')
            return None
        col_idx.append(matches[0])

    results = {}
    for run_id in ['fiducial'] + list(run_ids):
        if f'{run_id}_xi' not in data:
            print(f'Cache {npz_path} has no entry for run "{run_id}"; recomputing from maplogs.')
            return None
        results[run_id] = {'xi': data[f'{run_id}_xi'][col_idx],
                            'poisson': data[f'{run_id}_poisson'][col_idx],
                            'z_actual': data[f'{run_id}_z'][col_idx]}

    print(f'Loaded correlation functions from cache -> {npz_path}')
    return data['r'], results


def plot_grid(runs, results, redshifts, cmaps, clo, chi, abundance,
              save, str_identifier, plot_ratio=False, panel_size=(4, 3)):
    r = results['_r']
    row_names = list(RUN_GROUPS.keys())
    nrows, ncols = len(row_names), len(redshifts)

    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey='row', squeeze=False,
                             figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
    fig.subplots_adjust(wspace=0, hspace=0)
    data_ymax = np.zeros(nrows)
    fid = results['fiducial']

    for row, row_name in enumerate(row_names):
        run_ids, _ = RUN_GROUPS[row_name]
        cmap = colormaps[cmaps[row % len(cmaps)]]
        fracs = clo + (chi - clo) * np.linspace(0, 1, max(len(run_ids), 1))
        for col, z_req in enumerate(redshifts):
            ax = axes[row][col]
            fid_xi_col, fid_poisson_col = fid['xi'][col], fid['poisson'][col]
            zd = fid['z_actual'][col]
            lbl_d = f'Fiducial (z={zd:.3f})' if col == 0 else f'z={zd:.3f}'

            if plot_ratio:
                # fiducial / fiducial == 1 exactly, with no uncertainty --
                # draw it as a flat reference line rather than through
                # _xi_plot_values.
                ax.axhline(1.0, ls='--', lw=2, color='gray', label=lbl_d)
                ax.axhline(0.9, ls=':', lw=2.5, color='gray', alpha=0.8)
                ax.axhline(1.1, ls=':', lw=2.5, color='gray', alpha=0.8)
            else:
                y_d, yerr_d = _xi_plot_values(r, fid_xi_col, fid_poisson_col)
                ax.plot(r, y_d, '--', lw=2, color='gray', label=lbl_d)
                ax.fill_between(r, y_d - yerr_d, y_d + yerr_d, color='gray', alpha=0.3)
                data_ymax[row] = max(data_ymax[row], np.nanmax(y_d + yerr_d))

            for i, run_id in enumerate(run_ids):
                res = results[run_id]
                ref = (fid_xi_col, fid_poisson_col) if plot_ratio else None
                y, yerr = _xi_plot_values(r, res['xi'][col], res['poisson'][col], plot_ratio, ref)
                z_actual = res['z_actual'][col]
                base = runs[run_id]['label']
                lbl = f'{base} (z={z_actual:.3f})' if col == 0 else f'z={z_actual:.3f}'
                color = cmap(fracs[i])
                ax.plot(r, y, '-', lw=2, color=color, label=lbl)
                ax.fill_between(r, y - yerr, y + yerr, color=color, alpha=0.3)
                if not plot_ratio:
                    data_ymax[row] = max(data_ymax[row], np.nanmax(y + yerr))

            ax.set(xscale='log')
            ax.grid()
            ax.legend(fontsize='x-small')
            if row == 0:
                ax.set_title(f'z = {z_req:.2f}')

    for row in range(nrows):
        if plot_ratio:
            axes[row, 0].set_ylim(0.7, 1.3)
            axes[row, 0].set_ylabel(row_names[row] + '\n' + r'$\xi_{gg}/\xi_{gg,\rm ref}$', fontsize='small')
        else:
            axes[row, 0].set_ylim(0, data_ymax[row] * 1.05)
            axes[row, 0].set_ylabel(row_names[row], fontsize='small')
    for ax in axes[-1, :]:
        ax.set_xlabel(r'r [$h^{-1}Mpc$]')

    if save:
        # matches utilities.compare_correlation_functions: 'Images/' is the
        # repo-root directory (a symlink to /flare/Abacus/$USER/saves/),
        # reached via '../Images' since this script runs from map_analysis/utilities/
        outdir = Path('../Images')
        outdir.mkdir(exist_ok=True)
        fname = outdir / f'compare_correlation_functions_n{abundance:.2e}_{str_identifier}.png'
        fig.savefig(fname, bbox_inches='tight')
        print(f'Saved figure -> {fname}')

    return fig, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--nworkers', type=int, default=8,
                         help='parallel workers for read_maplog_dir (ASDF decompression)')
    parser.add_argument('--nthreads', type=int, default=26,
                         help='threads for get_xi pair counting')
    parser.add_argument('--abundance', type=float, default=5.01e-3,
                         help='target number density (Mpc/h)^-3 for the MAP threshold')
    parser.add_argument('--rmin', type=float, default=2e-2, help='min separation (Mpc/h)')
    parser.add_argument('--rmax', type=float, default=40.0, help='max separation (Mpc/h)')
    parser.add_argument('--nbins', type=int, default=30, help='number of log-spaced separation bins')
    parser.add_argument('--max-p', type=float, default=1e7, dest='max_p',
                         help='max points passed through to get_xi')
    parser.add_argument('--redshifts', type=float, nargs='+', default=[4.0, 2.0, 1.0, 0.5],
                         help='FIDUCIAL_DIR has no timeslice near z=0.2, so it is '
                              'excluded by default (see module docstring)')
    parser.add_argument('--recompute', action='store_true',
                         help='ignore any matching on-disk cache and re-read the '
                              'maplogs / re-run pair counts from scratch')
    args = parser.parse_args()

    r_edges = np.logspace(np.log10(args.rmin), np.log10(args.rmax), args.nbins + 1)

    runs, fiducial_par = build_runs()
    cache_path = CACHE_DIR / f'map_variations_xi_n{args.abundance:.2e}'

    cached = None if args.recompute else load_cache(cache_path, runs.keys(), args.redshifts)
    if cached is not None:
        r, results = cached
    else:
        r = r_edges[:-1] + np.diff(r_edges) / 2.0
        results = compute_all_xi(runs, args.redshifts, r_edges,
                                 nworkers=args.nworkers, nthreads=args.nthreads,
                                 abundance=args.abundance, max_p=args.max_p)
        save_cache(cache_path, r, args.redshifts, runs, results, fiducial_par)

    results['_r'] = r  # convenience for plot_grid; not written to the cache
    plot_kwargs = dict(cmaps=('Blues', 'Greens', 'Purples', 'Oranges', 'Reds', 'Greys'),
                        clo=0.4, chi=1.0, abundance=args.abundance, save=True)
    plot_grid(runs, results, args.redshifts, str_identifier='map_variations',
             plot_ratio=False, **plot_kwargs)
    plot_grid(runs, results, args.redshifts, str_identifier='map_variations_ratio',
             plot_ratio=True, **plot_kwargs)


if __name__ == '__main__':
    main()
