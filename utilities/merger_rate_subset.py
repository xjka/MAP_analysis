#!/usr/bin/env python
"""Merger rates of final-timeslice MAPs, from a random subset of maplog files.

Designed for simulations too large to read in full (e.g. AbacusAurora_demiflag_c200_ph410:
~2170 maplog files of ~4.6 GB each). A random subset of ``--nfiles`` files is
drawn (same draw on every rank, ``--seed``), dealt round-robin to MPI ranks
(run ONE rank per node), and within each rank read by a process pool of
``--nworkers`` workers (one file per task -- a file must be decompressed in full
to read any column, and each file holds whole merger trees, so files are
independent).

For every final-timeslice MAP with multiplicity >= ``--mult_min`` the merger
rate is estimated as a Poisson process from its primary-branch mergers (the run
of nodes from its head to the first Formation node; see
``utilities.min_merger_ratio_primary``): with N mergers at cosmic times
t_1 <= ... <= t_N there are N-1 separations whose sum is t_N - t_1, so

    rate = (N - 1) / (t_N - t_1)        [1/Gyr]

MAPs with N < 2, or whose mergers all fall in one timestep (t_N == t_1), have no
rate (NaN) and are left out of the rate histogram and scatter, but still count
in the multiplicity histogram. Cosmic time comes from the header cosmology
(same integral as ``utilities._cosmic_time_curve``).

A periodic reference run (``--reference``, default AbacusAurora_emulatormax_c200_ph310: same
cosmology, MAP parameters, softening/spacing ratio and particle mass to 0.07%) is sampled the
same way (``--ref_nfiles`` files, default ``--nfiles``) and overlaid, so the same ``--mult_min``
selects comparable MAPs. Because the two samples differ in size, the histograms are normalised
to unit area.

Output: results (rank 0) in ``--out_dir`` as one ``merger_rate_subset_<sim>.npz`` per run (mult,
n_merger, rate, file_index per MAP; written as soon as each run is read; ``--replot`` redraws the
figure from them without reading), figure in ``--image_dir``
(default ``<repo>/Images``) as ``merger_rate_subset_<sim>.png``: multiplicity histogram, rate
histogram, and rate vs multiplicity (2D histogram of the main run, with the median and 16-84%
range in multiplicity bins for both runs).

    mpiexec -n <nnodes> --ppn 1 python ~/MAP_analysis/utilities/merger_rate_subset.py SIMDIR --nfiles 64
"""
import argparse
import random
import time
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import asdf
from abacusnbody.data.read_abacus import read_asdf

from utilities import _cosmic_time_curve, _history_from_stream


def _merger_rate_one_file(f, mult_min):
    """Read one maplog file; return per-MAP (mult, n_merger, rate) for MAPs with mult >= mult_min."""
    asdf.get_config().validate_on_read = False
    tbl = read_asdf(f, load=['mult', 'control', 'node_type', 'timestep'])
    nt = np.asarray(tbl['node_type'])
    ts = np.asarray(tbl['timestep']).astype(np.int64)
    mult = np.asarray(tbl['mult']).astype(np.int64)
    meta = tbl.meta
    cosmo = {k: float(meta[k]) for k in ('H0', 'Omega_M', 'Omega_K', 'Omega_DE', 'w0', 'wa')}
    step_to_z = np.asarray(meta['Step_to_Z'])
    zg, tg = _cosmic_time_curve(cosmo)
    t_step = np.interp(step_to_z, zg, tg)            # cosmic time [Gyr] of each step (t(z=99) = 0 origin)

    def extractor(h):
        nt_, ts_ = h['nt'], h['ts']
        is_m = nt_ == 1
        t_node = t_step[ts_ - 1]                      # node timestep is one ahead of its redshift
        # +1 pad so a run ending at the last node is valid (see min_merger_ratio_primary)
        tmin = np.full(nt_.shape[0] + 1, np.inf)
        tmax = np.full(nt_.shape[0] + 1, -np.inf)
        cnt = np.zeros(nt_.shape[0] + 1)
        tmin[:-1][is_m] = t_node[is_m]
        tmax[:-1][is_m] = t_node[is_m]
        cnt[:-1][is_m] = 1.0
        n = h['reduce_dominant_run'](np.add, cnt)
        t0 = h['reduce_dominant_run'](np.minimum, tmin)
        t1 = h['reduce_dominant_run'](np.maximum, tmax)
        span = t1 - t0
        with np.errstate(invalid='ignore', divide='ignore'):
            rate = np.where((n >= 2) & (span > 0), (n - 1) / span, np.nan)
        return {'n_merger': n, 'rate': rate}

    # mult_sec and pos are not needed here: pass zero-memory broadcast stand-ins
    N = nt.shape[0]
    zeros = np.zeros(N, dtype=np.int64)
    pos = np.broadcast_to(np.zeros(3, dtype=np.float32), (N, 3))
    r = _history_from_stream(nt, ts, mult, zeros, pos, step_to_z, extractor)
    keep = r['mult'] >= mult_min
    return {'mult': r['mult'][keep], 'n_merger': r['values']['n_merger'][keep].astype(np.int32),
            'rate': r['values']['rate'][keep], 'step': r['step'], 'redshift': r['redshift']}


_REF_DEFAULT = '/flare/Abacus/AbacusAurora/AbacusAurora_emulatormax_c200_ph310'


def _binned_percentiles(mult, rate, mb, nmin=10):
    """Median and 16/84 percentiles of ``rate`` in the multiplicity bins ``mb`` (NaN where < nmin MAPs)."""
    out = np.full((3, len(mb) - 1), np.nan)
    idx = np.digitize(mult, mb) - 1
    for i in range(len(mb) - 1):
        v = rate[(idx == i) & np.isfinite(rate)]
        if len(v) >= nmin:
            out[:, i] = np.percentile(v, [16, 50, 84])
    return out


def _short(label):
    return label.replace('AbacusAurora_', '')


def _plot(runs, mult_min, path):
    """``runs``: list of dicts (label, mult, rate, nfiles, redshift); the first is the main run.

    Top row: multiplicity and rate distributions, and rate vs multiplicity. Bottom row: the same curves as a ratio of
    the main run to each other run (histogram bins with < 10 MAPs in either run are left blank).
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), constrained_layout=True,
                             gridspec_kw={'height_ratios': [3, 1.6]})
    allm = np.concatenate([r['mult'] for r in runs])
    allr = np.concatenate([r['rate'][np.isfinite(r['rate'])] for r in runs])
    mb = np.logspace(np.log10(max(mult_min, 1)), np.log10(allm.max()), 50)
    rb = np.logspace(np.log10(allr.min()), np.log10(allr.max()), 50)
    mid = np.sqrt(mb[:-1] * mb[1:])
    rmid = np.sqrt(rb[:-1] * rb[1:])
    colors = ['C0', 'C3', 'C2', 'C1']
    hist_m, hist_r, cnt_m, cnt_r, quant, lab_run = [], [], [], [], [], []
    for run, c in zip(runs, colors):
        ok = np.isfinite(run['rate'])
        lab = f'{_short(run["label"])}\n{len(run["mult"]) / 1e6:.1f}M MAPs, {run["nfiles"]} files'
        lab_run.append(lab)
        axes[0, 0].hist(run['mult'], bins=mb, density=True, histtype='step', lw=1.6, color=c, label=lab)
        axes[0, 1].hist(run['rate'][ok], bins=rb, density=True, histtype='step', lw=1.6, color=c)
        hist_m.append(np.histogram(run['mult'], bins=mb, density=True)[0])
        hist_r.append(np.histogram(run['rate'][ok], bins=rb, density=True)[0])
        cnt_m.append(np.histogram(run['mult'], bins=mb)[0])
        cnt_r.append(np.histogram(run['rate'][ok], bins=rb)[0])
        quant.append(_binned_percentiles(run['mult'], run['rate'], mb))
    axes[0, 0].set(xscale='log', yscale='log', ylabel='normalised N')
    axes[0, 0].legend(fontsize=7, loc='upper right')
    axes[0, 1].set(xscale='log', yscale='log', ylabel='normalised N')

    main = runs[0]
    ok = np.isfinite(main['rate'])
    h = axes[0, 2].hist2d(main['mult'][ok], main['rate'][ok], bins=[mb, rb], norm=matplotlib.colors.LogNorm(),
                          cmap='Greys')
    fig.colorbar(h[3], ax=axes[0, 2], label=f'N MAPs ({_short(main["label"])})')
    for run, q, c in zip(runs, quant, colors):
        axes[0, 2].plot(mid, q[1], color=c, lw=2, label=_short(run['label']))
        axes[0, 2].fill_between(mid, q[0], q[2], color=c, alpha=0.2)
    axes[0, 2].set(xscale='log', yscale='log', ylabel=r'merger rate [Gyr$^{-1}$]')
    axes[0, 2].legend(fontsize=7, loc='upper left', title='median, 16-84%', title_fontsize=7)

    # bottom row: main / each other run, on the same bins as the curves above
    nmin = 10
    for k in range(1, len(runs)):
        c = colors[k]
        lab = f'{_short(main["label"])} / {_short(runs[k]["label"])}'
        with np.errstate(invalid='ignore', divide='ignore'):
            for ax, x, hs, cn in ((axes[1, 0], mid, hist_m, cnt_m), (axes[1, 1], rmid, hist_r, cnt_r)):
                good = (cn[0] >= nmin) & (cn[k] >= nmin)
                ax.plot(x, np.where(good, hs[0] / hs[k], np.nan), color=c, lw=1.6, label=lab)
            q0, qk = quant[0], quant[k]
            axes[1, 2].plot(mid, q0[1] / qk[1], color=c, lw=2, label=lab)
            axes[1, 2].fill_between(mid, q0[0] / qk[0], q0[2] / qk[2], color=c, alpha=0.2)
    axes[1, 0].set(xscale='log', xlabel='multiplicity', ylabel='ratio')
    axes[1, 1].set(xscale='log', xlabel=r'merger rate [Gyr$^{-1}$]', ylabel='ratio')
    axes[1, 2].set(xscale='log', xlabel='multiplicity', ylabel='ratio')
    for ax in axes[1]:
        ax.axhline(1, color='k', lw=0.8)
    axes[1, 0].legend(fontsize=7, loc='upper left')
    fig.suptitle(f'z={main["redshift"]:.3g}   mult >= {mult_min}', y=1.0, va='bottom')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def run_subset(simdir, nfiles, mult_min, nworkers, seed, comm):
    """Sample ``nfiles`` random maplog files of ``simdir`` and return the per-MAP arrays on rank 0 (None elsewhere)."""
    rank, size = (comm.Get_rank(), comm.Get_size()) if comm is not None else (0, 1)
    files = sorted(glob(str(Path(simdir) / 'maplogs/*/maplog_z*_*.asdf')))
    if not files:
        raise FileNotFoundError(f'no maplog files under {simdir}')
    nfiles = min(nfiles, len(files))
    chosen = sorted(random.Random(seed).sample(range(len(files)), nfiles))   # identical on every rank
    mine = chosen[rank::size]
    if rank == 0:
        print(f'{Path(simdir).name}: {len(files)} files found, {nfiles} sampled (seed {seed}), {size} rank(s), '
              f'{nworkers} workers/rank', flush=True)
    t0 = time.time()
    results = []
    if mine:
        with ProcessPoolExecutor(max_workers=max(1, min(nworkers, len(mine)))) as ex:
            futs = [(i, ex.submit(_merger_rate_one_file, files[i], mult_min)) for i in mine]
            for i, fut in futs:
                r = fut.result()
                r['file'] = i
                results.append(r)
                print(f'[rank {rank}] {Path(files[i]).parent.name}/{Path(files[i]).name}: '
                      f'{len(r["mult"]):,} MAPs  ({time.time() - t0:.0f} s)', flush=True)
    gathered = comm.gather(results, root=0) if comm is not None else [results]
    if rank != 0:
        return None
    allres = [r for g in gathered for r in g]
    steps = {r['step'] for r in allres}
    assert len(steps) == 1, f'files disagree on the final timeslice: {steps}'
    return {'mult': np.concatenate([r['mult'] for r in allres]),
            'n_merger': np.concatenate([r['n_merger'] for r in allres]),
            'rate': np.concatenate([r['rate'] for r in allres]),
            'file': np.concatenate([np.full(len(r['mult']), r['file'], dtype=np.int32) for r in allres]),
            'redshift': allres[0]['redshift'], 'nfiles': nfiles, 'label': Path(simdir).name,
            'seconds': time.time() - t0}


def _save_run(out_dir, run, mult_min, seed):
    """Write one run's per-MAP arrays (``file_index`` -- not ``file``, which collides with np.savez's own argument)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'merger_rate_subset_{run["label"]}.npz'
    np.savez(path, mult=run['mult'], n_merger=run['n_merger'], rate=run['rate'], file_index=run['file'],
             redshift=run['redshift'], nfiles=run['nfiles'], label=run['label'], seconds=run['seconds'],
             mult_min=mult_min, seed=seed)
    print(f'saved {path}', flush=True)


def _load_run(out_dir, label):
    with np.load(Path(out_dir) / f'merger_rate_subset_{label}.npz') as z:
        return {'mult': z['mult'], 'n_merger': z['n_merger'], 'rate': z['rate'], 'file': z['file_index'],
                'redshift': float(z['redshift']), 'nfiles': int(z['nfiles']), 'label': str(z['label']),
                'seconds': float(z['seconds'])}


def _report(runs):
    for run in runs:
        ok = np.isfinite(run['rate'])
        print(f'{run["label"]}: {len(run["mult"]):,} MAPs, {ok.sum():,} with a rate; median rate '
              f'{np.median(run["rate"][ok]):.3g} /Gyr; {run["seconds"]:.0f} s', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('simdir')
    ap.add_argument('--nfiles', type=int, required=True, help='number of maplog files to sample at random')
    ap.add_argument('--mult_min', type=int, default=100, help='keep final MAPs with multiplicity >= this')
    ap.add_argument('--nworkers', type=int, default=16,
                    help='processes per rank; each holds one fully decompressed file, so watch node memory')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--reference', default=_REF_DEFAULT, help='reference run to overlay (same selection, same --seed)')
    ap.add_argument('--ref_nfiles', type=int, default=None, help='files sampled from the reference (default: --nfiles)')
    ap.add_argument('--no_reference', action='store_true')
    ap.add_argument('--replot', action='store_true',
                    help='no reading: redo the figure from the saved npz files of an earlier run (run without mpiexec)')
    ap.add_argument('--out_dir', default='analysis_data/merger_rate_subset', help='results (.npz) directory')
    ap.add_argument('--image_dir', default=str(Path(__file__).resolve().parent.parent / 'Images'),
                    help='figure (.png) directory')
    args = ap.parse_args()

    if args.replot:
        runs = [_load_run(args.out_dir, Path(args.simdir).name or Path(args.simdir).parent.name)]
        if not args.no_reference:
            runs.append(_load_run(args.out_dir, Path(args.reference).name))
        Path(args.image_dir).mkdir(parents=True, exist_ok=True)
        _plot(runs, args.mult_min, Path(args.image_dir) / f'merger_rate_subset_{runs[0]["label"]}.png')
        _report(runs)
        return

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # each run is written to disk as soon as it is read, so a failure later (e.g. in the figure) loses nothing
    main_run = run_subset(args.simdir, args.nfiles, args.mult_min, args.nworkers, args.seed, comm)
    if rank == 0:
        _save_run(args.out_dir, main_run, args.mult_min, args.seed)
    ref_run = None
    if not args.no_reference:
        ref_run = run_subset(args.reference, args.ref_nfiles or args.nfiles, args.mult_min, args.nworkers,
                             args.seed, comm)
        if rank == 0:
            _save_run(args.out_dir, ref_run, args.mult_min, args.seed)
    if rank != 0:
        return

    runs = [main_run] + ([ref_run] if ref_run is not None else [])
    Path(args.image_dir).mkdir(parents=True, exist_ok=True)
    _plot(runs, args.mult_min, Path(args.image_dir) / f'merger_rate_subset_{main_run["label"]}.png')
    _report(runs)
    print(f'saved {args.image_dir}/merger_rate_subset_{main_run["label"]}.png', flush=True)


if __name__ == '__main__':
    main()
