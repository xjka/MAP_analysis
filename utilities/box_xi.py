#!/usr/bin/env python
"""Real-space MAP correlation function xi(r) in a random non-periodic box of a huge run.

Reads the box with ``box_maps`` (see there for how the box and its files are chosen),
abundance-matches the MAPs inside it (top ``abundance * S^3`` by multiplicity, the same rule
as ``utilities.get_threshold``), counts pairs without periodic wrapping and saves only the
results (npz + png) -- the MAPs themselves are never written.

Because the survey volume is a plain box, no random catalog is needed: for two independent
uniform points in a box with sides L_i, the probability that their separation lies in a
bin is exactly

    P(bin) = V^-2 * integral_bin d^3s  prod_i (L_i - |s_i|),

so the expected random pair count is RR = N (N-1) P(bin). The integral is done by quadrature
(exact for r < min(L_i), where the integrand is a smooth polynomial; ``--rmax`` must be below
the box side). Two estimators use it:

* ``landyszalay`` (default): xi = (DD - 2 DR + RR)/RR with the analytic RR and DR counted
  against a uniform random catalog of ``--nrandoms`` x N points in total (each rank draws its share from the
  seed and counts its slice of the data against it). The DR term cancels the chance density gradients/clumps of the finite sample that
  the edge-weighted pair counts of a non-periodic box are sensitive to; analytic RR means
  RR itself carries no shot noise and is never counted.
* ``natural``: xi = DD/RR - 1, no randoms at all. Cheaper, but on a test Poisson sample
  (N=6e4) it scattered by ~0.5% at r = 0.5 L where Landy-Szalay was at the 1e-4 level,
  so it is only suitable where xi is much larger than that.

Parallelism: files are read across MPI ranks (one rank per node) and a process pool within
each. The DD counts are then split across ranks too: each rank counts its strided slice of
the MAPs against all of them (pycorr/Corrfunc, ``--nthreads`` threads) and the counts are
summed.

``--repeat K`` measures K independent random boxes in one invocation (the reference is read
once, and cached on disk -- ``<out_dir>/ref_xi_*.npz``, keyed on reference, abundance and bins;
``--refresh_reference`` rebuilds it) and writes a summary: the mean xi over boxes, the box-to-box
scatter, and the ratio to the reference with the error on the mean. The K origins come from
``--seed`` (printed), so the set can be reproduced; ``--origin`` is for a single box only.

    mpiexec -n 4 --ppn 1 python ~/map_analysis/utilities/box_xi.py SIMDIR --box_nzslices 3 [--seed S | --origin X Y Z] [--repeat K]
"""
import argparse
import os
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
from pycorr import TwoPointCounter

from box_maps import choose_box, describe_box, final_timeslice_step, read_box_maps, read_decomposition

_PC = dict(position_type='pos', bin_type='custom', engine='corrfunc')


def box_pair_probability(edges, sizes, nquad=24):
    """P[bin] that two independent uniform points in a box of ``sizes`` (3,) have separation
    in ``[edges[i], edges[i+1])``. Exact (to quadrature) for ``edges[-1] <= min(sizes)``."""
    sizes = np.asarray(sizes, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if edges[-1] > sizes.min():
        raise ValueError(f'rmax={edges[-1]:.3g} exceeds the smallest box side {sizes.min():.3g}')
    x, w = np.polynomial.legendre.leggauss(nquad)
    th, wth = 0.25 * np.pi * (x + 1), 0.25 * np.pi * w            # octant: theta, phi in [0, pi/2]
    st, ct = np.sin(th)[:, None], np.cos(th)[:, None]
    ph = th[None, :]
    nvec = np.stack([np.broadcast_to(st * np.cos(ph), (nquad, nquad)),
                     np.broadcast_to(st * np.sin(ph), (nquad, nquad)),
                     np.broadcast_to(ct, (nquad, nquad))])          # (3, nth, nph) unit vectors
    wang = 8 * (wth[:, None] * wth[None, :]) * st                    # 8 octants x sin(theta) dtheta dphi
    xr, wr = np.polynomial.legendre.leggauss(8)
    out = np.empty(len(edges) - 1)
    for i, (r1, r2) in enumerate(zip(edges[:-1], edges[1:])):
        r = 0.5 * (r2 - r1) * xr + 0.5 * (r2 + r1)
        wrr = 0.5 * (r2 - r1) * wr * r**2
        f = np.prod(sizes[:, None, None, None] - r[None, :, None, None] * nvec[:, None], axis=0)  # (nr, nth, nph)
        out[i] = np.sum(wrr[:, None, None] * wang[None] * f)
    return out / np.prod(sizes)**2


def count_pairs(pos1, pos2, edges, comm, nthreads, sepsavg=False, auto=False):
    """Ordered pair counts (and summed separations, if ``sepsavg``) per bin, no periodic wrap.
    ``pos1`` is strided over MPI ranks; each rank counts its slice against all of ``pos2`` (a
    rank-local array for a cross count) and the results are summed over ranks. ``auto=True``
    means ``pos2`` *is* the full ``pos1`` (identical on every rank, so slices don't overlap)."""
    rank, size = (comm.Get_rank(), comm.Get_size()) if comm is not None else (0, 1)
    sl = pos1[rank::size] if auto else pos1
    c = TwoPointCounter('s', edges, positions1=sl, positions2=pos2, compute_sepsavg=sepsavg,
                        nthreads=nthreads, **_PC)
    counts = np.asarray(c.wcounts, dtype=float)
    sepsum = np.nan_to_num(np.asarray(c.sepavg(), dtype=float)) * counts if sepsavg else None
    if comm is not None:
        from mpi4py import MPI
        counts = comm.allreduce(counts, op=MPI.SUM)
        if sepsavg:
            sepsum = comm.allreduce(sepsum, op=MPI.SUM)
    return counts, sepsum


_REF_DEFAULT = '/flare/Abacus/AbacusAurora/AbacusAurora_emulatormax_c200_ph310'


def _read_reference_file(f, top_k):
    """Final-timeslice MAPs of one reference maplog file, pre-cut to its ``top_k`` highest multiplicities."""
    asdf.get_config().validate_on_read = False
    final = final_timeslice_step(f)
    tbl = read_asdf(f, load=['pos', 'mult'], select={'node_type': 'timeslice', 'timestep': final})
    if len(tbl) == 0:
        raise RuntimeError(f'{f}: no TimeSlice nodes at the final step {final}')
    mult = np.asarray(tbl['mult'])
    pos = np.asarray(tbl['pos'])
    cut = 0
    if top_k < len(mult):
        cut = int(np.partition(mult, len(mult) - top_k)[len(mult) - top_k])
        keep = mult >= cut
        pos, mult = pos[keep], mult[keep]
    return {'pos': pos.astype(np.float32), 'mult': mult.astype(np.int32), 'cut': cut,
            'redshift': float(np.asarray(tbl.meta['Step_to_Z'])[final - 1])}


def reference_xi(refdir, edges, abundance, safety, nworkers, nthreads, comm):
    """Periodic-box xi(r) of a reference run at the same abundance and bins (rank 0's dict, None elsewhere).

    Files are read like ``read_box_maps`` (dealt over ranks, a process pool per rank), each pre-cut to
    ``safety * abundance * L^3 / nfiles`` MAPs (files are assumed to hold similar numbers of MAPs), then
    the exact abundance threshold is applied on rank 0 and ``utilities.get_xi`` (Corrfunc, periodic,
    natural estimator -- exact for a periodic box) counts the pairs."""
    rank, size = (comm.Get_rank(), comm.Get_size()) if comm is not None else (0, 1)
    asdf.get_config().validate_on_read = False
    files = sorted(glob(str(Path(refdir) / 'maplogs/*/maplog_z*_*.asdf')))
    if not files:
        raise FileNotFoundError(f'no maplog files under {refdir}')
    with asdf.open(files[0], lazy_load=True, validate_checksums=False) as af:
        L = float(af.tree['header']['BoxSize'])
    top_k = int(np.ceil(safety * abundance * L**3 / len(files)))
    mine = files[rank::size]
    t0 = time.time()
    out = []
    if mine:
        with ProcessPoolExecutor(max_workers=max(1, min(nworkers, len(mine)))) as ex:
            futs = [ex.submit(_read_reference_file, f, top_k) for f in mine]
            for fut in futs:
                out.append(fut.result())
    gathered = comm.gather(out, root=0) if comm is not None else [out]
    if rank != 0:
        return None
    allr = [r for g in gathered for r in g]
    mult = np.concatenate([r['mult'] for r in allr])
    pos = np.concatenate([r['pos'] for r in allr])
    ntarget = int(round(abundance * L**3))
    if len(mult) < ntarget:
        raise SystemExit(f'reference: only {len(mult):,} pre-selected MAPs but abundance {abundance:g} needs {ntarget:,}')
    thr = int(np.partition(mult, len(mult) - ntarget)[len(mult) - ntarget])
    if thr < max(r['cut'] for r in allr):
        raise SystemExit(f'reference threshold {thr} is below a per-file pre-cut; rerun with a larger --top_safety')
    pos = pos[mult >= thr]
    print(f'reference {Path(refdir).name}: {len(files)} files read in {time.time() - t0:.0f} s; {len(pos):,} MAPs with '
          f'mult >= {thr} (L={L:g}, z={allr[0]["redshift"]:.3f})', flush=True)
    from astropy import units as u
    from utilities import get_xi
    t0 = time.time()
    twopt = get_xi(pos, edges, L * u.Mpc, max_p=np.inf, nthreads=nthreads)
    print(f'reference pair counts in {time.time() - t0:.1f} s', flush=True)
    return {'xi': np.asarray(twopt.get_corr(return_sep=False)), 'N': len(pos), 'threshold': thr, 'L': L,
            'redshift': allr[0]['redshift'], 'name': Path(refdir).name}


def _ref_cache_path(args):
    key = f'{Path(args.reference).name}_n{args.abundance:g}_r{args.rmin:g}-{args.rmax:g}_{args.nbins}'
    return Path(args.out_dir) / f'ref_xi_{key}.npz'


def get_reference(args, edges, nthreads, comm):
    """Reference xi (rank 0's dict, None elsewhere), from the on-disk cache if present, else computed
    (collectively over all ranks) and cached."""
    rank = comm.Get_rank() if comm is not None else 0
    path = _ref_cache_path(args)
    hit = (not args.refresh_reference) and path.is_file() if rank == 0 else None
    hit = comm.bcast(hit, root=0) if comm is not None else hit
    if hit:
        if rank != 0:
            return None
        with np.load(path) as z:
            ref = {k: z[k] for k in z.files}
        ref.update(name=str(ref['name']), L=float(ref['L']), N=int(ref['N']), threshold=int(ref['threshold']),
                   redshift=float(ref['redshift']))
        print(f'reference xi loaded from cache {path}', flush=True)
        return ref
    ref = reference_xi(args.reference, edges, args.abundance, args.top_safety, args.nworkers, nthreads, comm)
    if rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + '.tmp.npz')
        np.savez(tmp, **ref)
        tmp.rename(path)
        print(f'reference xi cached -> {path}', flush=True)
    return ref


def measure_box(d, args, box, seed, edges, nthreads, comm):
    """xi(r) in one box (collective over ranks; the result dict is returned on rank 0, None elsewhere)."""
    rank = comm.Get_rank() if comm is not None else 0
    nsz = comm.Get_size() if comm is not None else 1
    S = box.size
    t0 = time.time()
    res = read_box_maps(d, box, nworkers=args.nworkers, comm=comm,
                        top_density=args.abundance, top_safety=args.top_safety)
    pos = None
    if rank == 0:
        mult = res['mult']
        ntarget = int(round(args.abundance * S**3))
        if len(mult) < ntarget:
            raise SystemExit(f'only {len(mult):,} MAPs in the box but abundance {args.abundance:g} needs {ntarget:,}')
        thresh = int(np.partition(mult, len(mult) - ntarget)[len(mult) - ntarget])
        if thresh < res['local_cut']:
            raise SystemExit(f'box multiplicity threshold {thresh} is below a per-file pre-cut ({res["local_cut"]}); '
                             'rerun with a larger --top_safety')
        sel = mult >= thresh
        pos = np.ascontiguousarray(res['pos'][sel])
        print(f'read in {time.time() - t0:.0f} s; {len(mult):,} pre-selected -> {len(pos):,} MAPs with mult >= {thresh} '
              f'(target {ntarget:,}, abundance {args.abundance:g}, z={res["redshift"]:.3f})', flush=True)
    if comm is not None:
        n = comm.bcast(len(pos) if rank == 0 else None, root=0)
        buf = pos if rank == 0 else np.empty((n, 3), dtype=np.float32)
        comm.Bcast(buf, root=0)
        pos = buf
    N = len(pos)

    t0 = time.time()
    dd, sepsum = count_pairs(pos, pos, edges, comm, nthreads, sepsavg=True, auto=True)
    P = box_pair_probability(edges, [S, S, S])
    rr = N * (N - 1) * P
    xi_nat = dd / rr - 1
    xi = xi_nat
    if args.estimator == 'landyszalay':
        # each rank draws its own randoms (independent seeds) and counts its data slice against them
        rng = np.random.default_rng([0 if seed is None else int(seed), rank])
        nr = int(np.ceil(args.nrandoms * N / nsz))      # per rank; the ranks' sets are independent, so the total is nrandoms * N
        rand = rng.uniform(0, S, (nr, 3)).astype(np.float32)
        dr, _ = count_pairs(pos[rank::nsz], rand, edges, comm, nthreads)
        # total data-random ordered pairs; the data are split over ranks, so normalise by N * (nr per rank)
        xi = dd / rr - 2 * dr / (N * nr * P) + 1
    if rank != 0:
        return None
    print(f'pair counts in {time.time() - t0:.1f} s', flush=True)
    r = np.where(dd > 0, sepsum / np.maximum(dd, 1), np.sqrt(edges[:-1] * edges[1:]))
    err = (1 + xi) / np.sqrt(np.maximum(dd / 2, 1))        # Poisson on unordered pairs; underestimates (pairs are correlated)
    return dict(r=r, xi=xi, xi_natural=xi_nat, err_poisson=err, edges=edges, DD=dd, RR=rr, N=N, mult_threshold=thresh,
                box_size=S, origin=box.origin, seed=-1 if seed is None else seed, nzslices=args.box_nzslices,
                abundance=args.abundance, redshift=res['redshift'], estimator=args.estimator)


def reference_poisson_error(ref, edges):
    """Poisson error on the periodic reference xi, same convention as the box's ``err_poisson``.

    The reference is a periodic box, so its random pair count is exact: RR = N (N-1) V_shell / L^3 (ordered pairs),
    DD = (1 + xi) RR, and err = (1 + xi) / sqrt(DD / 2) (unordered pairs; an underestimate since pairs are correlated).
    Needs only the cached xi, N, L, so cached reference files work without recomputation.
    """
    vshell = 4 * np.pi / 3 * np.diff(np.asarray(edges, dtype=float) ** 3)
    rr = ref['N'] * (ref['N'] - 1) * vshell / ref['L'] ** 3
    dd = np.maximum((1 + ref['xi']) * rr, 0)
    return (1 + ref['xi']) / np.sqrt(np.maximum(dd / 2, 1))


def _plot_box(out, ref, label, args, path):
    r, xi, xi_nat, err, S, N = out['r'], out['xi'], out['xi_natural'], out['err_poisson'], out['box_size'], out['N']
    fig, axes = plt.subplots(1, 3 if ref is not None else 2, figsize=(15 if ref is not None else 10, 4),
                             constrained_layout=True)
    if ref is not None:
        err_ref = reference_poisson_error(ref, out['edges'])
    for ax, ys, yl in ((axes[0], r**2 * xi, r'$r^2\,\xi(r)$ [(Mpc/$h$)$^2$]'), (axes[1], xi, r'$\xi(r)$')):
        f = 1 if ys is xi else r**2
        ax.errorbar(r, ys, yerr=f * err, fmt='o-', ms=3, label=f'box ({S:.0f} Mpc/$h$), {args.estimator}')
        if args.estimator != 'natural':
            ax.plot(r, f * xi_nat, 's--', ms=3, alpha=0.6, label='box, natural')
        if ref is not None:
            ax.plot(r, f * ref['xi'], '-', color='k', lw=1.2, label=f'{ref["name"]} (periodic, {ref["L"]:.0f} Mpc/$h$)')
            ax.fill_between(r, f * (ref['xi'] - err_ref), f * (ref['xi'] + err_ref), color='k', alpha=0.2, lw=0)
        ax.set(xscale='log', xlabel=r'$r$ [Mpc/$h$]', ylabel=yl)
    axes[1].set_yscale('log')
    axes[1].set_ylim(bottom=max(1e-3, np.nanmin(np.where(xi > 0, xi, np.nan)) / 2) if np.any(xi > 0) else None)
    axes[0].legend(fontsize=7)
    if ref is not None:
        ax = axes[2]
        ax.errorbar(r, xi / ref['xi'], yerr=err / np.abs(ref['xi']), fmt='o-', ms=3)
        if args.estimator != 'natural':
            ax.plot(r, xi_nat / ref['xi'], 's--', ms=3, alpha=0.6)
        ax.axhline(1, color='k', lw=0.8)
        rel = err_ref / np.abs(ref['xi'])
        ax.fill_between(r, 1 - rel, 1 + rel, color='k', alpha=0.2, lw=0)
        ax.set(xscale='log', xlabel=r'$r$ [Mpc/$h$]', ylabel=r'$\xi_{\rm box}\,/\,\xi_{\rm ref}$')
    fig.suptitle(f'{label}  z={out["redshift"]:.3g}  box {S:.0f} Mpc/h  N={N:,} (n={args.abundance:g})', y=1.0, va='bottom')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_summary(results, ref, label, args, path):
    """Mean xi over boxes with the box-to-box scatter, and its ratio to the reference."""
    K = len(results)
    r = np.mean([o['r'] for o in results], axis=0)
    xis = np.array([o['xi'] for o in results])
    mean, std = xis.mean(0), xis.std(0, ddof=1)
    S = results[0]['box_size']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for x in xis:
        axes[0].plot(r, r**2 * x, color='C0', lw=0.6, alpha=0.35)
    axes[0].errorbar(r, r**2 * mean, yerr=r**2 * std / np.sqrt(K), color='C0', lw=2, label=f'mean of {K} boxes ({S:.0f} Mpc/$h$)')
    if ref is not None:
        axes[0].plot(r, r**2 * ref['xi'], 'k-', lw=1.2, label=f'{ref["name"]} (periodic)')
    axes[0].set(xscale='log', xlabel=r'$r$ [Mpc/$h$]', ylabel=r'$r^2\,\xi(r)$ [(Mpc/$h$)$^2$]')
    axes[0].legend(fontsize=8)
    ax = axes[1]
    if ref is not None:
        q = xis / ref['xi']
        for x in q:
            ax.plot(r, x, color='C0', lw=0.6, alpha=0.35)
        ax.fill_between(r, q.mean(0) - q.std(0, ddof=1), q.mean(0) + q.std(0, ddof=1), color='C0', alpha=0.2,
                        label='box-to-box scatter (1$\\sigma$)')
        ax.errorbar(r, q.mean(0), yerr=q.std(0, ddof=1) / np.sqrt(K), color='C0', lw=2, label='mean, error on the mean')
        ax.axhline(1, color='k', lw=0.8)
        ax.set(ylabel=r'$\xi_{\rm box}\,/\,\xi_{\rm ref}$')
    else:
        ax.plot(r, std / np.abs(mean), 'o-', ms=3)
        ax.set(ylabel=r'box-to-box scatter / $\xi$', yscale='log')
    ax.set(xscale='log', xlabel=r'$r$ [Mpc/$h$]')
    if ref is not None:
        ax.legend(fontsize=8)
    fig.suptitle(f'{label}  z={results[0]["redshift"]:.3g}  {K} boxes of {S:.0f} Mpc/h  (n={args.abundance:g})', y=1.0, va='bottom')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('simdir')
    ap.add_argument('--box_nzslices', type=int, required=True)
    ap.add_argument('--origin', type=float, nargs=3, metavar=('X', 'Y', 'Z'), help='box lower corner (single box only)')
    ap.add_argument('--seed', type=int, default=None, help='seed for the random origin(s) (default: fresh entropy, printed)')
    ap.add_argument('--repeat', type=int, default=1, help='number of independent random boxes to measure and average')
    ap.add_argument('--abundance', type=float, default=5.01e-3, help='MAP number density [(h/Mpc)^3]')
    ap.add_argument('--rmin', type=float, default=2e-2)
    ap.add_argument('--rmax', type=float, default=40.0, help='must be below the box side')
    ap.add_argument('--nbins', type=int, default=30)
    ap.add_argument('--estimator', choices=['natural', 'landyszalay'], default='landyszalay')
    ap.add_argument('--nrandoms', type=float, default=40,
                    help='landyszalay: TOTAL randoms (summed over ranks) as a multiple of N, DR only; DR noise ~ 1/sqrt(nrandoms)')
    ap.add_argument('--nworkers', type=int, default=16, help='file-reading processes per rank')
    ap.add_argument('--nthreads', type=int, default=None, help='pair-counting threads per rank (default: all cores)')
    ap.add_argument('--top_safety', type=float, default=1.25,
                    help='per-file pre-cut keeps this multiple of the target MAP count (files are assumed to hold similar numbers)')
    ap.add_argument('--reference', default=_REF_DEFAULT, help='periodic reference run to compare against')
    ap.add_argument('--no_reference', action='store_true', help='skip the reference comparison')
    ap.add_argument('--refresh_reference', action='store_true', help='recompute the reference xi instead of using the cache')
    ap.add_argument('--per_box_images', action='store_true', help='with --repeat, also save a figure for every box')
    ap.add_argument('--x_slab_offset', type=int, default=None)
    ap.add_argument('--out_dir', default='analysis_data/box_xi', help='results (.npz) directory')
    ap.add_argument('--image_dir', default=str(Path(__file__).resolve().parent.parent / 'Images'), help='figure (.png) directory')
    args = ap.parse_args()
    if args.repeat > 1 and args.origin is not None:
        raise SystemExit('--origin selects one box; use --seed with --repeat')

    comm = None
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    except ImportError:
        pass
    rank = comm.Get_rank() if comm is not None else 0
    nthreads = args.nthreads or len(os.sched_getaffinity(0))

    d = read_decomposition(args.simdir, args.x_slab_offset)
    seed = args.seed
    if comm is not None:
        seed = comm.bcast(np.random.SeedSequence().entropy if (seed is None and rank == 0) else seed, root=0)
    K = args.repeat
    # one box: the seed is used directly (as before); several: one derived seed per box
    seeds = [seed] if K == 1 else [int(x) for x in np.random.SeedSequence(seed).generate_state(K, dtype=np.uint64)]
    edges = np.logspace(np.log10(args.rmin), np.log10(args.rmax), args.nbins + 1)
    S = args.box_nzslices * d.L / d.nz
    if edges[-1] > S:
        raise SystemExit(f'--rmax {args.rmax} exceeds the box side {S:.2f}')
    if rank == 0 and args.origin is None:
        print(f'random origin(s) from seed {seed}' + (f' (repeat with --seed {seed} --repeat {K})' if K > 1 else ''), flush=True)

    ref = None if args.no_reference else get_reference(args, edges, nthreads, comm)

    label = Path(args.simdir).name or Path(args.simdir).parent.name
    od, imgd = Path(args.out_dir), Path(args.image_dir)
    results = []
    for i, bseed in enumerate(seeds):
        box = choose_box(d, args.box_nzslices, origin=args.origin, seed=bseed)
        if rank == 0:
            print(f'=== box {i + 1}/{K} ===\n' + describe_box(d, box), flush=True)
            if args.origin is None:
                print(f'repeat this box with --origin {" ".join(f"{v:.6f}" for v in box.origin)}', flush=True)
        out = measure_box(d, args, box, bseed, edges, nthreads, comm)
        if rank != 0:
            continue
        if ref is not None:
            out.update(xi_ref=ref['xi'], ref_name=ref['name'], ref_N=ref['N'], ref_threshold=ref['threshold'],
                       ref_box_size=ref['L'], ref_redshift=ref['redshift'])
        tag = f'{label}_n{args.box_nzslices}_' + '_'.join(f'{v:.1f}' for v in box.origin)
        od.mkdir(parents=True, exist_ok=True)
        np.savez(od / f'box_xi_{tag}.npz', **out)
        saved = [f'{od}/box_xi_{tag}.npz']
        if K == 1 or args.per_box_images:
            imgd.mkdir(parents=True, exist_ok=True)
            _plot_box(out, ref, label, args, imgd / f'box_xi_{tag}.png')
            saved.append(f'{imgd}/box_xi_{tag}.png')
        print('saved ' + ' and '.join(saved), flush=True)
        results.append(out)

    if rank == 0 and K > 1:
        tag = f'{label}_n{args.box_nzslices}_mean{K}_seed{str(seed)[:8]}'
        summ = dict(r=np.array([o['r'] for o in results]), xi=np.array([o['xi'] for o in results]),
                    xi_natural=np.array([o['xi_natural'] for o in results]), edges=edges,
                    origins=np.array([o['origin'] for o in results]), N=np.array([o['N'] for o in results]),
                    mult_threshold=np.array([o['mult_threshold'] for o in results]), seed=seed,
                    abundance=args.abundance, box_size=S, redshift=results[0]['redshift'])
        if ref is not None:
            summ.update(xi_ref=ref['xi'], ref_name=ref['name'], ref_N=ref['N'], ref_threshold=ref['threshold'])
        np.savez(od / f'box_xi_{tag}.npz', **summ)
        imgd.mkdir(parents=True, exist_ok=True)
        _plot_summary(results, ref, label, args, imgd / f'box_xi_{tag}.png')
        if ref is not None:
            q = summ['xi'] / ref['xi']
            for j in range(0, args.nbins, 4):
                print(f'  r={summ["r"][:, j].mean():8.3g}: ratio {q[:, j].mean():.3f} +- {q[:, j].std(ddof=1) / np.sqrt(K):.3f} '
                      f'(scatter {q[:, j].std(ddof=1):.3f})', flush=True)
        print(f'saved summary {od}/box_xi_{tag}.npz and {imgd}/box_xi_{tag}.png', flush=True)


if __name__ == '__main__':
    main()
