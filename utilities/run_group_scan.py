"""Time-resolved central / infalling / orbiting classification of every MAP in a simulation's maplogs
(``read_forest`` -> ``scan_epochs`` -> optional row-matched label files). Run on a COMPUTE node:

    python ~/map_analysis/utilities/run_group_scan.py SIM --out /flare/Abacus/ajohnson/group_scan/NAME [--min-mult 100] [--labels]

Always runs the sanity checks first (forest heads vs the dominant-branch reader on one file, the
timeslice-vs-merger ordering assumption) and writes ``summary.json`` + ``diag.npz`` to ``--out``.
Never writes under /flare/Abacus/AbacusAurora (the simulations).
"""
import argparse, json, sys, time
from glob import glob
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import group_finding as gf


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def check_forest(T, sim, min_mult, nfiles=2):
    """Heads of the forest must equal the dominant-branch reader's heads (every tree head survives to the final
    epoch), with identical positions; absorbed tracks must have no timeslice node at their merge epoch."""
    flist = sorted(glob(str(Path(sim) / "maplogs/*/maplog_z*_*.asdf")))
    res = {}
    for i in np.unique(np.linspace(0, len(flist) - 1, nfiles).astype(int)):
        mb = gf._main_branch_one_file(flist[i], min_mult, False)
        sel = np.flatnonzero((T.file_index == i) & (T.parent_track < 0) & (T.last_k == len(T.grid) - 1))
        same_rows = np.array_equal(np.sort(mb['head_row']), np.sort(T.head_row[sel]))
        ok_pos = False
        if same_rows:
            a = mb['pos'][np.argsort(mb['head_row'])]
            b = T.pos[sel[np.argsort(T.head_row[sel])]]
            ok_pos = bool(np.array_equal(a, b, equal_nan=True))
        res[int(i)] = dict(n_main_branch=int(len(mb['head_row'])), n_forest_heads=int(len(sel)),
                           same_head_rows=bool(same_rows), same_positions=ok_pos)
    ab = np.flatnonzero((T.parent_track >= 0) & (T.merge_k < len(T.grid)))
    res['absorbed_with_sample_at_merge_epoch'] = int((T.row[ab, T.merge_k[ab]] >= 0).sum())
    res['n_absorbed'] = int(len(ab))
    prim = T.parent_track[ab]
    res['primary_missing_at_merge_epoch'] = int((T.row[prim, T.merge_k[ab]] < 0).sum())
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sim')
    ap.add_argument('--out', required=True)
    ap.add_argument('--min-mult', type=int, default=100)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--host-velocity', default='corrected', choices=['raw', 'corrected'])
    ap.add_argument('--f-min', type=float, default=0.2)
    ap.add_argument('--n-settle', type=int, default=3)
    ap.add_argument('--ever-rule', default='or', choices=['or', 'primary'])
    ap.add_argument('--prune-after', type=int, default=None)
    ap.add_argument('--labels', action='store_true', help='write the row-matched <maplog>.rowlabels.npz files')
    ap.add_argument('--save-scan', action='store_true', help='also save the (N,K) scan arrays to scan.npz')
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if str(out).startswith('/flare/Abacus/AbacusAurora'):
        sys.exit("refusing to write under the simulation tree")
    out.mkdir(parents=True, exist_ok=True)
    log('reading forest', a.sim)
    T = gf.read_forest(a.sim, min_mult=a.min_mult, with_rel_vel=a.host_velocity == 'corrected', nworkers=a.workers)
    N, K = T.mult.shape
    log(f'{N} tracks, {K} epochs (z {T.redshift[0]:.2f} -> {T.redshift[-1]:.2f}); absorbed {(T.parent_track >= 0).sum()}')
    chk = check_forest(T, a.sim, a.min_mult)
    log('checks:', json.dumps(chk))
    log('scanning')
    t0 = time.time()
    sc = gf.scan_epochs(T, host_velocity=a.host_velocity, f_min=a.f_min, n_settle=a.n_settle,
                        ever_rule=a.ever_rule, prune_after=a.prune_after, workers=a.workers, verbose=True)
    log(f'scan done in {time.time() - t0:.0f} s')
    count = lambda v: [int((sc.status[:, k] == v).sum()) for k in range(K)]
    summ = dict(sim=a.sim, n_tracks=int(N), n_epochs=int(K), redshift=T.redshift.tolist(), checks=chk,
                params=sc.params, n_central=count(0), n_infalling=count(1), n_orbiting=count(2), n_absorbed=count(3),
                n_ever_satellite=[int(sc.ever_satellite[:, k].sum()) for k in range(K)],
                n_lower_limit=[int(((sc.lower_reason[:, k] > 0) & (sc.status[:, k] == 2)).sum()) for k in range(K)],
                scan_seconds=time.time() - t0)
    (out / 'summary.json').write_text(json.dumps(summ, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    np.savez(out / 'diag.npz', redshift=T.redshift, **sc.diag)
    log('satellite (orbiting) fraction of alive MAPs, first / middle / last epoch:',
        [round(summ['n_orbiting'][k] / max(1, summ['n_central'][k] + summ['n_infalling'][k] + summ['n_orbiting'][k]), 3)
         for k in (0, K // 2, K - 1)])
    if a.save_scan:
        np.savez(out / 'scan.npz', status=sc.status, host=sc.host, infall_epoch=sc.infall_epoch, peri_epoch=sc.peri_epoch,
                 lower_reason=sc.lower_reason, r_over_R=sc.r_over_R, ever_satellite=sc.ever_satellite,
                 ever_epoch=sc.ever_epoch, ever_host=sc.ever_host, recent_merger=sc.recent_merger, rank=sc.rank,
                 file_index=T.file_index, head_row=T.head_row, row=T.row, redshift=T.redshift)
        log('saved scan.npz')
    if a.labels:
        paths = gf.write_row_labels(T, sc, a.sim, out / 'rowlabels', min_mult=a.min_mult)
        log(f'wrote {len(paths)} row-label files under {out / "rowlabels"}')
    log('done')


if __name__ == '__main__':
    main()
