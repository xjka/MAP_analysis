"""MAP-by-MAP comparison of two Abacus runs (e.g. old code vs fast-directs) at one MapLog timeslice.

Typical use from a notebook::

    import map_by_map_comparison as mbm
    ctx  = mbm.prepare(maplog, maplog2, sim, sim2, z_cmp=0.2, n_lookback=8)   # reads histories: slow, do once
    res  = mbm.match(ctx, mult_thresh=[5e3, 5e2])                              # fast: re-run with other parameters
    figs = mbm.plot_results(ctx, res)

Matching has two stages.

1. Trajectory tiers, largest MAPs first (large MAPs must have a counterpart, and matching them first removes the many
   small MAPs around them from the later, harder matching). Tier k takes the still-unmatched MAPs with multiplicity >=
   ``mult_thresh[k]`` (a final tier takes every remaining MAP). Distance = RMS over the last ``n_lookback`` epochs
   (current epoch included; the dominant branch back to the last merger) of the position separation. Pairs closer than
   ``d_pass1`` are linked, connected components are formed, and, by default, only the 1x1 components are matches (``tier_rule``).
2. Then an ellipse walk over the same tiers (``mult_thresh_ellipse``), again largest first and again accepting only 1x1
   components. The ellipse distance is meant to separate MAPs that are legitimately close and small: trajectory
   position, trajectory velocity and log10-multiplicity differences and the Lagrangian PID separation, each in units of
   the scatter of the trajectory matches (``var_from``). Finally (``final_mutual``) one ellipse sweep over everything
   left accepts mutually-best pairs, the only place merely-mutually-best pairs are taken. ``trajectory_walks > 1``
   repeats the trajectory walk before the ellipse walk.
"""
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, LogNorm
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from astropy import units as u

from utilities import read_maplog_histories, map_step_for_redshift, get_xi

OUTCOME_NAMES = ['matched, trajectory walk', 'matched, ellipse walk', 'unmatched (has candidates)',
                 'isolated (no candidate within D_ellipse)']


# ----------------------------------------------------------------------------------------------------------------
# setup: everything that is expensive to build or read and does not depend on the matching parameters
# ----------------------------------------------------------------------------------------------------------------
def prepare(maplog, maplog2, sim, sim2, z_cmp=0.2, n_lookback=8, nworkers=16, verbose=True):
    """Select the MAPs of both runs at the timeslice nearest ``z_cmp``, read their histories (``n_lookback`` epochs
    back, dominant branch to the last merger) and print the MAP counts. ``maplog``/``maplog2`` come from
    ``read_maplog_dir(..., compute_pids=True)``; ``sim``/``sim2`` are the run directories. Returns a context dict
    to pass to ``match`` and ``plot_results``."""
    mult_min = maplog['header'].get('Map_MinMultiplicity', 20)      # no MAP can have fewer particles than this
    L = maplog['header']['BoxSize']
    runs = []
    for ml, sdir, tag in ((maplog, sim, 'old'), (maplog2, sim2, 'fast-directs')):
        assert 'pid' in ml, "re-read the maplogs with compute_pids=True"
        step, zs = map_step_for_redshift(ml, z_cmp)
        heads = ml['control'] == 4 * step + 3                       # all timeslice heads at this step, file order
        sel = ml['mult'][heads] >= mult_min
        m = heads & (ml['mult'] >= mult_min)
        X = dict(step=step, z=zs, sdir=sdir, tag=tag, sel=sel,
                 pos=ml['pos'][m].astype(np.float64), vel=ml['vel'][m].astype(np.float64),
                 mult=ml['mult'][m].astype(np.float64), pid=ml['pid'][m],
                 dens=ml['density'][m].astype(np.float64), sigv=ml['vel_disp'][m].astype(np.float64))
        runs.append(X)
        if verbose:
            print(f"{tag}: step {step}, z={zs:.5f}")
    LA, LB = runs
    NA, NB = len(LA['mult']), len(LB['mult'])
    if verbose:
        print(f"\nMAPs with mult >= {mult_min}:  old {NA},  fast-directs {NB},  difference (old - fast-directs) = "
              f"{NA - NB} ({(NA - NB) / NA:+.4%} of old)")
        mbins = [mult_min, 30, 50, 100, 300, 1000, 1e4, np.inf]
        ca, cb = np.histogram(LA['mult'], mbins)[0], np.histogram(LB['mult'], mbins)[0]
        print(f"{'multiplicity':>16s} {'old':>10s} {'fast-directs':>13s} {'old - fd':>10s}")
        for lo, hi, a, b in zip(mbins[:-1], mbins[1:], ca, cb):
            print(f"{f'[{lo:g}, {hi:g})':>16s} {a:10d} {b:13d} {a - b:10d}")

    # histories; rows are aligned with LA/LB
    H = []
    for X in (LA, LB):
        h = read_maplog_histories(X['sdir'], X['step'], select=X.pop('sel'), n_back=n_lookback, nworkers=nworkers)
        assert len(h['kind']) == len(X['mult'])
        assert np.array_equal(h['pos'][:, 0].astype(np.float64), X['pos']), f"{X['tag']}: history rows not aligned"
        h = dict(pos=h['pos'], vel=h['vel'], mult=h['mult'], epoch=h['epoch'], kind=h['kind'],
                 truncated=h['truncated'])
        H.append(h)
        if verbose:
            nep = (h['kind'] == 3).sum(1)
            print(f"{X['tag']}: epochs since last merger (incl. current): p10/50/90 = "
                  f"{np.percentile(nep, [10, 50, 90]).tolist()}; cut off at n_lookback: {h['truncated'].mean():.1%}")

    pA = (LA['pos'] + L / 2) % L;  pA[pA >= L] = 0.                 # KD-tree coordinates in [0, L)
    pB = (LB['pos'] + L / 2) % L;  pB[pB >= L] = 0.
    ippd = int(round(float(maplog['header']['NP']) ** (1 / 3)))
    pidc = lambda h: np.stack([h & 0xFFFF, (h >> 16) & 0xFFFF, (h >> 32) & 0xFFFF], axis=1)   # p0 + p1<<16 + p2<<32
    return dict(L=L, cpd=int(maplog['header'].get('CPD', 0)), mult_min=mult_min, z_cmp=z_cmp, n_lookback=n_lookback, NA=NA, NB=NB, LA=LA, LB=LB,
                HA=H[0], HB=H[1], pA=pA, pB=pB, ippd=ippd, pcA=pidc(LA['pid']), pcB=pidc(LB['pid']))


# ----------------------------------------------------------------------------------------------------------------
# building blocks
# ----------------------------------------------------------------------------------------------------------------
def traj_features(ctx, I, J, chunk=400_000):
    """Trajectory distances between old MAPs ``I`` and fast-directs MAPs ``J`` (index arrays, one entry per pair),
    over the epochs (same MapLogRedshifts index, current epoch included) at which both have a history node.
    Returns n_common, RMS |dx| [kpc/h, periodic], RMS |dv| [km/s] and RMS dlog10(mult) [dex] over those epochs
    (NaN where there is no common epoch)."""
    HA, HB, L = ctx['HA'], ctx['HB'], ctx['L']
    n = np.zeros(len(I), np.int16)
    fx, fv, fm = (np.full(len(I), np.nan, np.float32) for _ in range(3))
    for s0 in range(0, len(I), chunk):
        sl = slice(s0, s0 + chunk)
        i, j = I[sl], J[sl]
        eA, eB = HA['epoch'][i], HB['epoch'][j]
        mA, mB = HA['mult'][i].astype(np.float32), HB['mult'][j].astype(np.float32)
        ok = (eA == eB) & (eA >= 0) & (mA > 0) & (mB > 0)
        dxx = HA['pos'][i] - HB['pos'][j]
        dxx -= L * np.round(dxx / L)
        dx2 = np.where(ok, (dxx**2).sum(2) * 1e6, np.nan)                                  # (kpc/h)^2
        dv2 = np.where(ok, ((HA['vel'][i] - HB['vel'][j])**2).sum(2), np.nan)              # (km/s)^2
        dm2 = np.where(ok, np.log10(np.where(ok, mA, 1.) / np.where(ok, mB, 1.))**2, np.nan)
        n[sl] = ok.sum(1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)       # all-NaN rows (no common epoch) -> NaN
            fx[sl], fv[sl], fm[sl] = (np.sqrt(np.nanmean(q, axis=1)) for q in (dx2, dv2, dm2))
    return n, fx, fv, fm


def mutual_best(I, J, d, nA, nB):
    """Boolean mask of the pairs (I[k], J[k]) at distance d[k] that are each other's strictly nearest candidate:
    J[k] is the best for I[k] and I[k] is the best for J[k], and neither MAP has a second candidate at the same
    distance. nA, nB are the numbers of old / fast-directs MAPs (indices run over range(nA), range(nB))."""
    if len(I) == 0:
        return np.zeros(0, bool)
    best, tie = [], []
    for a, b, n in ((I, J, nA), (J, I, nB)):
        o = np.lexsort((d, a))                       # sorted by MAP, then distance
        a_s, b_s, d_s = a[o], b[o], d[o]
        first = np.flatnonzero(np.r_[True, a_s[1:] != a_s[:-1]])
        nxt = np.minimum(first + 1, len(a_s) - 1)
        has2 = (first + 1 < len(a_s)) & (a_s[nxt] == a_s[first])
        bst = np.full(n, -1);  bst[a_s[first]] = b_s[first]
        tie_ = np.zeros(n, bool);  tie_[a_s[first][has2]] = d_s[nxt[has2]] <= d_s[first][has2]
        best.append(bst);  tie.append(tie_)
    return (best[0][I] == J) & (best[1][J] == I) & ~tie[0][I] & ~tie[1][J]


def pid_separation(ctx, I, J):
    """Periodic infinity-norm Lagrangian separation (lattice cells) between the PIDs of old MAPs I and new MAPs J."""
    a, b, ippd = ctx['pcA'][I], ctx['pcB'][J], ctx['ippd']
    return np.minimum(np.abs(a - b), ippd - np.abs(a - b)).max(axis=1)


def _candidates(ctx, eA, eB, radius):
    """Cross-run pairs (old index, fast-directs index) closer than ``radius`` [Mpc/h] in current position, among the
    MAPs ``eA`` / ``eB`` (index arrays)."""
    L = ctx['L']
    e = cKDTree(ctx['pA'][eA], leafsize=32, boxsize=L).sparse_distance_matrix(
        cKDTree(ctx['pB'][eB], leafsize=32, boxsize=L), radius, output_type='ndarray')
    return eA[e['i']], eB[e['j']]


def _shape_table(I, J, eA, eB, NA, NB, top=6):
    """Print the most common (old x fast-directs) component shapes of the link graph over MAPs eA / eB."""
    ia, ib = np.full(NA, -1), np.full(NB, -1)
    ia[eA], ib[eB] = np.arange(len(eA)), np.arange(len(eB))
    n = len(eA) + len(eB)
    g = coo_matrix((np.ones(len(I), np.int8), (ia[I], len(eA) + ib[J])), shape=(n, n))
    nc, lab = connected_components(g, directed=False)
    na, nb = np.bincount(lab[:len(eA)], minlength=nc), np.bincount(lab[len(eA):], minlength=nc)
    cu, cn = np.unique(np.minimum(na, 6) * 7 + np.minimum(nb, 6), return_counts=True)
    shapes = []
    for k in np.argsort(cn)[::-1][:top]:
        a, b = divmod(cu[k], 7)
        shapes.append(f"{a}{'+' if a == 6 else ''}x{b}{'+' if b == 6 else ''}: {cn[k]}")
    print("      component shapes (old x fast-directs): " + ", ".join(shapes))


def _per_tier(x, n):
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if len(x) not in (1, n):
        raise ValueError(f"need a scalar or {n} values (one per tier, incl. the final all-remaining tier), got {len(x)}")
    return np.full(n, x[0]) if len(x) == 1 else x


def _print_tier_table(infos, cutname, max_rows=20):
    """Compact per-stage table: the tiers are grouped into at most ``max_rows`` rows of consecutive tiers. Linked pairs
    and matches are summed over a row's tiers; 'left' is what is still unmatched with multiplicity >= the row's last
    threshold after its last tier (large MAPs should all have matched); 'p99' is the largest p99 over the row of the
    accepted pairs' distance, next to the largest cut."""
    if not infos:
        return
    groups = np.array_split(np.arange(len(infos)), min(len(infos), max_rows))
    print(f"  {'tiers':>9s} {'mult >= (first..last)':>23s} {'linked':>9s} {'matches':>9s} "
          f"{'left old':>9s} {'left new':>9s} {'p99 / cut ' + cutname:>20s}")
    for gi in groups:
        g = [infos[i] for i in gi]
        tr = f"{gi[0]}" if len(gi) == 1 else f"{gi[0]}-{gi[-1]}"
        mr = f"{g[0]['mult_thresh']:g}" if len(g) == 1 else f"{g[0]['mult_thresh']:g}..{g[-1]['mult_thresh']:g}"
        p99 = np.nanmax([x['p99'] for x in g]) if np.isfinite([x['p99'] for x in g]).any() else np.nan
        print(f"  {tr:>9s} {mr:>23s} {sum(x['linked'] for x in g):9d} {sum(x['matches'] for x in g):9d} "
              f"{g[-1]['left_A']:9d} {g[-1]['left_B']:9d} {f'{p99:.3g} / {max(x['cut'] for x in g):.3g}':>20s}")


# ----------------------------------------------------------------------------------------------------------------
# the matching
# ----------------------------------------------------------------------------------------------------------------
def _accept(I, J, d, NA, NB, rule):
    """Pairs accepted from the linked pairs (I, J) at distances d: '1x1' = each MAP's only candidate (a 1x1
    component); 'mutual' = mutually-best pairs (see ``mutual_best``)."""
    if rule == '1x1':
        return (np.bincount(I, minlength=NA)[I] == 1) & (np.bincount(J, minlength=NB)[J] == 1)
    if rule == 'mutual':
        return mutual_best(I, J, d, NA, NB)
    raise ValueError("rule must be '1x1' or 'mutual'")


def auto_mult_thresh(ctx, n_bins=20, n_per_bin=1000):
    """Tier thresholds (largest first) from the multiplicity distribution: going down from the largest MAPs, each of
    the first ``n_bins - 1`` bins holds ~``n_per_bin`` MAPs per run (2 * n_per_bin of the two runs pooled). The final
    all-remaining bin, down to the smallest possible MAP, holds the rest, so more or larger bins shrink it. Ties in
    multiplicity can merge bins, and thresholds at or below ``mult_min`` are dropped. ``n_bins=np.inf`` keeps chunking
    in ``n_per_bin`` steps until every MAP is in a bin, so the final bin holds fewer than ``n_per_bin`` per run."""
    srt = np.sort(np.concatenate([ctx['LA']['mult'], ctx['LB']['mult']]))[::-1]
    n_chunks = len(srt) // (2 * n_per_bin) + 1 if np.isinf(n_bins) else n_bins     # n_bins = inf: chunk to the end
    idx = 2 * n_per_bin * np.arange(1, n_chunks) - 1
    thr = np.unique(srt[idx[idx < len(srt)]].astype(int))[::-1]
    return thr[thr > ctx['mult_min']]


def match(ctx, mult_thresh=None, n_bins=20, n_per_bin=1000, d_pass1=None, n_min_common=2, trajectory_walks=1, ellipse_walks=1, tier_rule='1x1',
          var_from='first', n_min_sigma=100, D_ellipse=4., mult_thresh_ellipse=None, n_min_common_ellipse=1,
          r_cand_ellipse=0.5, ellipse_rule='1x1', final_mutual=True, final_bin=None, verbose=True):
    """Trajectory walk(s) over the multiplicity tiers, then an ellipse walk over the tiers (see the module docstring).

    Tier k of a walk takes the still-unmatched MAPs with multiplicity >= the k-th threshold (a final tier takes every
    remaining MAP); matched MAPs leave the pool, so a MAP that was contested in an early tier can become a clean 1x1
    pair later.

    mult_thresh   : tier thresholds, largest first, for the trajectory walk. Default None: computed by
                    ``auto_mult_thresh(ctx, n_bins, n_per_bin)`` -- ``n_bins - 1`` bins of ~``n_per_bin`` MAPs per run
                    from the top down, plus the final all-remaining bin (more bins make that last bin smaller).
    d_pass1       : trajectory link/accept distance, RMS position separation [kpc/h]; a scalar or one value per tier
                    (one more than the number of thresholds, the last being the all-remaining tier). Large MAPs have
                    larger offsets between the runs, so the early tiers need a larger value. Default None: falls
                    geometrically from 180 (largest MAPs) to 50 (final tier) kpc/h, a first guess to tune.
    n_min_common  : the trajectory walk needs at least this many common epochs (current epoch included)
    trajectory_walks : how many times the whole trajectory walk is repeated before the ellipse walk
    ellipse_walks : how many times the ellipse walk over the tiers is repeated; 0 turns it off (``final_mutual`` is a
                    separate sweep and still runs unless set False). With no ellipse stage at all no MAP is tested for
                    candidates, so unmatched MAPs are all outcome 2 (none 'isolated').
    tier_rule     : '1x1' (default) accepts only 1x1 components in the trajectory tiers, 'mutual' mutually-best pairs
    n_bins        : number of tiers for the default thresholds; ``np.inf`` iterates in ``n_per_bin`` chunks until all
                    MAPs have been processed in the trajectory walk. The number of loops (tiers) is printed and
                    returned as ``n_loops``, and ``var_from`` is then overridden to the last tier.
    var_from      : which first-walk tier's matches set the ellipse variances: 'first' (tier 0, the largest MAPs),
                    'all' (every tier pooled) or a tier index. A choice with fewer than n_min_sigma matches falls
                    back to 'all'.
    D_ellipse     : ellipse distance cut (a typical matched pair has D ~ 2); scalar or one value per ellipse tier
    mult_thresh_ellipse : tier thresholds of the ellipse walk (default: the same as mult_thresh)
    n_min_common_ellipse, r_cand_ellipse : common-epoch minimum and current-position candidate radius [Mpc/h]
    ellipse_rule  : '1x1' (default) or 'mutual', for the ellipse tiers
    final_bin     : dict of settings for the final all-remaining bin only, which is huge and mostly small MAPs and so
                    may need different treatment. Keys (all optional, default the general value): 'd_pass1',
                    'n_min_common', 'tier_rule' (trajectory walk); 'D_ellipse', 'n_min_common_ellipse',
                    'ellipse_rule' (ellipse walk); 'skip_trajectory' / 'skip_ellipse' (True skips that stage's
                    final bin, e.g. leave the catch-all bin to the ellipse walk alone).
    final_mutual  : after the ellipse tiers, one more ellipse sweep over everything left that accepts mutually-best
                    pairs (the only place merely-mutually-best pairs are accepted, unless a rule above says so)

    Returns a dict: ``outA``/``outB`` per-MAP outcome (indices into OUTCOME_NAMES), ``tier_A``/``tier_B`` the id of the
    stage-tier in which each MAP was matched (-1 if unmatched; names in ``tier_labels``), the matched pairs ``mp_ia``,
    ``mp_ib``, ``mp_pass`` (0 = trajectory, 1 = ellipse), ``mp_tier`` and their residuals (fast-directs - old at the
    current epoch) ``mp_dx``, ``mp_dv``, ``mp_dmf``, plus ``tiers`` (per-tier summaries), ``var`` (the ellipse
    variances) and the parameters."""
    NA, NB, L, nl = ctx['NA'], ctx['NB'], ctx['L'], ctx['n_lookback']
    LA, LB = ctx['LA'], ctx['LB']
    if mult_thresh is None:
        mult_thresh = auto_mult_thresh(ctx, n_bins, n_per_bin)
    thr = list(mult_thresh) + [ctx['mult_min']]
    if np.isinf(n_bins):
        var_from = len(thr) - 1                        # the last bin
        if verbose:
            print(f"n_bins = inf: {len(thr)} loops of ~{n_per_bin} MAPs per run; ellipse variances from the last bin")
    thr_e = (list(mult_thresh) if mult_thresh_ellipse is None else list(mult_thresh_ellipse)) + [ctx['mult_min']]
    nt, nte = len(thr), len(thr_e)
    if d_pass1 is None:
        d_pass1 = np.geomspace(180., 50., nt)
    if verbose:
        nA_last = np.sum((LA['mult'] < thr[-2]) if nt > 1 else (LA['mult'] > 0))
        nB_last = np.sum((LB['mult'] < thr[-2]) if nt > 1 else (LB['mult'] > 0))
        print(f"{nt - 1} multiplicity thresholds {thr[0]:g} ... {thr[-2] if nt > 1 else '-'}; the final all-remaining "
              f"bin (mult < {thr[-2] if nt > 1 else '-'}) holds {nA_last} old / {nB_last} fast-directs MAPs "
              f"({nA_last / NA:.1%} / {nB_last / NB:.1%})")
    d1s, D2s = _per_tier(d_pass1, nt), _per_tier(D_ellipse, nte)
    fb = dict(final_bin or {})
    def rule_str(rule, key, skip):
        # the rule as used, including a different rule for the final all-remaining bin (final_bin) and whether it is skipped
        out = repr(rule)
        if skip:
            return out + ', final bin skipped'
        if fb.get(key, rule) != rule:
            out += f", final bin {fb[key]!r}"
        return out
    if 'd_pass1' in fb:
        d1s[-1] = fb['d_pass1']
    if 'D_ellipse' in fb:
        D2s[-1] = fb['D_ellipse']
    matchedA, matchedB = np.zeros(NA, bool), np.zeros(NB, bool)
    tierA, tierB = np.full(NA, -1), np.full(NB, -1)
    hasA, hasB = np.zeros(NA, bool), np.zeros(NB, bool)        # MAP had an ellipse candidate with D <= the cut
    pairs, tiers, tvar, tier_labels = [], [], [], []

    def record(a, b, kind, label):
        tid = len(tier_labels);  tier_labels.append(label)
        matchedA[a], matchedB[b] = True, True
        tierA[a], tierB[b] = tid, tid
        pairs.append((a, b, np.full(len(a), kind, np.int8), np.full(len(a), tid)))

    # ---- trajectory walk(s) over the multiplicity tiers, largest MAPs first
    for w in range(trajectory_walks):
        for k in range(nt):
            last = k == nt - 1                       # the final all-remaining bin may be treated differently
            ncom = fb.get('n_min_common', n_min_common) if last else n_min_common
            rule = fb.get('tier_rule', tier_rule) if last else tier_rule
            if last and fb.get('skip_trajectory'):
                if w == 0:
                    tvar.append(dict(n=0, x=np.nan, v=np.nan, m=np.nan, p=np.nan))
                if verbose:
                    print(f"\ntrajectory final bin (mult >= {thr[k]:g}) skipped")
                continue
            eA = np.flatnonzero(~matchedA & (LA['mult'] >= thr[k]))
            eB = np.flatnonzero(~matchedB & (LB['mult'] >= thr[k]))
            wlab = f'walk {w + 1} ' if trajectory_walks > 1 else ''
            info = dict(stage='trajectory', walk=w, tier=k, mult_thresh=thr[k], nA=len(eA), nB=len(eB), cut=d1s[k])
            r_cand1 = d1s[k] * np.sqrt(nl + 1) / 1e3  # RMS over n <= nl+1 epochs >= |dx_now| / sqrt(n): misses nothing
            I, J = _candidates(ctx, eA, eB, r_cand1)
            n1, x1, v1, m1 = traj_features(ctx, I, J)
            g = (n1 >= ncom) & (x1 <= d1s[k])
            I, J, n1, x1, v1, m1 = (a[g] for a in (I, J, n1, x1, v1, m1))
            acc = _accept(I, J, x1, NA, NB, rule)
            a1, b1 = I[acc], J[acc]
            record(a1, b1, 0, f'trajectory {wlab}tier {k}: mult >= {thr[k]:g}')
            info['linked'], info['matches'] = int(g.sum()), len(a1)
            info['p99'] = float(np.percentile(x1[acc], 99)) if len(a1) else np.nan
            # large MAPs should always have a counterpart: what is still unmatched above this tier's threshold
            info['left_A'] = int((~matchedA & (LA['mult'] >= thr[k])).sum())
            info['left_B'] = int((~matchedB & (LB['mult'] >= thr[k])).sum())
            tiers.append(info)
            if w == 0:                                          # the ellipse variances come from the first walk
                if len(a1):
                    p1 = pid_separation(ctx, a1, b1)
                    tvar.append(dict(n=len(a1), x=np.mean(x1[acc].astype(np.float64)**2),
                                     v=np.mean(v1[acc].astype(np.float64)**2),
                                     m=np.mean(m1[acc].astype(np.float64)**2), p=p1.mean()))
                else:
                    tvar.append(dict(n=0, x=np.nan, v=np.nan, m=np.nan, p=np.nan))
        if verbose:
            wi = [t for t in tiers if t['stage'] == 'trajectory' and t['walk'] == w]
            print(f"\ntrajectory walk{f' {w + 1}' if trajectory_walks > 1 else ''}: {len(wi)} tiers, "
                  f"{sum(t['matches'] for t in wi)} matches (rule {rule_str(tier_rule, 'tier_rule', fb.get('skip_trajectory'))}; "
                  f">= {n_min_common} common epochs); RMS |dx| in kpc/h")
            _print_tier_table(wi, 'kpc/h')

    # ---- ellipse variances from the first walk's matches in the chosen tier(s)
    ns = np.array([t['n'] for t in tvar])
    def pooled():
        w_ = ns / ns.sum()
        return {kk: float(np.nansum([wi * t[kk] for wi, t in zip(w_, tvar) if t['n']])) for kk in 'xvmp'}
    if var_from == 'all':
        var, src = pooled(), 'all tiers pooled'
    else:
        kt = 0 if var_from == 'first' else int(var_from)
        if ns[kt] >= n_min_sigma:
            var, src = {kk: tvar[kt][kk] for kk in 'xvmp'}, f'tier {kt}'
        else:
            var, src = pooled(), f'all tiers pooled (tier {kt} has only {ns[kt]} matches)'
    var['n'] = int(ns.sum()) if src.startswith('all') else int(ns[kt])
    if verbose:
        print(f"\nellipse variances from the trajectory matches of {src}: sig_x {np.sqrt(var['x']):.3g} kpc/h, "
              f"sig_v {np.sqrt(var['v']):.3g} km/s, sig_m {np.sqrt(var['m']):.3g} dex, mean PID separation "
              f"{var['p']:.3g} cells")

    # ---- ellipse walk over the multiplicity tiers (then, optionally, one mutually-best sweep of what is left)
    def ellipse_pass(eA, eB, cut, rule, label, info, n_min):
        I, J = _candidates(ctx, eA, eB, r_cand_ellipse)
        n2, x2, v2, m2 = traj_features(ctx, I, J)
        D = np.sqrt(x2.astype(np.float64)**2 / var['x'] + v2.astype(np.float64)**2 / var['v']
                    + m2.astype(np.float64)**2 / var['m'] + (pid_separation(ctx, I, J) / var['p'])**2)
        g = (n2 >= n_min) & (D <= cut)
        I, J, D = I[g], J[g], D[g]
        hasA[I], hasB[J] = True, True
        acc = _accept(I, J, D, NA, NB, rule)
        record(I[acc], J[acc], 1, label)
        info.update(linked=int(g.sum()), matches=int(acc.sum()), cut=cut,
                    p99=float(np.percentile(D[acc], 99)) if acc.any() else np.nan)
        return info

    for k in (k for _ in range(ellipse_walks) for k in range(nte)):
        last = k == nte - 1                          # the final all-remaining bin may be treated differently
        if last and fb.get('skip_ellipse'):
            if verbose:
                print(f"\nellipse final bin (mult >= {thr_e[k]:g}) skipped")
            continue
        eA = np.flatnonzero(~matchedA & (LA['mult'] >= thr_e[k]))
        eB = np.flatnonzero(~matchedB & (LB['mult'] >= thr_e[k]))
        info = ellipse_pass(eA, eB, D2s[k], fb.get('ellipse_rule', ellipse_rule) if last else ellipse_rule,
                            f'ellipse tier {k}: mult >= {thr_e[k]:g}',
                            dict(stage='ellipse', tier=k, mult_thresh=thr_e[k], nA=len(eA), nB=len(eB)),
                            fb.get('n_min_common_ellipse', n_min_common_ellipse) if last else n_min_common_ellipse)
        info['left_A'] = int((~matchedA & (LA['mult'] >= thr_e[k])).sum())
        info['left_B'] = int((~matchedB & (LB['mult'] >= thr_e[k])).sum())
        tiers.append(info)
    if verbose and ellipse_walks:
        ei = [t for t in tiers if t['stage'] == 'ellipse']
        print(f"\nellipse walk: {len(ei)} tiers, {sum(t['matches'] for t in ei)} matches (rule {rule_str(ellipse_rule, 'ellipse_rule', fb.get('skip_ellipse'))}); "
              f"ellipse distance D")
        _print_tier_table(ei, 'D')
    if final_mutual:
        eA, eB = np.flatnonzero(~matchedA), np.flatnonzero(~matchedB)
        tiers.append(ellipse_pass(eA, eB, D2s[-1], 'mutual', 'ellipse, mutually-best sweep',
                                  dict(stage='ellipse sweep', tier=nte, mult_thresh=ctx['mult_min'], nA=len(eA),
                                       nB=len(eB)), n_min_common_ellipse))
        if verbose:
            t = tiers[-1]
            print(f"ellipse sweep (mutually-best) over {t['nA']} old / {t['nB']} fast-directs unmatched MAPs: "
                  f"{t['matches']} matches")

    mp_ia, mp_ib, mp_pass, mp_tier = (np.concatenate(c) for c in zip(*pairs))
    outA, outB = np.full(NA, 2), np.full(NB, 2)
    if ellipse_walks or final_mutual:                # isolated = no ellipse candidate, so needs an ellipse stage
        outA[~hasA & ~matchedA] = 3;  outB[~hasB & ~matchedB] = 3
    outA[mp_ia] = mp_pass
    outB[mp_ib] = mp_pass
    dx = LB['pos'][mp_ib] - LA['pos'][mp_ia]
    dx -= L * np.round(dx / L)
    res = dict(outA=outA, outB=outB, tier_A=tierA, tier_B=tierB, tier_labels=tier_labels, mp_ia=mp_ia, mp_ib=mp_ib,
               mp_pass=mp_pass, mp_tier=mp_tier, mp_dx=dx, mp_dv=LB['vel'][mp_ib] - LA['vel'][mp_ia],
               mp_dmf=(LB['mult'][mp_ib] - LA['mult'][mp_ia]) / LA['mult'][mp_ia],
               tiers=tiers, var=var, mult_thresh=thr, n_loops=nt,
               params=dict(d_pass1=d_pass1, n_min_common=n_min_common, trajectory_walks=trajectory_walks, ellipse_walks=ellipse_walks,
                           tier_rule=tier_rule, var_from=var_from, D_ellipse=D_ellipse,
                           mult_thresh_ellipse=thr_e[:-1], n_min_common_ellipse=n_min_common_ellipse,
                           r_cand_ellipse=r_cand_ellipse, ellipse_rule=ellipse_rule, final_mutual=final_mutual))
    if verbose:
        print("\n" + "\n".join(f"{tag}: " + ", ".join(f"{n} {np.mean(o == i):.3%}" for i, n in enumerate(OUTCOME_NAMES))
                               for tag, o in (('old', outA), ('fast-directs', outB))))
        nuA, nuB = (outA >= 2).sum(), (outB >= 2).sum()
        print(f"unmatched: old {nuA}, fast-directs {nuB}; difference {nuA - nuB} = N_old - N_fast-directs = {NA - NB}")
    return res


# ----------------------------------------------------------------------------------------------------------------
# plots (each returns its figure)
# ----------------------------------------------------------------------------------------------------------------
def _legend(ax, max_entries=12, **kw):
    """Legend that stays readable with many populations (e.g. one per tier): above ``max_entries`` only the first
    entry ('all matched'), the first and last populations are listed, and a middle entry says how many are omitted."""
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) > max_entries:
        keep = [0, 1, len(labels) - 1]
        labels = [labels[0], labels[1], f'... ({len(labels) - 3} more, see colour scale) ...', labels[-1]]
        handles = [handles[0], handles[1], plt.Line2D([], [], color='none'), handles[-1]]
    ax.legend(handles, labels, **kw)


def _populations(res, by):
    """(label, mask over matched pairs, colour) for 'all matched' plus the sub-populations: by='pass' (trajectory
    walk / ellipse walk) or by='tier' (one per stage-tier)."""
    pops = [('all matched', np.ones(len(res['mp_pass']), bool), 'k')]
    if by == 'pass':
        pops += [('trajectory walk', res['mp_pass'] == 0, 'C0'), ('ellipse walk', res['mp_pass'] == 1, 'C1')]
    else:
        labels = res['tier_labels']
        cols = plt.cm.viridis(np.linspace(0, 0.95, len(labels)))
        pops += [(lab, res['mp_tier'] == i, to_hex(cols[i])) for i, lab in enumerate(labels)]
    return pops


def _panels(ctx):
    """The three properties the figures are binned in, as ``[(axis label, [old values, fast-directs values])]`` and the
    matching log-spaced bin edges (multiplicity from the smallest possible MAP; density and sigma_v spanning the
    0.1-99.9 percentiles of the pooled positive values)."""
    LA, LB, mult_min = ctx['LA'], ctx['LB'], ctx['mult_min']
    panels = [('multiplicity', [LA['mult'], LB['mult']]),
              (r'environmental density $\rho/\bar\rho$', [LA['dens'], LB['dens']]),
              (r'environmental $\sigma_v$ [km/s]', [LA['sigv'], LB['sigv']])]
    mb = np.unique(np.logspace(np.log10(mult_min), np.log10(max(LA['mult'].max(), LB['mult'].max())), 25).astype(int))
    pbins = [np.r_[mb, mb[-1] + 1]]
    for _, vals in panels[1:]:
        pooled = np.concatenate([v[np.isfinite(v) & (v > 0)] for v in vals])
        pbins.append(np.logspace(*np.log10(np.percentile(pooled, [0.1, 99.9])), 26))
    return panels, pbins


def plot_match_rate(ctx, res):
    """Figure: fraction of MAPs matched (solid; matched in the trajectory walk only, dashed) in bins of multiplicity,
    environmental density and environmental velocity dispersion, old = C0 and fast-directs = C3. The rest of each
    bin's MAPs are unmatched. Returns the figure."""
    panels, pbins = _panels(ctx)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (xlab, vals), bins in zip(axs, panels, pbins):
        for (tag, out, col), v in zip((('old', res['outA'], 'C0'), ('fast-directs', res['outB'], 'C3')), vals):
            good = np.isfinite(v) & (v > 0)
            idx = np.digitize(v, bins) - 1
            good &= (idx >= 0) & (idx < len(bins) - 1)
            n_tot = np.bincount(idx[good], minlength=len(bins))
            n_mat = np.bincount(idx[good & (out <= 1)], minlength=len(bins))
            n_trj = np.bincount(idx[good & (out == 0)], minlength=len(bins))
            ok = n_tot > 0
            ax.plot(bins[ok], n_mat[ok] / n_tot[ok], color=col, label=f'{tag}: matched')
            ax.plot(bins[ok], n_trj[ok] / n_tot[ok], color=col, ls='--', label=f'{tag}: matched in trajectory walk')
        ax.set_xlim(bins[0], None)
        ax.set_xscale('log'); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
        ax.set_xlabel(xlab); ax.set_ylabel('fraction of MAPs in bin')
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_unmatched(ctx, res, by='pass'):
    """Figure 1: number of unmatched MAPs per bin (solid; isolated only dashed) vs multiplicity, environmental density and
    environmental velocity dispersion, old = C0 and fast-directs = C3. Figure 1b: the same three properties for the
    matched MAPs (old-run values), one histogram (counts per bin) per population. Returns (fig1, fig1b)."""
    NA, mult_min = ctx['NA'], ctx['mult_min']
    outA, outB = res['outA'], res['outB']
    panels, pbins = _panels(ctx)

    fig1, axs = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (xlab, vals), bins in zip(axs, panels, pbins):
        for (tag, out, col), v in zip((('old', outA, 'C0'), ('fast-directs', outB, 'C3')), vals):
            good = np.isfinite(v) & (v > 0)
            idx = np.digitize(v, bins) - 1
            good &= (idx >= 0) & (idx < len(bins) - 1)
            n_tot = np.bincount(idx[good], minlength=len(bins))
            n_unm = np.bincount(idx[good & (out >= 2)], minlength=len(bins))
            n_iso = np.bincount(idx[good & (out == 3)], minlength=len(bins))
            ok = n_tot > 0
            ax.plot(bins[ok], n_unm[ok], color=col, label=f'{tag}: unmatched')
            ax.plot(bins[ok], n_iso[ok], color=col, ls='--', label=f'{tag}: isolated')
        ax.set_xlim(bins[0], None)   # multiplicity axis starts at the smallest possible multiplicity
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(xlab); ax.set_ylabel('MAPs per bin')
    axs[0].legend(fontsize=8)
    fig1.tight_layout()

    pops = _populations(res, by)
    fig1b, axs = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (xlab, vals), bins in zip(axs, panels, pbins):
        v = vals[0]
        for name, sel, col in pops:
            mA = np.zeros(NA, bool);  mA[res['mp_ia'][sel]] = True       # old MAPs in this population
            s_ = mA & np.isfinite(v) & (v > 0)
            if s_.any():
                ax.hist(v[s_], bins=bins, histtype='step', color=col, lw=1.6 if col == 'k' else 1,
                        label=f'{name} ({mA.sum() / NA:.2%})')
        ax.set_xlim(bins[0], None)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(xlab); ax.set_ylabel('matched MAPs per bin')
    _legend(axs[0], fontsize=8, title=f'fraction of eligible old MAPs (mult >= {mult_min})', title_fontsize=8)
    fig1b.tight_layout()
    return fig1, fig1b


def _log_bins(r, q, n_lin=3, nbins=30):
    """Bin edges for a non-negative magnitude on a log axis that still has a zero bin: one linear bin of width ``q`` (the
    lattice step, so the first bins are not aliased with the quantization) per lattice step up to ``n_lin`` steps,
    starting with [0, q/2) for exact zeros, then ``nbins`` log-spaced bins up to the maximum. Returns (edges, linthresh),
    linthresh being where a symlog axis switches from linear to log."""
    top = (n_lin + 0.5) * q
    edges = np.r_[0., (np.arange(n_lin + 1) + 0.5) * q]
    if r.max() > top:
        edges = np.r_[edges, np.logspace(np.log10(top), np.log10(r.max() * 1.001), nbins)[1:]]
    return edges, top


def plot_residuals(ctx, res, by='pass', nbins=30):
    """Figure 2: residuals (fast-directs - old, current epoch) of the matched pairs, total over all matched pairs
    (black) and per population. Row 1 histograms, row 2 CDFs from the same histograms. All three x-axes are
    logarithmic. Exact matches cannot sit on a true log axis, so each magnitude has a zero bin below the smallest
    nonzero value on a symlog axis (linear only below ``n_lin`` lattice steps; positions lie on a 1/128 Mpc/h lattice
    and velocities have a 2 km/s quantum, so those first bins are one lattice step wide), then log-spaced bins."""
    NA, mult_min, z_cmp = ctx['NA'], ctx['mult_min'], ctx['z_cmp']
    q_kpc, q_kms = 1e3 / 128, 2.
    pops = _populations(res, by)
    rx = np.linalg.norm(res['mp_dx'], axis=1) * 1e3
    rv = np.linalg.norm(res['mp_dv'], axis=1)
    rn = np.abs(res['mp_dmf'])
    rn_min = rn[rn > 0].min()
    bx, lx = _log_bins(rx, q_kpc, nbins=nbins)
    bv, lv = _log_bins(rv, q_kms, nbins=nbins)
    quant = [(rx, r'$|\Delta \mathbf{x}|$ [kpc/$h$]', bx, lx),
             (rv, r'$|\Delta \mathbf{v}|$ [km/s]', bv, lv),
             (rn, r'$|\Delta N| / N_{\rm old}$', np.r_[0., np.logspace(np.log10(rn_min), np.log10(rn.max()), nbins)],
              rn_min)]
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.4))
    for col, (r, lab, rbins, lth) in enumerate(quant):
        for name, sel, c in pops:
            if not sel.any():
                continue
            h, hb, _ = axs[0, col].hist(r[sel], bins=rbins, density=True, histtype='step', color=c,
                                        lw=1.6 if c == 'k' else 1, label=f'{name} ({sel.sum() / NA:.2%})')
            axs[1, col].step(hb[:-1], np.cumsum(h * np.diff(hb)), where='pre', color=c)
        axs[0, col].set_yscale('log')
        axs[0, col].set_ylabel('density');  axs[1, col].set_ylabel('CDF')
        axs[1, col].set_ylim(0, 1.02);  axs[1, col].grid(alpha=0.3)
        for ax in axs[:, col]:
            ax.set_xlabel(lab)
            if lth is not None:
                ax.set_xscale('symlog', linthresh=lth, linscale=0.3)
            ax.set_xlim(left=0)   # magnitudes: axis margins must not extend below zero
    _legend(axs[0, 0], fontsize=8, title=f'fraction of eligible old MAPs (mult >= {mult_min})', title_fontsize=8)
    fig.suptitle(f'z={z_cmp}, matched pairs with mult >= {mult_min}')
    fig.tight_layout()
    return fig


def _signed_row(axs, res, pops, r, q, unit, symbol, clip_pct, max_bins=100):
    """One row of the signed residual figure: ``r`` is (n_pairs, 3) in ``unit``, on a lattice of step ``q``; histograms
    of each axis into ``axs``. Returns the fraction of pairs beyond the plotted range."""
    lim = max(q, np.percentile(np.abs(r), clip_pct))
    w = q * max(1, int(np.ceil(2 * lim / q / max_bins)))   # a whole number of lattice steps per bin, at most ~max_bins bins
    k = int(np.ceil(lim / w - 0.5))
    bins = (np.arange(-k, k + 2) - 0.5) * w
    for a, ax in enumerate(axs):
        for name, sel, c in pops:
            if not sel.any():
                continue
            d = r[sel, a]
            sem = d.std() / np.sqrt(len(d))
            ax.hist(d, bins=bins, histtype='step', color=c, lw=1.6 if c == 'k' else 1,
                    label=f'{name}: mean {d.mean():+.2f} $\\pm$ {sem:.2f}, median {np.median(d):+.2f}')
        ax.axvline(0, color='gray', ls=':')
        ax.set_yscale('log')
        ax.set_xlabel(rf'$\Delta {symbol}_{"xyz"[a]}$ (fast-directs $-$ old) [{unit}]')
        ax.grid(alpha=0.3)
    axs[0].set_ylabel('matched pairs')
    _legend(axs[0], fontsize=7)
    return (np.abs(r) > bins[-1]).any(axis=1).mean()


def plot_signed_residuals(ctx, res, by='pass', clip_pct=99.9):
    """Figure 2b: signed per-axis residuals (fast-directs - old, current epoch) of the matched pairs, to look for a
    systematic displacement. Returns two figures, ``(positions [kpc/h, periodically wrapped], velocities [km/s])``, each with
    one panel per axis. Each panel overlays 'all matched' (black) and the populations; the legend gives each population's
    mean +- standard error and median, and the dotted line marks zero. Bins are a whole number of lattice steps wide
    (positions 1/128 Mpc/h, velocities 2 km/s; one step unless that would exceed ~100 bins), centred on lattice values,
    spanning +-``clip_pct`` percentile of the absolute residual over the three axes. The percentage of pairs beyond the
    plotted range (in any axis) is given in the title; y is log."""
    mult_min, z_cmp = ctx['mult_min'], ctx['z_cmp']
    pops = _populations(res, by)
    figs = []
    for r, q, unit, symbol, name in [(res['mp_dx'] * 1e3, 1e3 / 128, r'kpc/$h$', 'x', 'position'),
                                     (res['mp_dv'], 2., 'km/s', 'v', 'velocity')]:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
        c = _signed_row(axs, res, pops, r, q, unit, symbol, clip_pct)
        fig.suptitle(f'z={z_cmp}, signed {name} residuals of matched pairs with mult >= {mult_min}\n'
                     f'{c:.2%} of pairs beyond the plotted range', fontsize=11)
        fig.tight_layout()
        figs.append(fig)
    return tuple(figs)


def plot_unmatched_xi(ctx, res, rmin=0.04, rmax=18., nbins=20, nthreads=16):
    """Figure 3: real-space auto-correlation functions of the MAP samples of each run at the comparison epoch, with no
    multiplicity selection beyond the eligibility cut (mult >= mult_min): all MAPs, matched, unmatched (outcome >= 2) and
    isolated (outcome 3). Row 1 holds the unmatched and isolated samples, row 2 all and matched (their r^2 xi differ too much to share an
    axis). Left: r^2 xi(r) [(Mpc/h)^2], old solid and fast-directs dashed, one colour per sample, Poisson error bands
    (from the pair counts) on the old run. Right: xi_fast-directs / xi_old per sample. Log-spaced bins from
    ``rmin`` to ``rmax`` [Mpc/h]. Unmatched MAPs are mostly small, so differences from the matched sample mix a match
    failure with the multiplicity dependence of clustering. Returns ``(fig, data)``, data[tag][sample] = (xi, poisson)."""
    L, mult_min, z_cmp = ctx['L'], ctx['mult_min'], ctx['z_cmp']
    edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
    r = edges[:-1] + np.diff(edges) / 2.
    samples = [('all', lambda o: np.ones(len(o), bool), 'k'),
               ('matched', lambda o: o < 2, 'C2'),
               ('unmatched', lambda o: o >= 2, 'C1'),
               ('isolated', lambda o: o == 3, 'C3')]
    data = {}
    for tag, X, out in (('old', ctx['LA'], res['outA']), ('fast-directs', ctx['LB'], res['outB'])):
        data[tag] = {}
        for name, f, _ in samples:
            m = f(out)
            tp = get_xi(X['pos'][m], edges, L * u.Mpc, nthreads=nthreads)
            xi = tp.get_corr(return_sep=False)
            data[tag][name] = (xi, np.sqrt(tp.D1D2.wcounts) / tp.R1R2.wcounts)
            print(f"{tag}, {name}: {m.sum()} MAPs")
    # the unmatched / isolated populations are far more strongly clustered than all / matched MAPs: separate rows
    rows = [('unmatched', 'isolated'), ('all', 'matched')]
    colour = {name: c for name, _, c in samples}
    select = {name: f for name, f, _ in samples}
    fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for row, names in zip(axs, rows):
        for name in names:
            c = colour[name]
            (xo, po), (xf, _) = data['old'][name], data['fast-directs'][name]
            n = select[name](res['outA']).sum(), select[name](res['outB']).sum()
            row[0].plot(r, r**2 * xo, color=c, label=f'{name} (old {n[0]}, fd {n[1]})')
            row[0].fill_between(r, r**2 * (xo - po), r**2 * (xo + po), color=c, alpha=0.15, lw=0)
            row[0].plot(r, r**2 * xf, color=c, ls='--')
            row[1].plot(r, xf / xo, color=c)
        row[0].set_ylabel(r'$r^2 \xi(r)$ [(Mpc/$h$)$^2$]')
        row[0].legend(fontsize=8, title='solid: old, dashed: fast-directs', title_fontsize=8)
        row[1].axhline(1, color='gray', ls=':')
        row[1].set_ylabel(r'$\xi_{\rm fast-directs} / \xi_{\rm old}$')
        for ax in row:
            ax.set_xscale('log')
            ax.grid(alpha=0.3)
    for ax in axs[1]:
        ax.set_xlabel(r'$r$ [Mpc/$h$]')
    fig.suptitle(f'z={z_cmp}, MAP correlation functions (mult >= {mult_min})')
    fig.tight_layout()
    return fig, data


def _xi_cross(posA, posB, edges, L, nthreads):
    """Cross-correlation of two position samples: ``(xi, poisson)`` on ``edges`` (same convention as the auto version)."""
    tp = get_xi(posA, edges, L * u.Mpc, pos2=posB, nthreads=nthreads)
    return tp.get_corr(return_sep=False), np.sqrt(tp.D1D2.wcounts) / tp.R1R2.wcounts


def plot_unmatched_cross_xi(ctx, res, rmin=0.04, rmax=18., nbins=20, nthreads=16):
    """Figure 4: cross-correlation functions of the unmatched / isolated MAPs. Returns ``(fig_matched, fig_runs, data)``.

    ``fig_matched``: each run's isolated (solid) and unmatched (dashed) MAPs against that run's matched MAPs (old = C0,
    fast-directs = C3); left r^2 xi, right the fast-directs / old ratio. Unmatched MAPs that sit preferentially near matched
    MAPs (more than their auto-clustering and multiplicity alone would give) point to a failure that depends on the
    environment, e.g. crowding.

    ``fig_runs``: cross-correlation of the old run's isolated MAPs with the fast-directs run's isolated MAPs (and the same
    for unmatched). The two runs share the initial conditions, so positions are directly comparable: a positive
    cross-correlation says the same places produce unmatched MAPs in both runs (an environmental or systematic cause), a
    small-scale spike says the matcher missed near-matches, zero says the failures are uncorrelated between runs. Left
    r^2 xi, right xi on log axes (positive values only)."""
    L, mult_min, z_cmp = ctx['L'], ctx['mult_min'], ctx['z_cmp']
    edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
    r = edges[:-1] + np.diff(edges) / 2.
    runs = {'old': (ctx['LA']['pos'], res['outA'], 'C0'), 'fast-directs': (ctx['LB']['pos'], res['outB'], 'C3')}
    pick = {'isolated': lambda o: o == 3, 'unmatched': lambda o: o >= 2, 'matched': lambda o: o < 2}
    data = {}
    for tag, (pos, out, _) in runs.items():
        for name in ('isolated', 'unmatched'):
            data[f'{name} x matched, {tag}'] = _xi_cross(pos[pick[name](out)], pos[pick['matched'](out)], edges, L, nthreads)
    pO, oO, _ = runs['old'];  pF, oF, _ = runs['fast-directs']
    for name in ('isolated', 'unmatched'):
        data[f'{name} x {name}, old x fast-directs'] = _xi_cross(pO[pick[name](oO)], pF[pick[name](oF)], edges, L, nthreads)

    fig_m, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for tag, (_, _, c) in runs.items():
        for name, ls in (('isolated', '-'), ('unmatched', '--')):
            xi, po = data[f'{name} x matched, {tag}']
            axs[0].plot(r, r**2 * xi, color=c, ls=ls, label=f'{tag}: {name} x matched')
            if ls == '-':
                axs[0].fill_between(r, r**2 * (xi - po), r**2 * (xi + po), color=c, alpha=0.15, lw=0)
    for name, ls in (('isolated', '-'), ('unmatched', '--')):
        axs[1].plot(r, data[f'{name} x matched, fast-directs'][0] / data[f'{name} x matched, old'][0], color='k', ls=ls,
                    label=name)
    axs[0].set_ylabel(r'$r^2 \xi_{\rm x\,matched}(r)$ [(Mpc/$h$)$^2$]')
    axs[0].legend(fontsize=8)
    axs[1].axhline(1, color='gray', ls=':')
    axs[1].set_ylabel(r'$\xi_{\rm fast-directs} / \xi_{\rm old}$')
    axs[1].legend(fontsize=8)
    for ax in axs:
        ax.set_xscale('log');  ax.set_xlabel(r'$r$ [Mpc/$h$]');  ax.grid(alpha=0.3)
    fig_m.suptitle(f'z={z_cmp}, cross-correlation with matched MAPs of the same run (mult >= {mult_min})')
    fig_m.tight_layout()

    fig_r, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, c in (('isolated', 'C3'), ('unmatched', 'C1')):
        xi, po = data[f'{name} x {name}, old x fast-directs']
        axs[0].plot(r, r**2 * xi, color=c, label=f'{name}(old) x {name}(fast-directs)')
        axs[0].fill_between(r, r**2 * (xi - po), r**2 * (xi + po), color=c, alpha=0.15, lw=0)
        axs[1].plot(r, np.where(xi > 0, xi, np.nan), color=c, label=name)
    axs[0].set_ylabel(r'$r^2 \xi_\times(r)$ [(Mpc/$h$)$^2$]');  axs[0].legend(fontsize=8)
    axs[1].set_yscale('log');  axs[1].set_ylabel(r'$\xi_\times(r)$ (positive values)')
    for ax in axs:
        ax.set_xscale('log');  ax.set_xlabel(r'$r$ [Mpc/$h$]');  ax.grid(alpha=0.3)
    fig_r.suptitle(f'z={z_cmp}, cross-correlation between the runs (mult >= {mult_min})')
    fig_r.tight_layout()
    return fig_m, fig_r, data


def plot_cell_position(ctx, res, cpd=None, bin_kpc=25.):
    """Figure 5: do unmatched / isolated MAPs depend on where they sit in the Abacus cell grid? For each axis, the fraction
    of MAPs that are unmatched (row 1) / isolated (row 2) vs the distance to the nearest cell face along that axis, from the
    face out to half a cell (cell = BoxSize / CPD) in ``bin_kpc`` bins, old = C0 and fast-directs = C3, with binomial error
    bands and the overall fraction dotted. Without a cell-position dependence the curves are flat; the legend gives each
    curve's chi^2/dof against a constant. ``cpd`` defaults to the context's (``prepare`` stores it), else pass the
    header's CPD. Returns the figure."""
    L, mult_min, z_cmp = ctx['L'], ctx['mult_min'], ctx['z_cmp']
    cpd = cpd or ctx.get('cpd')
    if not cpd:
        raise ValueError("pass cpd (the run's CPD): ctx has none")
    w = L / cpd * 1e3                                              # cell width [kpc/h]
    bins = np.arange(0., w / 2 + bin_kpc, bin_kpc)
    bins[-1] = w / 2
    mid = (bins[:-1] + bins[1:]) / 2
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.4), sharex=True)
    for tag, p, out, c in (('old', ctx['pA'], res['outA'], 'C0'), ('fast-directs', ctx['pB'], res['outB'], 'C3')):
        f = (p * 1e3) % w                                          # position within the cell along each axis
        d = np.minimum(f, w - f)
        for row, (label, sel) in enumerate((('unmatched', out >= 2), ('isolated', out == 3))):
            p0 = sel.mean()
            for a in range(3):
                n = np.histogram(d[:, a], bins)[0].astype(float)
                k = np.histogram(d[sel, a], bins)[0].astype(float)
                frac = k / np.maximum(n, 1)
                err = np.sqrt(frac * (1 - frac) / np.maximum(n, 1))
                chi2 = np.sum((k - n * p0)**2 / (n * p0 * (1 - p0)))
                ax = axs[row, a]
                ax.plot(mid, frac, color=c, label=rf'{tag}: $\chi^2$/dof = {chi2 / (len(n) - 1):.2f}')
                ax.fill_between(mid, frac - err, frac + err, color=c, alpha=0.15, lw=0)
                ax.axhline(p0, color=c, ls=':', lw=0.8)
    for a in range(3):
        axs[1, a].set_xlabel(rf'distance to nearest ${"xyz"[a]}$ face [kpc/$h$]')
        axs[0, a].set_title(f'{"xyz"[a]} axis')
        for row in range(2):
            axs[row, a].grid(alpha=0.3)
            axs[row, a].legend(fontsize=8)
            axs[row, a].set_xlim(0, w / 2)
    axs[0, 0].set_ylabel('unmatched fraction');  axs[1, 0].set_ylabel('isolated fraction')
    fig.suptitle(f'z={z_cmp}, MAPs vs position in the cell grid (cell = {w:.0f} kpc/$h$, mult >= {mult_min})')
    fig.tight_layout()
    return fig


def plot_residual_cell_position(ctx, res, cpd=None, bin_kpc=25.):
    """Figure 6: signed residuals (fast-directs - old, current epoch) of the matched pairs vs the old run's position across the
    whole Abacus cell (cell = BoxSize / CPD; the sign matters, so the full width is kept rather than folded onto the
    nearest face). Returns two figures, ``(positions [kpc/h], velocities [km/s])``. In each, one column per axis (the residual
    along the axis whose in-cell position is on the x-axis); row 1 the mean +- standard error per bin, row 2 the standard
    deviation. Cell faces are at 0 and the cell width. A bug tied to the cell grid shows as a mean that changes sign or
    steps across the cell or as a spread that grows toward the faces; the legend gives the chi^2/dof of the means against
    zero (the global offset is in ``plot_signed_residuals``) and against their own constant value. ``cpd`` defaults to the
    context's, else pass the header's CPD."""
    L, mult_min, z_cmp = ctx['L'], ctx['mult_min'], ctx['z_cmp']
    cpd = cpd or ctx.get('cpd')
    if not cpd:
        raise ValueError("pass cpd (the run's CPD): ctx has none")
    w = L / cpd * 1e3
    nb = max(1, int(round(w / bin_kpc)))
    bins = np.linspace(0., w, nb + 1)
    mid = (bins[:-1] + bins[1:]) / 2
    f = (ctx['pA'][res['mp_ia']] * 1e3) % w
    idx = np.minimum(np.digitize(f, bins) - 1, nb - 1)
    figs = []
    for r, unit, symbol, name in [(res['mp_dx'] * 1e3, r'kpc/$h$', 'x', 'position'), (res['mp_dv'], 'km/s', 'v', 'velocity')]:
        fig, axs = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
        for a in range(3):
            n = np.bincount(idx[:, a], minlength=nb).astype(float)
            m = np.bincount(idx[:, a], r[:, a], nb) / np.maximum(n, 1)
            sd = np.sqrt(np.maximum(np.bincount(idx[:, a], r[:, a]**2, nb) / np.maximum(n, 1) - m**2, 0))
            sem = sd / np.sqrt(np.maximum(n, 1))
            ok = n > 1
            mbar = np.sum(m[ok] / sem[ok]**2) / np.sum(1 / sem[ok]**2)
            chi0 = np.sum((m[ok] / sem[ok])**2) / ok.sum()
            chic = np.sum(((m[ok] - mbar) / sem[ok])**2) / (ok.sum() - 1)
            axs[0, a].errorbar(mid[ok], m[ok], sem[ok], color='k', marker='.', ls='-', lw=0.8,
                               label=rf'$\chi^2$/dof vs 0: {chi0:.2f}, vs constant: {chic:.2f}')
            axs[0, a].axhline(0, color='gray', ls=':')
            axs[1, a].plot(mid[ok], sd[ok], color='k', marker='.', lw=0.8)
            axs[0, a].set_title(f'{"xyz"[a]} axis')
            axs[0, a].set_ylabel(rf'mean $\Delta {symbol}_{"xyz"[a]}$ [{unit}]')
            axs[1, a].set_ylabel(rf'std $\Delta {symbol}_{"xyz"[a]}$ [{unit}]')
            axs[1, a].set_xlabel(rf'${"xyz"[a]}$ position in cell [kpc/$h$]')
            axs[0, a].legend(fontsize=8)
            for ax in axs[:, a]:
                ax.grid(alpha=0.3)
                ax.set_xlim(0, w)
        fig.suptitle(f'z={z_cmp}, matched-pair {name} residuals vs position in the cell grid (cell = {w:.0f} kpc/$h$, '
                     f'mult >= {mult_min}, {len(f)} pairs)')
        fig.tight_layout()
        figs.append(fig)
    return tuple(figs)


def plot_separation_multiplicity(ctx, res, nx=30, ny=40, normalize=True):
    """Figure 5: separation |dx| (fast-directs - old position at the current epoch, periodically wrapped) of the matched pairs vs
    the MAP's multiplicity, one panel per run (left the old run's multiplicity, right the fast-directs run's). The
    colour is, with ``normalize=True`` (default), the fraction of the matched pairs in each multiplicity bin that fall in
    each separation bin, i.e. every column is divided by the total number of matched pairs in that multiplicity bin
    (zero-separation pairs included), which takes out the shape of the multiplicity function and leaves where MAPs of
    each multiplicity sit in separation (columns sum to 1, less any zero-separation pairs); with ``normalize=False``
    the raw number of pairs per bin. Colour is on a log scale. The lines are the median (solid) and 16th / 84th percentiles
    (dashed) of the separation in multiplicity bins with at least 10 pairs. Both axes are logarithmic: pairs with exactly
    zero separation cannot be drawn and are left out of the colour map (their share is printed in the panel title) but
    are counted in the percentiles."""
    mult_min, z_cmp = ctx['mult_min'], ctx['z_cmp']
    sep = np.linalg.norm(res['mp_dx'], axis=1) * 1e3                 # kpc/h
    pos = sep > 0
    top = max(ctx['LA']['mult'][res['mp_ia']].max(), ctx['LB']['mult'][res['mp_ib']].max())
    xb = np.logspace(np.log10(mult_min), np.log10(top), nx + 1)
    yb = np.logspace(np.log10(sep[pos].min()), np.log10(sep.max()), ny + 1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (tag, mult) in zip(axs, (('old', ctx['LA']['mult'][res['mp_ia']]),
                                     ('fast-directs', ctx['LB']['mult'][res['mp_ib']]))):
        h = np.histogram2d(mult[pos], sep[pos], bins=[xb, yb])[0]
        if normalize:
            h = h / np.maximum(np.histogram(mult, xb)[0], 1)[:, None]      # one total multiplicity histogram (same xb), broadcast over all rows
        im = ax.pcolormesh(xb, yb, np.where(h > 0, h, np.nan).T, norm=LogNorm(), cmap='viridis')
        idx = np.digitize(mult, xb) - 1
        mid = np.sqrt(xb[:-1] * xb[1:])
        q = np.full((3, nx), np.nan)
        for i in range(nx):
            v = sep[idx == i]
            if len(v) >= 10:
                q[:, i] = np.percentile(v, [16, 50, 84])
        ax.plot(mid, q[1], color='w', lw=2.2);  ax.plot(mid, q[1], color='C1', lw=1.4, label='median')
        for k in (0, 2):
            ax.plot(mid, q[k], color='w', lw=2.2);  ax.plot(mid, q[k], color='C1', lw=1.2, ls='--')
        ax.plot([], [], color='C1', ls='--', label='16th / 84th percentile')
        ax.set_xscale('log');  ax.set_yscale('log')
        ax.set_xlabel(f'multiplicity ({tag} run)')
        ax.set_title(f'{tag}: {(~pos).mean():.2%} of pairs at zero separation (not drawn)', fontsize=9)
        ax.grid(alpha=0.3)
        fig.colorbar(im, ax=ax, label='fraction of multiplicity bin' if normalize else 'matched pairs per bin')
    axs[0].set_ylabel(r'separation $|\Delta \mathbf{x}|$ [kpc/$h$]')
    axs[0].legend(fontsize=8, loc='upper right')
    fig.suptitle(f'z={z_cmp}, matched pairs with mult >= {mult_min}')
    fig.tight_layout()
    return fig


def _choose_centres(ctx, res, run, population, min_mult, n, seed):
    """``n`` random MAPs of ``run`` in ``population`` ('matched', 'unmatched' or 'isolated') with multiplicity >= ``min_mult``:
    a list of ``(run, MAP index)``."""
    X, out = (ctx['LA'], res['outA']) if run == 'old' else (ctx['LB'], res['outB'])
    in_pop = {'matched': out < 2, 'unmatched': out >= 2, 'isolated': out == 3}
    if population not in in_pop:
        raise ValueError(f"population must be one of {list(in_pop)}, got {population!r}")
    cand = np.flatnonzero(in_pop[population] & ((X['mult'] >= min_mult) if min_mult is not None else True))
    if len(cand) == 0:
        raise ValueError(f'no {population} {run} MAP with multiplicity >= {min_mult}')
    return [(run, int(i)) for i in np.random.default_rng(seed).choice(cand, size=min(n, len(cand)), replace=False)]


def plot_unmatched_neighborhoods(ctx, res, box=2., min_mult=None, n=3, run='old', seed=None, label_mult=False, link=True,
                                 population='unmatched', slab=None, picks=None):
    """Figure 6: the neighbourhood of ``n`` randomly chosen MAPs of one population. Each row is one chosen MAP (a MAP of
    ``run`` = 'old' or 'fast-directs' in ``population`` = 'matched' (outcome < 2), 'unmatched' (outcome >= 2, the default)
    or 'isolated' (outcome 3), with multiplicity >= ``min_mult`` if given; it sits at the origin, drawn like any other MAP of its run) and shows the
    x-y, y-z and x-z projections of every MAP of BOTH runs inside the cube of side ``box`` [Mpc/h] centred on it
    (periodically wrapped; positions relative to the chosen MAP). The marker shows the run: open circle 'o' = old, 'x' =
    fast-directs. Unmatched MAPs are gray. Each matched pair has its own colour, shared by its two members; ``link`` also
    joins the members of a pair with a thin line (extending to a partner that lies outside the box, clipped at the panel
    edge); unmatched MAPs have no line. Colours repeat beyond 20 pairs in a box. Marker area grows with log10(multiplicity). The
    dotted circle (radius ``box``/2) is the largest circle inside the box, a stand-in for the sphere of that radius.
    ``label_mult`` writes each MAP's multiplicity next to it. ``seed`` fixes the random choice.
    ``slab`` (thickness in Mpc/h, default None) turns the projections into slices: each panel then shows only the MAPs within
    ``slab``/2 of the chosen MAP along the axis that is projected out (z for x-y, ...), still within ``box``/2 along the two
    plotted axes, so what appears close together in the page is also close along the line of sight. With ``slab=None``
    every panel shows the whole cube. A pair's line is drawn in a panel when at least one member appears there.
    ``picks`` (a list of ``(run, MAP index)``, as returned) reuses those central MAPs instead of choosing new ones; the
    selection arguments are then ignored.
    Returns ``(fig, picks)``, picks = [(run, MAP index)]."""
    L, mult_min, z_cmp = ctx['L'], ctx['mult_min'], ctx['z_cmp']
    LA, LB = ctx['LA'], ctx['LB']
    runs = {'old': (LA, res['outA'], 'gray', 'o'), 'fast-directs': (LB, res['outB'], 'gray', 'x')}
    partnerA = np.full(ctx['NA'], -1);  partnerA[res['mp_ia']] = res['mp_ib']      # old index -> fast-directs index
    partnerB = np.full(ctx['NB'], -1);  partnerB[res['mp_ib']] = res['mp_ia']
    pair_colours = plt.cm.tab20(np.arange(20))
    wrap = lambda d: d - L * np.round(d / L)
    picks = _choose_centres(ctx, res, run, population, min_mult, n, seed) if picks is None else list(picks)
    pairs = [('xy', 0, 1), ('yz', 1, 2), ('xz', 0, 2)]
    fig, axs = plt.subplots(len(picks), 3, figsize=(15, 5 * len(picks)), squeeze=False)
    h = box / 2
    hs = h if slab is None else slab / 2          # half-depth along the projected axis
    for row, (run_i, i) in zip(axs, picks):
        X = runs[run_i][0]
        centre = X['pos'][i]
        D = {tag: wrap(Y['pos'] - centre) for tag, (Y, o, c, mk) in runs.items()}
        panels = []                                # per panel: members of each run, with the pair label of the matched ones
        for name, a, b in pairs:
            c_ax = 3 - a - b                       # the projected-out axis
            members = []
            for tag, (Y, o, c, mk) in runs.items():
                d = D[tag]
                m = np.flatnonzero((np.abs(d[:, a]) <= h) & (np.abs(d[:, b]) <= h) & (np.abs(d[:, c_ax]) <= hs))
                unm = o[m] >= 2
                pair = (m if tag == 'old' else partnerB[m])         # a matched pair is labelled by its old-run index
                members.append((tag, c, mk, d[m], Y['mult'][m], unm, pair))
            panels.append(members)
        keys = sorted(set().union(*(set(mm[6][~mm[5]].tolist()) for members in panels for mm in members)))
        colour = {k: pair_colours[n_ % 20] for n_, k in enumerate(keys)}
        t = np.linspace(0, 2 * np.pi, 200)
        for ax, (name, a, b), members in zip(row, pairs, panels):
            if link:
                for k in sorted(set().union(*(set(mm[6][~mm[5]].tolist()) for mm in members))):
                    pa, pb = wrap(LA['pos'][k] - centre), wrap(LB['pos'][partnerA[k]] - centre)  # partner may be outside
                    ax.plot([pa[a], pb[a]], [pa[b], pb[b]], color=colour[k], lw=0.7, alpha=0.7, zorder=1)
            for tag, c, mk, d, mult, unm, pair in members:
                size = 12 + 40 * np.log10(np.maximum(mult, mult_min) / mult_min)
                mt = ~unm
                for sel, cols, z in ((unm, c, 2), (mt, [colour[k] for k in pair[mt]], 3)):
                    if not sel.any():
                        continue
                    if mk == 'o':   # open circles, so overlapping members stay visible
                        ax.scatter(d[sel, a], d[sel, b], s=size[sel], marker=mk, facecolors='none', edgecolors=cols, lw=1.2, zorder=z)
                    else:
                        ax.scatter(d[sel, a], d[sel, b], s=size[sel], marker=mk, c=cols, lw=1.2, zorder=z)
                if label_mult:
                    for (u_, v_), mm in zip(d[:, [a, b]], mult):
                        ax.annotate(f'{mm:.0f}', (u_, v_), fontsize=6, color=c, xytext=(3, 3), textcoords='offset points')
            ax.plot(h * np.cos(t), h * np.sin(t), 'k:', lw=1)
            ax.set_xlim(-h, h);  ax.set_ylim(-h, h);  ax.set_aspect('equal')
            ax.set_xlabel(rf'$\Delta {name[0]}$ [Mpc/$h$]');  ax.set_ylabel(rf'$\Delta {name[1]}$ [Mpc/$h$]')
            ax.grid(alpha=0.3)
            npair = len(set().union(*(set(mm[6][~mm[5]].tolist()) for mm in members)))
            depth = f'|$\\Delta {"xyz"[3 - a - b]}$| < {hs:g}' if slab is not None else 'whole cube'
            ax.set_title(f'{depth}: ' + ', '.join(f'{len(mm[3])} {mm[0]}' for mm in members) + f' MAPs, {npair} matched pairs',
                         fontsize=8)
        row[0].annotate(f'{run_i} MAP {i}: mult {X["mult"][i]:.0f}, at ({centre[0]:.2f}, {centre[1]:.2f}, {centre[2]:.2f})',
                        (0, 1.16), xycoords='axes fraction', fontsize=9, fontweight='bold')
    handles = [plt.Line2D([], [], color='k', marker='o', mfc='none', ls='none', label='old'),
               plt.Line2D([], [], color='k', marker='x', ls='none', label='fast-directs'),
               plt.Line2D([], [], color='C0', ls='-', label='matched: shared colour + line'),
               plt.Line2D([], [], color='gray', marker='s', ls='none', label='unmatched: gray')]
    fig.legend(handles=handles, loc='lower center', fontsize=8, ncol=4)
    fig.suptitle(f'z={z_cmp}, {population if len({r for r, _ in picks}) == 1 else "chosen"} {picks[0][0]} MAPs (mult >= {min_mult if min_mult is not None else mult_min}) and their '
                 f'surroundings, box {box:g} Mpc/$h$', fontsize=11)
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    return fig, picks


def _clip_segment(p0, p1, h):
    """Part of the segment p0 -> p1 inside the cube [-h, h]^3 (Liang-Barsky), or None."""
    d = p1 - p0
    t0, t1 = 0., 1.
    for k in range(3):
        if abs(d[k]) < 1e-12:
            if abs(p0[k]) > h:
                return None
            continue
        ta, tb = (-h - p0[k]) / d[k], (h - p0[k]) / d[k]
        t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
        if t0 > t1:
            return None
    return p0 + t0 * d, p0 + t1 * d


def plot_neighborhoods_3d(ctx, res, picks, box=2., label_mult=False, link=True, elev=25., azim=-60., rotate=45.):
    """Figure 7: three-dimensional views of the neighbourhoods drawn by ``plot_unmatched_neighborhoods``: one figure per chosen
    MAP in ``picks`` (the list of ``(run, MAP index)`` it returns, so the same MAPs), each a 3-D scatter of every MAP of both
    runs inside the whole cube of side ``box`` [Mpc/h] centred on the chosen MAP (no slab), relative to it. Same encoding: 'o'
    (open) old, 'x' fast-directs, gray unmatched, matched pairs share a colour and are joined by a line (clipped to the
    cube), marker area grows with log10(multiplicity); the faint wireframe sphere has radius ``box``/2. Each figure has two
    panels of the same MAPs: the view at elevation ``elev`` and azimuth ``azim`` [deg] (matplotlib's default is 25, -60) and the
    same view rotated by ``rotate`` degrees about z. In an interactive backend (e.g. ``%matplotlib widget`` with ipympl) both
    can be rotated with the mouse. Returns the list of figures."""
    L, mult_min, z_cmp = ctx['L'], ctx['mult_min'], ctx['z_cmp']
    LA, LB = ctx['LA'], ctx['LB']
    runs = {'old': (LA, res['outA'], 'o'), 'fast-directs': (LB, res['outB'], 'x')}
    partnerA = np.full(ctx['NA'], -1);  partnerA[res['mp_ia']] = res['mp_ib']
    partnerB = np.full(ctx['NB'], -1);  partnerB[res['mp_ib']] = res['mp_ia']
    pair_colours = plt.cm.tab20(np.arange(20))
    wrap = lambda d: d - L * np.round(d / L)
    h = box / 2
    u_, v_ = np.meshgrid(np.linspace(0, 2 * np.pi, 25), np.linspace(0, np.pi, 13))
    figs = []
    for run_i, i in picks:
        centre = runs[run_i][0]['pos'][i]
        members, keys = [], set()
        for tag, (Y, o, mk) in runs.items():
            d = wrap(Y['pos'] - centre)
            m = np.flatnonzero((np.abs(d) <= h).all(axis=1))
            unm = o[m] >= 2
            pair = m if tag == 'old' else partnerB[m]
            keys |= set(pair[~unm].tolist())
            members.append((tag, mk, d[m], Y['mult'][m], unm, pair))
        colour = {k: pair_colours[n_ % 20] for n_, k in enumerate(sorted(keys))}
        fig = plt.figure(figsize=(15, 7.6))
        for col, az in enumerate((azim, azim + rotate)):
            ax = fig.add_subplot(1, 2, col + 1, projection='3d')
            ax.plot_wireframe(h * np.cos(u_) * np.sin(v_), h * np.sin(u_) * np.sin(v_), h * np.cos(v_), color='gray', lw=0.3,
                              alpha=0.4)
            if link:
                for k in sorted(keys):
                    seg = _clip_segment(wrap(LA['pos'][k] - centre), wrap(LB['pos'][partnerA[k]] - centre), h)
                    if seg is not None:
                        ax.plot(*zip(*seg), color=colour[k], lw=0.8, alpha=0.7)
            for tag, mk, d, mult, unm, pair in members:
                size = 12 + 40 * np.log10(np.maximum(mult, mult_min) / mult_min)
                for sel, cols in ((unm, 'gray'), (~unm, [colour[k] for k in pair[~unm]])):
                    if not sel.any():
                        continue
                    if mk == 'o':
                        ax.scatter(*d[sel].T, s=size[sel], marker=mk, facecolors='none', edgecolors=cols, lw=1.2)
                    else:
                        ax.scatter(*d[sel].T, s=size[sel], marker=mk, c=cols, lw=1.2)
                if label_mult:
                    for p_, mm in zip(d, mult):
                        ax.text(*p_, f'{mm:.0f}', fontsize=6, color='gray')
            ax.set_xlim(-h, h);  ax.set_ylim(-h, h);  ax.set_zlim(-h, h)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=elev, azim=az)
            ax.set_xlabel(r'$\Delta x$ [Mpc/$h$]');  ax.set_ylabel(r'$\Delta y$ [Mpc/$h$]');  ax.set_zlabel(r'$\Delta z$ [Mpc/$h$]')
            ax.set_title(f'view: elevation {elev:g}$^\\circ$, azimuth {az:g}$^\\circ$', fontsize=9)
            if col == 0:
                ax.legend(handles=[plt.Line2D([], [], color='k', marker='o', mfc='none', ls='none', label='old'),
                                   plt.Line2D([], [], color='k', marker='x', ls='none', label='fast-directs'),
                                   plt.Line2D([], [], color='C0', ls='-', label='matched: shared colour + line'),
                                   plt.Line2D([], [], color='gray', marker='s', ls='none', label='unmatched: gray')],
                          fontsize=7, loc='upper left')
        fig.suptitle(f'z={z_cmp}, {run_i} MAP {i}: mult {runs[run_i][0]["mult"][i]:.0f} at ({centre[0]:.2f}, {centre[1]:.2f}, '
                     f'{centre[2]:.2f}); box {box:g} Mpc/$h$\n' + ', '.join(f'{len(m[2])} {m[0]}' for m in members)
                     + f' MAPs, {len(keys)} matched pairs', fontsize=10)
        figs.append(fig)
    return figs


def plot_results(ctx, res, by='pass'):
    """All five figures: ``(unmatched counts, matched-MAP properties, match rates, residuals, separation vs multiplicity)``.
    ``by='pass'`` splits the matched population into trajectory tiers / ellipse pass, ``by='tier'`` into the individual
    multiplicity tiers."""
    f1, f1b = plot_unmatched(ctx, res, by)
    return f1, f1b, plot_match_rate(ctx, res), plot_residuals(ctx, res, by), plot_separation_multiplicity(ctx, res)
