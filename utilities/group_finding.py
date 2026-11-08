"""Friends-of-friends group finding on MAPs (from maplogs) and a MAP-population HOD.

Typical notebook use (see GroupFinding.ipynb)::

    maplog = read_maplog_dir(sim, node_type='timeslice')
    maps = select_final_maps(maplog, min_mult=200)       # cheap; reuse across linking lengths
    cats = {ll: find_groups(maps, ll) for ll in (0.1, 0.2)}   # Mpc/h; keep in memory
    plot_group_statistics(cats)
    plot_map_hod(cats)

All functions are pure (no hidden caching, nothing written to disk). The
pericentric-passage test between MAPs inside a group is not implemented yet;
``find_groups`` returns the group membership it will operate on.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def select_final_maps(maplog, min_mult=1):
    """Final-timeslice head nodes of a ``read_maplog_dir`` dict, with mult >= min_mult.

    Returns a dict with ``pos`` (wrapped into [0, BoxSize)), ``vel``, ``mult``,
    ``density``, ``box``, ``mass_per_mult`` (Msun/h per unit multiplicity) and
    ``index`` (row indices into the input maplog), for the selected MAPs.
    """
    hdr = maplog['header']
    box = float(hdr['BoxSize'])
    final = np.max(maplog['timestep'])
    msk = (maplog['control'] == final * 4 + 3) & (maplog['mult'] >= min_mult)
    out = {
        'pos': np.mod(np.asarray(maplog['pos'])[msk], box),
        'mult': np.asarray(maplog['mult'])[msk],
        'index': np.flatnonzero(msk),
        'box': box,
        'mass_per_mult': float(hdr['ParticleMassHMsun']),
        'timestep': int(final),
    }
    for k in ('vel', 'density'):
        if k in maplog:
            out[k] = np.asarray(maplog[k])[msk]
    return out


def find_groups(maps, linking_length, absolute=True):
    """Friends-of-friends on MAP positions with a periodic cKDTree.

    Parameters
    ----------
    maps : dict from ``select_final_maps`` (needs ``pos``, ``mult``, ``box``;
        ``density`` is used to break mass ties, ``mass_per_mult`` for masses).
    linking_length : float
        Comoving Mpc/h if ``absolute``; otherwise a fraction of the mean
        interparticle spacing ``box / N**(1/3)`` of the selected MAPs.
    Returns
    -------
    dict (the input fields plus):
      linking_length : absolute linking length used [Mpc/h]
      label     : (N,) group id per MAP, 0..ngroups-1
      is_central: (N,) bool; largest-mass MAP in its group. Ties -> higher
                  ``density``, then lowest index.
      central   : (ngroups,) MAP index of each group's central
      nmaps     : (ngroups,) number of MAPs per group
      mass      : (ngroups,) summed MAP mass [Msun/h]
      group_mult: (ngroups,) summed multiplicity (per-MAP ``mult`` is unchanged)
    """
    pos, mult, box = maps['pos'], maps['mult'], maps['box']
    n = len(pos)
    ll = linking_length if absolute else linking_length * box / n ** (1 / 3.)

    tree = cKDTree(pos, boxsize=box)
    pairs = tree.query_pairs(ll, output_type='ndarray')
    graph = coo_matrix((np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])),
                       shape=(n, n))
    ngroups, label = connected_components(graph, directed=False)

    # Sort so that the last member of each group is its central:
    # primary key label, then mult, then density, then lowest index.
    idx = np.arange(n)
    dens = maps['density'] if 'density' in maps else np.zeros(n)
    order = np.lexsort((-idx, dens, mult, label))
    last = np.r_[label[order][1:] != label[order][:-1], True]
    central = order[last]               # sorted by label since label is primary key
    is_central = np.zeros(n, bool)
    is_central[central] = True

    mult_g = np.bincount(label, weights=mult, minlength=ngroups)
    out = dict(maps)
    out.update(linking_length=ll, label=label, is_central=is_central, central=central,
               nmaps=np.bincount(label, minlength=ngroups))
    out['group_mult'] = mult_g
    out['mass'] = mult_g * maps['mass_per_mult']
    return out


def hod_from_groups(cat, bins=None, nbins=15):
    """Bin groups by summed MAP mass; mean centrals/satellites and <N(N-1)>/<N> for satellites.

    Returns dict with ``edges``, ``center`` (geometric), ``ngroups`` per bin,
    ``n_cen``, ``n_sat`` (means), ``poisson`` (<Ns(Ns-1)>/<Ns>, 1 for Poisson;
    NaN where <Ns> = 0) and ``mult_per_mass`` for the multiplicity axis.
    """
    mass = cat['mass']
    nsat = cat['nmaps'] - 1
    ncen = np.ones_like(nsat)           # exactly one central per group
    if bins is None:
        bins = np.logspace(np.log10(mass.min()), np.log10(mass.max()) * (1 + 1e-12), nbins + 1)
    which = np.digitize(mass, bins) - 1
    nb = len(bins) - 1
    ok = (which >= 0) & (which < nb)
    cnt = np.bincount(which[ok], minlength=nb)
    def mean(x):
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.bincount(which[ok], weights=x[ok], minlength=nb) / cnt
    ns, ns2 = mean(nsat.astype(float)), mean((nsat * (nsat - 1)).astype(float))
    with np.errstate(invalid='ignore', divide='ignore'):
        poisson = np.where(ns > 0, ns2 / ns, np.nan)
    return dict(edges=bins, center=np.sqrt(bins[:-1] * bins[1:]), ngroups=cnt,
                n_cen=mean(ncen.astype(float)), n_sat=ns, poisson=poisson,
                mult_per_mass=1. / cat['mass_per_mult'])


def plot_group_statistics(cats, fig=None):
    """Group-size distribution and group-mass function for catalogs from
    ``find_groups``; ``cats`` is {label: catalog} (label e.g. the linking length)."""
    import matplotlib.pyplot as plt
    if not isinstance(cats, dict):
        cats = {f"{cats['linking_length']:.3g} Mpc/h": cats}
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4)) if fig is None else (fig, fig.subplots(1, 2))
    for lab, c in cats.items():
        nm = c['nmaps']
        sizes = np.unique(nm)
        counts = np.bincount(nm)[sizes]
        lab = f"{lab} ($N_g$={len(nm)})" if not isinstance(lab, str) else lab
        a0.loglog(sizes, counts, 'o-', ms=3, label=lab)
        edges = np.logspace(np.log10(c['mass'].min()), np.log10(c['mass'].max()), 30)
        h, _ = np.histogram(c['mass'], edges)
        a1.loglog(np.sqrt(edges[1:] * edges[:-1]), np.where(h > 0, h, np.nan), label=lab)
    a0.set(xlabel='MAPs per group', ylabel='number of groups')
    a1.set(xlabel=r'group MAP mass [$M_\odot/h$]', ylabel='number of groups')
    a0.legend(title='linking length')
    fig.tight_layout()
    return fig


def plot_map_hod(cats, bins=None, nbins=15, min_group_count=1, ratio_to=None):
    """MAP-population HOD: <N_sat> vs group mass (top, with a secondary multiplicity axis) and
    <N_sat(N_sat-1)>/<N_sat> beneath (1 = Poisson).

    ``cats`` is a catalog or {label: catalog}. Bins with fewer than ``min_group_count`` groups are
    hidden. If ``ratio_to`` is a key of ``cats``, a second column shows every other curve divided
    by that one (<N_sat> on top, the Poisson ratio below); all curves then share common mass bins.
    Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    if not isinstance(cats, dict):
        cats = {f"{cats['linking_length']:.3g} Mpc/h": cats}
    if ratio_to is not None and ratio_to not in cats:
        raise KeyError(f"ratio_to={ratio_to!r} is not one of {list(cats)}")
    if ratio_to is not None and bins is None:
        lo = min(c['mass'].min() for c in cats.values())
        hi = max(c['mass'].max() for c in cats.values())
        bins = np.logspace(np.log10(lo), np.log10(hi) * (1 + 1e-12), nbins + 1)
    ncol = 2 if ratio_to is not None else 1
    fig, axs = plt.subplots(2, ncol, figsize=(6.5 * ncol, 7), sharex=True, squeeze=False,
                            gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05, wspace=0.28))
    a0, a1 = axs[0, 0], axs[1, 0]
    hods = {lab: hod_from_groups(c, bins=bins, nbins=nbins) for lab, c in cats.items()}
    mpm = next(iter(hods.values()))['mult_per_mass']
    for i, (lab, h) in enumerate(hods.items()):
        s = h['ngroups'] >= min_group_count
        col = f'C{i}'
        a0.loglog(h['center'][s], np.where(h['n_sat'][s] > 0, h['n_sat'][s], np.nan), 'o-',
                  ms=3, color=col, label=lab)
        a1.semilogx(h['center'][s], h['poisson'][s], 'o-', ms=3, color=col)
    a1.axhline(1, color='k', lw=0.8, ls=':')
    a0.set(ylabel=r'$\langle N_{\rm sat}\rangle$')
    a1.set(xlabel=r'group MAP mass $\sum$ mult $\cdot\, m_p$ [$M_\odot/h$]',
           ylabel=r'$\langle N_s(N_s-1)\rangle/\langle N_s\rangle$')
    a0.legend(fontsize=8)
    top = a0.secondary_xaxis('top', functions=(lambda m: m * mpm, lambda n: n / mpm))
    top.set_xlabel('summed MAP multiplicity of group')
    if ratio_to is not None:
        r0, r1 = axs[0, 1], axs[1, 1]
        ref = hods[ratio_to]
        for i, (lab, h) in enumerate(hods.items()):
            if lab == ratio_to:
                continue
            s = (h['ngroups'] >= min_group_count) & (ref['ngroups'] >= min_group_count)
            with np.errstate(invalid='ignore', divide='ignore'):
                r0.semilogx(h['center'][s], np.where(ref['n_sat'][s] > 0, h['n_sat'][s] / ref['n_sat'][s], np.nan),
                            'o-', ms=3, color=f'C{i}', label=lab)
                r1.semilogx(h['center'][s], h['poisson'][s] / ref['poisson'][s], 'o-', ms=3, color=f'C{i}')
        for r in (r0, r1):
            r.axhline(1, color='k', lw=0.8, ls=':')
        r0.set(ylabel=rf'$\langle N_{{\rm sat}}\rangle$ / that of {ratio_to}')
        r1.set(xlabel=a1.get_xlabel(), ylabel='Poisson ratio / that of ' + ratio_to)
        r0.legend(fontsize=8)
        top2 = r0.secondary_xaxis('top', functions=(lambda m: m * mpm, lambda n: n / mpm))
        top2.set_xlabel('summed MAP multiplicity of group')
    for a in axs.ravel():
        a.grid(True, which='both', alpha=0.3)
    return fig


# =====================================================================================
# Orbit-based centrals/satellites from MAP trajectories
# =====================================================================================
# A MAP is a satellite iff it has had a pericentre around a more massive MAP (Osinga's
# subhalo definition); pericentres follow Diemer (2022, sec. 2.4). Stages:
#   1. read_trajectories     main-branch (pos, vel, mult, rel_vel) of each final MAP on the
#                            timeslice-epoch grid, straight from the maplog node stream
#   2. candidate_pairs       (member i, host j): j more massive, i within search_factor R_j
#   3. first_pericentre      per-pair radial-velocity-flip + angular-sweep test
#   4. find_orbit_groups     each MAP joins the most massive host it orbits; roll up to the top
#   5. groups_from_orbits    repackage as a `find_groups`-style catalogue for the HOD plots
import math
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial

NODE_FORMATION, NODE_MERGER, NODE_LIGHTCONE, NODE_TIMESLICE = 0, 1, 2, 3
LOWER_BORN_INSIDE, LOWER_SWEPT, LOWER_SWITCHES = 1, 2, 3


@dataclass
class Trajectories:
    """Main-branch histories of N final-timeslice MAPs on a K-epoch grid (NaN before formation)."""
    grid: np.ndarray        # (K,) timesteps, ascending
    redshift: np.ndarray    # (K,)
    pos: np.ndarray         # (N, K, 3) float32 Mpc/h
    vel: np.ndarray         # (N, K, 3) float32 km/s
    mult: np.ndarray        # (N, K) float32
    ids: np.ndarray         # (N, 4) int64: formation pid triple + formation timestep (unique identity)
    file_index: np.ndarray  # (N,) source file
    head_row: np.ndarray    # (N,) row of the head node in its file
    header: dict
    rel_vel: np.ndarray = None   # (N, K, 3) float32 km/s: mean local DM velocity minus MAP velocity
    density: np.ndarray = None   # (N,) float32, env. density at the final epoch
    # --- only set by read_forest (every branch of the merger tree, not just the dominant one) ---
    parent_track: np.ndarray = None  # (N,) row of the track this one is absorbed into (-1: a tree head)
    merge_k: np.ndarray = None       # (N,) first epoch index at which this track no longer exists on its
                                     #   own (absorbed): the first epoch STRICTLY after the Merger's timestep (a
                                     #   timeslice at a step is written before that step's mergers, so a track
                                     #   merging at an epoch step still has its sample there); K if never absorbed
    first_k: np.ndarray = None       # (N,) first / last epoch index with a timeslice node
    last_k: np.ndarray = None
    merge_event: np.ndarray = None   # (N, K) bool: a secondary was absorbed into this track between
                                     #   epoch k-1 and k (pos/vel jump to the merger's centre of mass)
    row: np.ndarray = None           # (N, K) int32: row in the track's own maplog file of its timeslice node
                                     #   at epoch k (-1: none); file = file_index


def _main_branch_one_file(f, min_mult, with_rel):
    """One file's trajectories. The dominant branch of a head is exactly its rows up to and
    including the first Formation node (the larger progenitor is listed first at each merger),
    so no tree walk is needed -- see doc/claude/reference_maplog.md sec. 2.3."""
    import asdf
    from abacusnbody.data.read_abacus import read_asdf
    asdf.get_config().validate_on_read = False
    cols = ['pos', 'vel', 'mult', 'timestep', 'node_type', 'pid', 'density']
    if with_rel:
        cols += ['rel_vel', 'rel_vel_healpix']
    t = read_asdf(f, load=cols)
    nt, ts, mult = (np.asarray(t[k]) for k in ('node_type', 'timestep', 'mult'))
    is_ts = nt == NODE_TIMESLICE
    grid = np.unique(ts[is_ts])
    heads = np.flatnonzero(is_ts & (ts == grid[-1]) & (mult >= min_mult))

    form = np.flatnonzero(nt == NODE_FORMATION)
    end = form[np.searchsorted(form, heads)]            # first Formation at/after each head
    lens = end - heads + 1
    owner = np.repeat(np.arange(len(heads)), lens)
    rows = np.repeat(heads, lens) + np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
    keep = is_ts[rows]
    rows, owner = rows[keep], owner[keep]
    gi = np.searchsorted(grid, ts[rows])
    if np.unique(owner * len(grid) + gi).size != rows.size:
        raise ValueError(f"{f}: a main branch has two timeslice nodes at one epoch")

    N, K = len(heads), len(grid)
    def scatter(a, shape, gathered=False):
        out = np.full(shape, np.nan, np.float32)
        out[owner, gi] = a if gathered else np.asarray(a)[rows]
        return out
    pos = scatter(t['pos'], (N, K, 3))
    vel = scatter(t['vel'], (N, K, 3))
    m = scatter(mult, (N, K))
    out = dict(grid=grid, pos=pos, vel=vel, mult=m, head_row=heads, header=t.meta,
               density=np.asarray(t['density'])[heads].astype(np.float32),
               ids=np.concatenate([np.asarray(t['pid'])[end].astype(np.int64),
                                   ts[end, None].astype(np.int64)], axis=1))
    if with_rel:
        from abacusnbody.data.output_particle import rel_vel_healpix_lookup
        mag = np.asarray(t['rel_vel'], dtype=np.float32)[rows, None]
        out['rel_vel'] = scatter(mag * rel_vel_healpix_lookup[np.asarray(t['rel_vel_healpix'])[rows]],
                                 (N, K, 3), gathered=True)
    return out


def read_trajectories(mdir, min_mult=50, with_rel_vel=False, nworkers=8):
    """Main-branch trajectories of every final-timeslice MAP with final mult >= min_mult.

    ``mdir`` is a sim directory (``maplogs/*/maplog_z*_*.asdf``). Memory is ~(40-80) K bytes
    per MAP (K epochs), so choose ``min_mult`` accordingly. Redshift uses ``Step_to_Z[timestep]``
    (checked against the timeslice epochs; the ``-1`` in ``read_maplog_dir`` looks one step off).
    """
    files = sorted(glob(str(Path(mdir) / "maplogs/*/maplog_z*_*.asdf")))
    if not files:
        raise FileNotFoundError(f"no maplog files under {mdir}")
    worker = partial(_main_branch_one_file, min_mult=min_mult, with_rel=with_rel_vel)
    with ProcessPoolExecutor(max_workers=nworkers) as ex:
        res = list(ex.map(worker, files))
    grid = res[0]['grid']
    if any(not np.array_equal(r['grid'], grid) for r in res):
        raise ValueError("files disagree on the timeslice epoch grid")
    cat = lambda k: np.concatenate([r[k] for r in res])
    hdr = res[0]['header']
    return Trajectories(
        grid=grid, redshift=np.asarray(hdr['Step_to_Z'])[grid], pos=cat('pos'), vel=cat('vel'),
        mult=cat('mult'), ids=cat('ids'), head_row=cat('head_row'), header=hdr,
        file_index=np.concatenate([np.full(len(r['head_row']), i, np.int32) for i, r in enumerate(res)]),
        rel_vel=cat('rel_vel') if with_rel_vel else None, density=cat('density'))


# ----------------------------------------------------------------------------------------------
# Full merger-tree walk: every MAP that ever existed at a timeslice, not only final-epoch survivors
# ----------------------------------------------------------------------------------------------
from numba import njit


@njit(cache=True)
def _parse_trees(nt, n_forms, n_mergers):
    """Stack-parse a file's node stream (doc/claude/reference_maplog.md sec. 2.2): returns
    (tree_starts, sec_start, complete). At a Merger the resume point is pushed; at a Formation it is
    popped and the next row starts that Merger's secondary subtree; a Formation with an empty stack
    closes a tree. ``sec_start[m]`` is the first row of the secondary branch of Merger row m."""
    n = nt.shape[0]
    sec = np.full(n, -1, np.int64)
    stack = np.empty(n_mergers + 1, np.int64)
    starts = np.empty(n_forms + 1, np.int64)
    sp, ntree, new_tree = 0, 0, True
    for i in range(n):
        if new_tree:
            starts[ntree] = i
            ntree += 1
            new_tree = False
        t = nt[i]
        if t == 1:
            stack[sp] = i
            sp += 1
        elif t == 0:
            if sp > 0:
                sp -= 1
                sec[stack[sp]] = i + 1
            else:
                new_tree = True
    return starts[:ntree], sec, (sp == 0 and new_tree)


def forest_from_nodes(nt, ts, mult, pos, vel, pid, density, min_mult, rel_vel=None, layout_only=False):
    """Every track (a MAP between its formation and its absorption/the end) of one node stream.

    A track is the run of rows from a branch start to the first Formation after it: the primary child
    of a Merger node is the next row, so a track is contiguous, and the secondary branches follow after
    the whole primary subtree. Track starts are the tree heads and the secondary starts.
    ``min_mult`` cuts on the multiplicity at the track's last timeslice node. Returns a dict (see
    ``read_forest``); parents are indices into the returned (cut) tracks.
    """
    nt = np.asarray(nt)
    is_ts = nt == NODE_TIMESLICE
    grid = np.unique(ts[is_ts])
    K = len(grid)
    form = np.flatnonzero(nt == NODE_FORMATION)
    merg = np.flatnonzero(nt == NODE_MERGER)
    heads, sec, complete = _parse_trees(nt.astype(np.int8), len(form), len(merg))
    if not complete:
        raise ValueError("node stream does not end on a complete tree (truncated file?)")
    mrow_all = np.concatenate([np.full(len(heads), -1, np.int64), merg])      # merger row (-1: head)
    start_all = np.concatenate([heads, sec[merg]])
    order = np.argsort(start_all, kind='stable')
    start, mrow = start_all[order], mrow_all[order]
    n_all = len(start)
    parent = np.where(mrow >= 0, np.searchsorted(start, mrow, side='right') - 1, -1)
    end = form[np.searchsorted(form, start)]                  # first Formation at/after each start
    lens = end - start + 1
    owner = np.repeat(np.arange(n_all), lens)
    rows = np.repeat(start, lens) + np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
    row_owner = np.full(len(nt), -1, np.int64)       # every row's track (before the timeslice-only filter)
    row_owner[rows] = owner
    keep_row = is_ts[rows]
    rows, owner = rows[keep_row], owner[keep_row]
    gi = np.searchsorted(grid, ts[rows])
    if np.unique(owner * K + gi).size != rows.size:
        raise ValueError("a track has two timeslice nodes at one epoch")
    first_k = np.full(n_all, K, np.int64)
    last_k = np.full(n_all, -1, np.int64)
    np.minimum.at(first_k, owner, gi)
    np.maximum.at(last_k, owner, gi)
    last_mult = np.zeros(n_all, np.float64)
    at_last = gi == last_k[owner]
    last_row = np.full(n_all, -1, np.int64)
    last_row[owner[at_last]] = rows[at_last]
    has = last_k >= 0
    last_mult[has] = np.asarray(mult)[last_row[has]]
    keep = has & (last_mult >= min_mult)
    # reparent onto the nearest kept ancestor (normally the parent itself: a secondary is never heavier)
    par = parent.copy()
    for _ in range(64):
        bad = (par >= 0) & ~keep[np.maximum(par, 0)]
        if not bad.any():
            break
        par[bad] = parent[par[bad]]
    newidx = np.cumsum(keep) - 1
    N = int(keep.sum())
    row_track = np.where(row_owner >= 0, np.where(keep[np.maximum(row_owner, 0)], newidx[np.maximum(row_owner, 0)], -1), -1)
    if layout_only:
        return dict(grid=grid, row_track=row_track, n_tracks=N)
    sel = keep[owner]
    rows, owner, gi = rows[sel], newidx[owner[sel]], gi[sel]

    def scatter(a, shape):
        out = np.full(shape, np.nan, np.float32)
        out[owner, gi] = np.asarray(a)[rows]
        return out
    rowtab = np.full((N, K), -1, np.int32)
    rowtab[owner, gi] = rows
    out = dict(
        row=rowtab, grid=grid, pos=scatter(pos, (N, K, 3)), vel=scatter(vel, (N, K, 3)), mult=scatter(mult, (N, K)),
        head_row=start[keep], first_k=first_k[keep], last_k=last_k[keep],
        parent_track=np.where(par[keep] >= 0, newidx[np.maximum(par[keep], 0)], -1),
        merge_k=np.where(mrow[keep] >= 0, np.searchsorted(grid, ts[np.maximum(mrow[keep], 0)], side='right'), K),
        density=np.asarray(density)[last_row[keep]].astype(np.float32),
        ids=np.concatenate([np.asarray(pid)[end[keep]].astype(np.int64), ts[end[keep], None].astype(np.int64)], axis=1))
    if rel_vel is not None:
        rv = np.full((N, K, 3), np.nan, np.float32)
        rv[owner, gi] = rel_vel[rows]
        out['rel_vel'] = rv
    return out


def _forest_one_file(f, min_mult, with_rel):
    import asdf
    from abacusnbody.data.read_abacus import read_asdf
    asdf.get_config().validate_on_read = False
    cols = ['pos', 'vel', 'mult', 'timestep', 'node_type', 'pid', 'density']
    if with_rel:
        cols += ['rel_vel', 'rel_vel_healpix']
    t = read_asdf(f, load=cols)
    rel = None
    if with_rel:
        from abacusnbody.data.output_particle import rel_vel_healpix_lookup
        rel = (np.asarray(t['rel_vel'], dtype=np.float32)[:, None]
               * rel_vel_healpix_lookup[np.asarray(t['rel_vel_healpix'])])
    out = forest_from_nodes(np.asarray(t['node_type']), np.asarray(t['timestep']), np.asarray(t['mult']),
                            np.asarray(t['pos']), np.asarray(t['vel']), np.asarray(t['pid']),
                            np.asarray(t['density']), min_mult, rel)
    out['header'] = t.meta
    return out


def read_forest(mdir, min_mult=50, with_rel_vel=False, nworkers=8):
    """Like ``read_trajectories`` but for EVERY track in the merger trees: absorbed MAPs are rows too
    (NaN after their absorption), so each epoch has the full set of MAPs with last-epoch mult >=
    ``min_mult``. Extra Trajectories fields: ``parent_track`` (the track absorbed into; the dominant,
    primary progenitor survives), ``merge_k`` (K if never absorbed), ``first_k``, ``last_k`` and
    ``merge_event`` (N, K). Use ``survivor_at(T, k)`` for the object each track belongs to at epoch k.
    Memory is roughly (tracks / final MAPs) times that of ``read_trajectories``."""
    files = sorted(glob(str(Path(mdir) / "maplogs/*/maplog_z*_*.asdf")))
    if not files:
        raise FileNotFoundError(f"no maplog files under {mdir}")
    worker = partial(_forest_one_file, min_mult=min_mult, with_rel=with_rel_vel)
    with ProcessPoolExecutor(max_workers=nworkers) as ex:
        res = list(ex.map(worker, files))
    grid = res[0]['grid']
    if any(not np.array_equal(r['grid'], grid) for r in res):
        raise ValueError("files disagree on the timeslice epoch grid")
    cat = lambda k: np.concatenate([r[k] for r in res])
    offs = np.cumsum([0] + [len(r['head_row']) for r in res])[:-1]
    parent = np.concatenate([np.where(r['parent_track'] >= 0, r['parent_track'] + o, -1)
                             for r, o in zip(res, offs)])
    K, hdr = len(grid), res[0]['header']
    merge_k = cat('merge_k')
    N = len(parent)
    ev = np.zeros((N, K), bool)
    c = np.flatnonzero((parent >= 0) & (merge_k < K))
    ev[parent[c], merge_k[c]] = True
    return Trajectories(
        grid=grid, redshift=np.asarray(hdr['Step_to_Z'])[grid], pos=cat('pos'), vel=cat('vel'),
        mult=cat('mult'), ids=cat('ids'), head_row=cat('head_row'), header=hdr,
        file_index=np.concatenate([np.full(len(r['head_row']), i, np.int32) for i, r in enumerate(res)]),
        rel_vel=cat('rel_vel') if with_rel_vel else None, density=cat('density'),
        parent_track=parent, merge_k=merge_k, first_k=cat('first_k'), last_k=cat('last_k'), merge_event=ev,
        row=cat('row'))


def survivor_at(T, k):
    """(N,) row of the object each track belongs to at epoch k: itself if it exists then, else the track
    it was absorbed into (followed up the chain until one that exists at k). A track not yet formed at
    k maps to itself. Needs a ``read_forest`` Trajectories."""
    cur = np.arange(len(T.parent_track))
    for _ in range(256):
        m = (T.merge_k[cur] <= k) & (T.parent_track[cur] >= 0)
        if not m.any():
            break
        cur[m] = T.parent_track[cur[m]]
    return cur


def halo_radius(mult, z, header, mass_factor=4.5, so_delta=200.0):
    """Stand-in host radius [Mpc/h]: R_{so_delta,mean} of a halo of ``mass_factor`` x mult particles.

    ``mass_factor`` (~ M200m / central-MAP multiplicity; 4.5 was calibrated against Rockstar by the
    aurora-products collaborator) is a float or a callable ``mass_factor(mult, z)`` returning the
    factor elementwise. ``mult`` and ``z`` broadcast against each other (e.g. mult (B, K) with
    z (K,)), and the result has the broadcast shape. This is the one halo-model input of the method.
    """
    mf = mass_factor(mult, z) if callable(mass_factor) else mass_factor
    nbar = float(header['NP']) / float(header['BoxSize']) ** 3
    return np.cbrt(3 * mf * mult / (4 * np.pi * so_delta * nbar))


def mass_rank(T):
    """Rank by final multiplicity, descending; ties -> higher density, then formation id (so the
    order never depends on file layout). rank 0 = most massive."""
    m0 = T.mult[:, -1]
    order = np.lexsort((T.ids[:, 3], T.ids[:, 2], T.ids[:, 1], T.ids[:, 0], -T.density, -m0))
    rank = np.empty(len(order), np.int64)
    rank[order] = np.arange(len(order))
    return rank


def candidate_pairs(T, radius_fn, search_factor=3.0, rank=None, workers=8):
    """(i, j) with j more massive than i and i within search_factor * R_j of j at the final epoch.
    ``radius_fn(mult, z)`` gives the host radius [Mpc/h]."""
    box = float(T.header['BoxSize'])
    rank = mass_rank(T) if rank is None else rank
    p = np.mod(T.pos[:, -1].astype(np.float64), box)
    tree = cKDTree(p, boxsize=box)
    R = radius_fn(T.mult[:, -1].astype(np.float64), T.redshift[-1])      # final-epoch radii
    nb = tree.query_ball_point(p, search_factor * R, workers=workers)
    cnt = np.fromiter(map(len, nb), np.int64, count=len(nb))
    J = np.repeat(np.arange(len(nb)), cnt)
    I = np.concatenate([np.asarray(x, np.int64) for x in nb]) if cnt.sum() else np.empty(0, np.int64)
    k = rank[J] < rank[I]
    return I[k], J[k]


def first_pericentre(r, rhat, vr, Rh, track_factor=2.0, phi_first=math.cos(math.radians(45)),
                     phi_lower=-0.8, return_state=False):
    """First pericentre of each pair (Diemer 2022 sec. 2.4), vectorised over pairs (axis 0), looping
    over the K epochs.

    r (B,K) separation, rhat (B,K,3) unit separation, vr (B,K) physical radial velocity
    (peculiar + Hubble flow), Rh (B,K) host radius. NaN where either MAP does not yet exist.

    Tracking starts at the first epoch within track_factor*Rh; that direction is the reference
    and phi(t) = rhat(t).rhat(t0). A pericentre is vr going from <0 to >=0 with phi < phi_first at
    that epoch or the one before; a flip at too-large phi stays "pending" and fires once phi drops
    below phi_first with vr still >0. Lower limits (still counted as orbiting): the MAP is already
    inside Rh at the pair's first common epoch; phi < phi_lower without a pericentre; or 3 flips
    inside Rh without one passing the angle test.

    Returns (epoch index of the pericentre or -1, lower-limit reason 0/1/2/3). With
    ``return_state`` also a third array: True where the pair ends the history with a flip still
    pending (a flip at too-large phi that never resolved), i.e. an undecided pair.
    """
    B, K = r.shape
    ok = np.isfinite(r) & np.isfinite(vr)
    near = ok & (r < track_factor * Rh)
    t0 = np.where(near.any(1), near.argmax(1), K)
    k1 = np.where(ok.any(1), ok.argmax(1), K)
    ar = np.arange(B)
    kc = np.minimum(k1, K - 1)
    born_in = (k1 < K) & (r[ar, kc] < Rh[ar, kc])
    phi = (rhat * rhat[ar, np.minimum(t0, K - 1)][:, None, :]).sum(2)

    first = np.where(born_in, k1, -1)
    reason = np.where(born_in, LOWER_BORN_INSIDE, 0).astype(np.int8)
    pending = np.zeros(B, bool)
    nflip = np.zeros(B, np.int64)
    for k in range(1, K):
        live = (first < 0) & (k > t0) & ok[:, k] & ok[:, k - 1]
        if not live.any():
            continue
        flip = live & (vr[:, k - 1] < 0) & (vr[:, k] >= 0)
        angle_ok = (phi[:, k] < phi_first) | (phi[:, k - 1] < phi_first)
        fire = (flip & angle_ok) | (live & pending & (vr[:, k] > 0) & (phi[:, k] < phi_first))
        nflip += flip & (r[:, k] < Rh[:, k])
        swept = live & ~fire & (phi[:, k] < phi_lower)
        three = live & ~fire & ~swept & (nflip >= 3)
        pending = live & ~fire & ~swept & ~three & ((flip & ~angle_ok) | (pending & (vr[:, k] > 0)))
        first[fire | swept | three] = k
        reason[swept] = LOWER_SWEPT
        reason[three] = LOWER_SWITCHES
    return (first, reason, pending) if return_state else (first, reason)


def find_orbit_groups(T, mass_factor=4.5, search_factor=3.0, so_delta=200.0, track_factor=2.0,
                      phi_first=math.cos(math.radians(45)), phi_lower=-0.8,
                      host_velocity='raw', batch=100_000, workers=8):
    """Centrals/satellites from pericentres (see the stage list above).

    mass_factor: float, or callable ``mass_factor(mult, z)`` (see ``halo_radius``).

    host_velocity: 'raw' uses both MAPs' own velocities; 'corrected' replaces the host's with
    MAP + rel_vel (the mean local DM velocity, a better halo-bulk proxy for a central; needs
    ``read_trajectories(..., with_rel_vel=True)``). The member always keeps its own velocity.

    Returns dict: is_central (N,), host (N,) top-level host (self for centrals), parent (N,) host
    orbited directly, peri_epoch (N,) grid index (-1 for centrals), peri_z (N,), infall_epoch (N,)
    grid index of the first epoch within track_factor * R_parent (<= peri_epoch; -1 for centrals),
    infall_z (N,), peri_depth (N,) r/R_host at the pericentre, lower_reason (N,) int8, plus rank,
    n_pairs, n_orbits. ``peri_epoch`` / ``infall_epoch`` index ``T.grid`` / ``T.redshift``; for a
    lower-limit satellite (lower_reason > 0) the pericentre epoch is only a bound.
    """
    if host_velocity not in ('raw', 'corrected'):
        raise ValueError("host_velocity must be 'raw' or 'corrected'")
    if host_velocity == 'corrected' and T.rel_vel is None:
        raise ValueError("host_velocity='corrected' needs read_trajectories(with_rel_vel=True)")
    box = float(T.header['BoxSize'])
    N, K = T.mult.shape
    radius = lambda m, zz: halo_radius(m, zz, T.header, mass_factor, so_delta)
    rank = mass_rank(T)
    I, J = candidate_pairs(T, radius, search_factor, rank, workers)

    om = float(T.header['Omega_M'])
    z = np.asarray(T.redshift, np.float64)
    hubble_a = 100 * np.sqrt(om * (1 + z) ** 3 + 1 - om) / (1 + z)   # H(z) a, km/s per comoving Mpc/h

    orbit = np.zeros(I.size, bool)
    kper = np.full(I.size, -1, np.int64)
    depth = np.full(I.size, np.nan)
    why = np.zeros(I.size, np.int8)
    tracked = np.zeros(I.size, bool)     # pair came within track_factor * R_host at some epoch
    pending = np.zeros(I.size, bool)     # ...and ended with an unresolved flip
    infall_pair = np.full(I.size, -1, np.int64)   # first epoch within track_factor * R_host (-1: never)
    for a in range(0, I.size, batch):
        i, j = I[a:a + batch], J[a:a + batch]
        d = (T.pos[i] - T.pos[j]).astype(np.float64)
        d -= box * np.round(d / box)
        r = np.linalg.norm(d, axis=2)
        Rh = radius(T.mult[j].astype(np.float64), z[None, :])      # host radius at each epoch
        with np.errstate(invalid='ignore', divide='ignore'):
            near = np.flatnonzero((r < max(track_factor, 1.0) * Rh).any(1))   # others can't orbit
            if not near.size:
                continue
            i, j, d, r, Rh = i[near], j[near], d[near], r[near], Rh[near]
            rhat = d / r[..., None]
            vj = T.vel[j].astype(np.float64)
            if host_velocity == 'corrected':
                vj = vj + np.nan_to_num(T.rel_vel[j].astype(np.float64))
            vr = ((T.vel[i] - vj) * rhat).sum(2) + hubble_a[None, :] * r
        f, why_b, pend_b = first_pericentre(r, rhat, vr, Rh, track_factor, phi_first, phi_lower,
                                            return_state=True)
        hit = f >= 0
        sel = a + near
        orbit[sel], kper[sel], why[sel] = hit, f, why_b
        tracked[sel], pending[sel] = True, pend_b
        neark = np.isfinite(r) & np.isfinite(vr) & (r < track_factor * Rh)   # as in first_pericentre
        infall_pair[sel] = np.where(neark.any(1), neark.argmax(1), -1)
        h = np.flatnonzero(hit)
        depth[sel[h]] = r[h, f[h]] / Rh[h, f[h]]

    # each member joins the most massive host it orbits; process in mass order so hosts resolve first
    host, parent = np.arange(N), np.arange(N)
    is_central = np.ones(N, bool)
    peri_epoch = np.full(N, -1, np.int64)
    infall_epoch = np.full(N, -1, np.int64)
    peri_depth = np.full(N, np.nan)
    lower = np.zeros(N, np.int8)
    o = np.flatnonzero(orbit)
    if o.size:
        s = np.lexsort((rank[J[o]], I[o]))
        o = o[s]
        first_of_member = o[np.r_[True, I[o][1:] != I[o][:-1]]]
        order = first_of_member[np.argsort(rank[I[first_of_member]])]
        for p in order:
            i, j = I[p], J[p]
            parent[i], host[i] = j, host[j]
            is_central[i] = False
            peri_epoch[i], peri_depth[i], lower[i] = kper[p], depth[p], why[p]
            infall_epoch[i] = kper[p] if infall_pair[p] < 0 else min(infall_pair[p], kper[p])
    return dict(is_central=is_central, host=host, parent=parent, peri_epoch=peri_epoch,
                peri_z=np.where(peri_epoch >= 0, T.redshift[np.maximum(peri_epoch, 0)], np.nan),
                infall_epoch=infall_epoch,
                infall_z=np.where(infall_epoch >= 0, T.redshift[np.maximum(infall_epoch, 0)], np.nan),
                peri_depth=peri_depth, lower_reason=lower, rank=rank,
                n_pairs=int(I.size), n_orbits=int(orbit.sum()),
                # every candidate pair and its outcome (before the best host is chosen); I = member, J = host
                pairs=dict(I=I, J=J, orbit=orbit, peri_epoch=kper, depth=depth, reason=why,
                           tracked=tracked, pending=pending, infall_epoch=infall_pair))


def groups_from_orbits(T, orb):
    """Repackage ``find_orbit_groups`` output like ``find_groups`` (group = a central and its
    satellites) so ``hod_from_groups`` / ``plot_map_hod`` / ``plot_group_statistics`` work."""
    cen = np.flatnonzero(orb['is_central'])
    gid = np.full(len(orb['host']), -1, np.int64)
    gid[cen] = np.arange(len(cen))
    label = gid[orb['host']]
    m0 = T.mult[:, -1].astype(np.float64)
    mpm = float(T.header['ParticleMassHMsun'])
    gm = np.bincount(label, weights=m0, minlength=len(cen))
    return dict(label=label, is_central=orb['is_central'], central=cen, host=orb['host'],
                nmaps=np.bincount(label, minlength=len(cen)), group_mult=gm, mass=gm * mpm,
                mult=m0, mass_per_mult=mpm, box=float(T.header['BoxSize']),
                linking_length=np.nan)


def orbit_catalogue(T, orb):
    """Per-MAP central/satellite catalogue (a MAP-population mock) from ``find_orbit_groups``.

    Returns a dict of (N,) arrays, row-matched to ``T``: ``id`` (N,4 formation id), ``pos`` (final
    epoch, Mpc/h), ``vel``, ``mult``, ``is_central``, ``host`` / ``parent`` (row indices; self for
    centrals), ``host_id`` / ``parent_id`` (formation ids), ``host_mult``, ``n_sat`` (satellites
    of each central, 0 for satellites), ``peri_z``, ``peri_depth``, ``lower_reason``.
    """
    host = orb['host']
    cen = orb['is_central']
    m0 = T.mult[:, -1]
    return dict(
        id=T.ids, pos=T.pos[:, -1], vel=T.vel[:, -1], mult=m0, is_central=cen,
        host=host, parent=orb['parent'], host_id=T.ids[host], parent_id=T.ids[orb['parent']],
        host_mult=m0[host], n_sat=np.where(cen, np.bincount(host, minlength=len(host)) - 1, 0),
        peri_z=orb['peri_z'], infall_z=orb['infall_z'], peri_depth=orb['peri_depth'],
        lower_reason=orb['lower_reason'])


def epoch_index(T, z):
    """Index into ``T.grid`` / ``T.redshift`` of the epoch nearest redshift ``z``."""
    return int(np.argmin(np.abs(np.asarray(T.redshift, np.float64) - z)))


def population_masks(T, orb, k, lower_limits='include'):
    """Central / infalling / orbiting populations at epoch index ``k`` of ``T`` (bool (N,) each).

    Only MAPs that exist at k are in any population. With S the satellites of the final catalogue:
      orbiting   S and peri_epoch  <= k                      (has passed its first pericentre)
      infalling  S and infall_epoch <= k < peri_epoch        (inside track_factor * R_host, no pericentre yet)
      central    exists at k and in neither of the above     (includes MAPs that will later fall in, and
                 final centrals; there is no exit/backsplash logic, so status never reverts)
    ``lower_limits``: 'include' (default) keeps lower-limit satellites, whose epochs are bounds, not
    measurements; 'exclude' drops them from all three populations (they are ambiguous, not central).
    Returns dict(central, infalling, orbiting, exists, k, z).
    """
    if lower_limits not in ('include', 'exclude'):
        raise ValueError("lower_limits must be 'include' or 'exclude'")
    exists = np.isfinite(T.mult[:, k])
    sat = ~orb['is_central']
    if lower_limits == 'exclude':
        exists = exists & ~(sat & (orb['lower_reason'] > 0))
    orbiting = exists & sat & (orb['peri_epoch'] <= k)
    infalling = exists & sat & ~orbiting & (orb['infall_epoch'] <= k)
    central = exists & ~orbiting & ~infalling
    return dict(central=central, infalling=infalling, orbiting=orbiting, exists=exists, k=int(k),
                z=float(T.redshift[k]))


def check_catalogue(cat):
    """Invariants of a central/satellite catalogue; raises AssertionError on the first failure.
    Returns a dict of summary numbers."""
    n = len(cat['host'])
    idx, cen, host = np.arange(n), cat['is_central'], cat['host']
    assert np.array_equal(host[cen], idx[cen]), "centrals must host themselves"
    assert not np.any(host[~cen] == idx[~cen]), "satellites must host elsewhere"
    assert cen[host].all(), "every host must be a central (roll-up incomplete)"
    assert (cat['host_mult'][~cen] >= cat['mult'][~cen]).all(), "a host must not be lighter than its satellite"
    assert cat['n_sat'].sum() == (~cen).sum(), "satellite counts must add up"
    return dict(n=n, n_central=int(cen.sum()), n_sat=int((~cen).sum()),
                sat_fraction=float((~cen).mean()),
                lower_limit_fraction=float((cat['lower_reason'][~cen] > 0).mean()) if (~cen).any() else 0.0)


# =====================================================================================
# Comparing groupings (e.g. FOF vs orbit)
# =====================================================================================
def maps_from_trajectories(T):
    """``find_groups`` input from trajectories: the same MAPs, row-matched, final epoch."""
    box = float(T.header['BoxSize'])
    return dict(pos=np.mod(T.pos[:, -1].astype(np.float64), box), mult=T.mult[:, -1],
                density=T.density, box=box, mass_per_mult=float(T.header['ParticleMassHMsun']),
                vel=T.vel[:, -1])


def _pairs(n):
    return n * (n - 1) / 2.0


def compare_groupings(a, b, mult=None):
    """How different are two groupings of the same MAPs? ``a``, ``b`` are catalogues from
    ``find_groups`` / ``groups_from_orbits`` (need ``label``, ``is_central``), row-matched.

    Returned numbers (a = reference, b = other):
      pair_precision/recall : of the MAP pairs sharing a group in b (a), the fraction also
                              sharing one in a (b). Singletons have no pairs.
      ari                   : adjusted Rand index (1 = identical partitions, 0 = chance).
      central_agree         : fraction of MAPs with the same central/satellite flag.
      confusion             : 2x2 [[cen_a&cen_b, cen_a&sat_b], [sat_a&cen_b, sat_a&sat_b]].
      sat_b_in_a_group_of_host : of b's satellites, the fraction in the same a-group as b's host
                                 (requires b to carry ``host`` rows -- only orbit catalogues; else None).
      nsat_a, nsat_b        : total satellites.
      frac_mass_changed     : (if ``mult``) mult-weighted fraction of MAPs whose flag differs.
    """
    la, lb = np.asarray(a['label']), np.asarray(b['label'])
    n = len(la)
    na, nb = np.bincount(la), np.bincount(lb)
    key = la.astype(np.int64) * (lb.max() + 1) + lb
    _, nij = np.unique(key, return_counts=True)
    sij, sa, sb, tot = _pairs(nij).sum(), _pairs(na).sum(), _pairs(nb).sum(), _pairs(n)
    expected = sa * sb / tot
    ca, cb = np.asarray(a['is_central']), np.asarray(b['is_central'])
    out = dict(
        n=n, ngroups_a=len(na), ngroups_b=len(nb),
        pair_precision=float(sij / sb) if sb else np.nan,     # b's pairs found in a
        pair_recall=float(sij / sa) if sa else np.nan,        # a's pairs found in b
        ari=float((sij - expected) / (0.5 * (sa + sb) - expected)) if (0.5 * (sa + sb) - expected) else np.nan,
        central_agree=float((ca == cb).mean()),
        confusion=np.array([[np.sum(ca & cb), np.sum(ca & ~cb)], [np.sum(~ca & cb), np.sum(~ca & ~cb)]]),
        nsat_a=int((~ca).sum()), nsat_b=int((~cb).sum()),
        sat_b_in_a_group_of_host=None)
    host = b.get('host')
    if host is not None and (~cb).any():
        s = np.flatnonzero(~cb)
        out['sat_b_in_a_group_of_host'] = float((la[s] == la[host[s]]).mean())
    if mult is not None:
        out['frac_mass_changed'] = float(np.sum(np.asarray(mult)[ca != cb]) / np.sum(mult))
    return out


# =====================================================================================
# Diagnostics: who is "central" only by default, and how ambiguous is host assignment?
# =====================================================================================
CATEGORIES = ('satellite', 'satellite (lower limit)', 'central: no candidate host',
              'central: never tracked', 'central: tracked, undecided', 'central: tracked, no pericentre')


def classify_maps(T, orb, min_epochs=5, mult_edges=None):
    """Why is each MAP a central or a satellite?

    Every MAP gets a label (central unless it orbits something), but "central" is a *default*:
    the absence of a pericentre, not a positive determination. This splits the centrals by how
    much evidence there was:
      0 satellite (firm)            orbits a more massive MAP, membership not a lower limit
      1 satellite (lower limit)     membership rests on a lower-limit rule (born inside / swept / flips)
      2 central: no candidate host  nothing more massive within search_factor * R at the final epoch
      3 central: never tracked      candidates exist, none ever came within track_factor * R_host
      4 central: tracked, undecided some pair ends with a flip pending (neither confirmed nor ruled out)
      5 central: tracked, no pericentre   came close, settled without a pericentre (infalling/flyby)
    ``short_history`` flags MAPs with fewer than ``min_epochs`` epochs of existence (any label is
    weak for them). Returns dict: category (N,), names, counts, fraction, short_history, and, if
    ``mult_edges`` is given, counts per final-multiplicity bin.
    """
    P = orb['pairs']
    N = len(orb['is_central'])
    sat = ~orb['is_central']
    cat = np.where(sat, np.where(orb['lower_reason'] > 0, 1, 0), 5)
    has_cand = np.bincount(P['I'], minlength=N) > 0
    was_tracked = np.bincount(P['I'][P['tracked']], minlength=N) > 0
    undecided = np.bincount(P['I'][P['pending']], minlength=N) > 0
    cen = ~sat
    cat[cen & ~has_cand] = 2
    cat[cen & has_cand & ~was_tracked] = 3
    cat[cen & was_tracked] = 5
    cat[cen & undecided & was_tracked] = 4
    counts = np.bincount(cat, minlength=len(CATEGORIES))
    out = dict(category=cat, names=CATEGORIES, counts=counts, fraction=counts / N,
               short_history=np.isfinite(T.mult).sum(1) < min_epochs)
    out['n_short_history'] = int(out['short_history'].sum())
    if mult_edges is not None:
        b = np.digitize(T.mult[:, -1], mult_edges) - 1
        ok = (b >= 0) & (b < len(mult_edges) - 1)
        out['mult_edges'] = np.asarray(mult_edges)
        out['counts_by_mult'] = np.array([np.bincount(cat[ok & (b == k)], minlength=len(CATEGORIES))
                                          for k in range(len(mult_edges) - 1)])
    return out


def candidate_structure(T, orb):
    """Is the candidate list a set of disjoint groups? How often does a MAP orbit several hosts?

    Per member (rows of ``T``):
      n_cand          candidate hosts (more massive MAPs within search_factor * R at the final epoch)
      n_orbited       candidates it passes the pericentre test with
      n_top_hosts     distinct *top-level* hosts among those orbited (after rolling each host up to
                      its own host). >1 means the choice between hosts changes the grouping;
                      n_orbited > 1 with n_top_hosts == 1 is just a nested chain (satellite of a
                      satellite of the same central).
    Connected components (edge = pair) of the candidate graph and of the orbit graph (only pairs
    that passed). ``*_label`` per MAP, ``*_sizes`` per component; ``frac_in_largest`` is the share
    of MAPs in the biggest component (near 1 = the graph percolates and cannot be split into
    independent groups). ``n_centrals_in_component`` counts resolved centrals per component:
    anything above 1 means that component contains several final groups.
    """
    P = orb['pairs']
    N = len(orb['is_central'])
    I, J, o = P['I'], P['J'], P['orbit']
    n_cand = np.bincount(I, minlength=N)
    n_orb = np.bincount(I[o], minlength=N)
    key = np.unique(I[o].astype(np.int64) * N + orb['host'][J[o]])
    n_top = np.bincount(key // N, minlength=N)

    def components(i, j):
        g = coo_matrix((np.ones(len(i), np.int8), (i, j)), shape=(N, N))
        nc, lab = connected_components(g, directed=False)
        sizes = np.bincount(lab, minlength=nc)
        cen = np.bincount(lab[orb['is_central']], minlength=nc)
        return dict(label=lab, sizes=sizes, n_components=int(nc),
                    n_nonsingleton=int((sizes > 1).sum()),
                    frac_in_largest=float(sizes.max() / N) if N else np.nan,
                    n_centrals_in_component=cen)

    cand, orbg = components(I, J), components(I[o], J[o])
    sat = ~orb['is_central']
    return dict(
        n_cand=n_cand, n_orbited=n_orb, n_top_hosts=n_top,
        frac_sats_multi_orbited=float((n_orb[sat] > 1).mean()) if sat.any() else np.nan,
        frac_sats_multi_top_host=float((n_top[sat] > 1).mean()) if sat.any() else np.nan,
        frac_all_multi_candidate=float((n_cand > 1).mean()),
        candidate_graph=cand, orbit_graph=orbg,
        frac_cand_components_multi_central=float((cand['n_centrals_in_component'] > 1).mean()),
        frac_orbit_components_multi_central=float((orbg['n_centrals_in_component'][orbg['sizes'] > 1] > 1).mean())
        if orbg['n_nonsingleton'] else 0.0)


# ======================================================================================================
# Time-resolved classification over the full merger forest (read_forest): a causal pair state machine.
#
# At every epoch k the alive tracks go into a periodic cKDTree and every (member, host) pair within
# search_factor * R_host that is not yet in the pair table is added. Each pair in the table carries the
# same pericentre state as ``first_pericentre`` (far / infalling / orbiting / orbiting lower limit) and is
# advanced one epoch at a time, so the state at epoch k only depends on epochs <= k. Per-MAP status and
# host at epoch k are read off the pair table. Hosts are ordered by PEAK multiplicity (time invariant), so
# the host graph can never contain a cycle. Merger handling: see ``_merge_surgery``.
# ======================================================================================================
ST_FAR, ST_INFALL, ST_ORBIT, ST_ORBIT_LOWER = 0, 1, 2, 3        # pair states
STATUS_NONE, STATUS_CENTRAL, STATUS_INFALLING, STATUS_ORBITING, STATUS_ABSORBED = -1, 0, 1, 2, 3  # per MAP

_PAIR_SPEC = dict(      # field: (dtype, extra shape, default)
    i=(np.int64, (), 0), j=(np.int64, (), 0),                 # member (lighter), host (heavier) track rows
    state=(np.int8, (), 0), peri=(np.int16, (), -1), infall=(np.int16, (), -1), reason=(np.int8, (), 0),
    pending=(bool, (), False), nflip=(np.int8, (), 0), ref=(np.float32, (3,), 0.0),
    vr_prev=(np.float32, (), np.nan), phi_prev=(np.float32, (), 1.0),
    last_eval=(np.int16, (), -2),       # last epoch both were alive and the pair was advanced (-2: never)
    far=(np.int16, (), 0),              # consecutive epochs beyond the search radius (for optional pruning)
    carry=(np.int16, (), -1), rk=(np.int16, (), -1),          # infall epoch carried over a restart; restart epoch
    nsup=(np.int16, (), 0))             # pericentre flips suppressed by the settling window


class _PairTable:
    def __init__(self):
        self.a = {n: np.zeros((0,) + sh, dt) for n, (dt, sh, _) in _PAIR_SPEC.items()}

    def __len__(self):
        return len(self.a['i'])

    def keep(self, mask):
        for n in self.a:
            self.a[n] = self.a[n][mask]

    def append(self, **cols):
        n = len(next(iter(cols.values())))
        for name, (dt, sh, dflt) in _PAIR_SPEC.items():
            v = np.asarray(cols[name], dt) if name in cols else np.full((n,) + sh, dflt, dt)
            self.a[name] = np.concatenate([self.a[name], v])


def peak_rank(T):
    """Rank by PEAK multiplicity over the track's life, descending (0 = heaviest); ties -> higher density,
    then formation id. Time invariant, so 'j hosts i' never reverses along the history."""
    peak = np.nanmax(np.where(np.isfinite(T.mult), T.mult, -1.0), axis=1)
    order = np.lexsort((T.ids[:, 3], T.ids[:, 2], T.ids[:, 1], T.ids[:, 0], -T.density, -peak))
    rank = np.empty(len(order), np.int64)
    rank[order] = np.arange(len(order))
    return rank


def _merge_surgery(tb, cons_of, minor, ncnt, rank, k, N):
    """Re-key the pair table at the epoch k at which tracks are absorbed.

    ``cons_of[c]`` = the surviving object M that constituent track c becomes part of (-1: not involved; M
    itself is a constituent: the primary), ``minor[c]`` = c is a minor secondary (mass fraction < f_min),
    ``ncnt[M]`` = number of non-minor constituents of M (>= 1).

    For each new pair (Mi member, Mj host), from the old pairs (ci member, cj host) of constituents (only
    pairs whose member ci is non-minor and whose orientation matches the new one count as evidence):
      * every non-minor constituent of Mi orbits SOME constituent of Mj (member side: all; host side:
        any, i.e. the satellites of every constituent join the merged halo)  ->  Mi orbits Mj; pericentre
        epoch = latest over member constituents of the earliest over host constituents; lower-limit if
        any pair used was;
      * else if Mi has one non-minor constituent and Mj has one (a minor merger on either side): the
        best old pair (highest state, then the primary pair) is inherited unchanged;
      * else (a major merger that breaks the orbit) the pair restarts at k: far/infalling with a fresh
        reference direction, the earliest old infall epoch carried over.
    Pairs of minor member secondaries, pairs between constituents of the same object and pairs whose
    orientation reverses are dropped.
    """
    counts = dict(success=0, inherit=0, restart=0)
    if not len(tb):
        return counts
    A = tb.a
    ci, cj = cons_of[A['i']], cons_of[A['j']]
    aff = (ci >= 0) | (cj >= 0)
    if not aff.any():
        return counts
    old = np.flatnonzero(aff)
    Mi = np.where(ci[old] >= 0, ci[old], A['i'][old])
    Mj = np.where(cj[old] >= 0, cj[old], A['j'][old])
    ev = (Mi != Mj) & ~minor[A['i'][old]] & (rank[Mi] > rank[Mj])
    ev_rows, Mi, Mj = old[ev], Mi[ev], Mj[ev]
    new = {}
    if ev_rows.size:
        g_key = Mi.astype(np.int64) * N + Mj
        gk, g = np.unique(g_key, return_inverse=True)
        G = len(gk)
        gMi, gMj = gk // N, gk % N
        need, nh = ncnt[gMi], ncnt[gMj]
        st = A['state'][ev_rows].astype(np.int64)
        member = A['i'][ev_rows]
        # orbiting evidence: per (group, member constituent) the earliest pericentre over host constituents
        o = np.flatnonzero(st >= ST_ORBIT)
        n_orb = np.zeros(G, np.int64)
        gperi = np.full(G, -1, np.int64)
        greason = np.zeros(G, np.int64)
        glower = np.zeros(G, bool)
        if o.size:
            s = np.lexsort((A['peri'][ev_rows][o], member[o], g[o]))
            o = o[s]
            first = np.r_[True, (g[o][1:] != g[o][:-1]) | (member[o][1:] != member[o][:-1])]
            u = o[first]
            np.add.at(n_orb, g[u], 1)
            np.maximum.at(gperi, g[u], A['peri'][ev_rows][u])
            np.maximum.at(greason, g[u], A['reason'][ev_rows][u])
            glower[g[u][st[u] == ST_ORBIT_LOWER]] = True
        inf = A['infall'][ev_rows].astype(np.int64)
        ginfall = np.full(G, 10 ** 9, np.int64)
        has_inf = (st >= ST_INFALL) & (inf >= 0)
        np.minimum.at(ginfall, g[has_inf], inf[has_inf])
        ginfall = np.where(ginfall == 10 ** 9, -1, ginfall)
        gbest = np.full(G, -1, np.int64)
        np.maximum.at(gbest, g, st)
        success = n_orb == need
        inherit = ~success & (need == 1) & (nh == 1) & (gbest >= ST_INFALL)
        restart = ~success & ~inherit & (ginfall >= 0)
        # inherit: best old row per group (highest state; primary-primary row on ties; earliest peri)
        prim = (cons_of[A['i'][ev_rows]] == A['i'][ev_rows]) & (cons_of[A['j'][ev_rows]] == A['j'][ev_rows])
        sel = np.lexsort((A['peri'][ev_rows], -prim.astype(np.int64), -st, g))
        firstg = sel[np.r_[True, g[sel][1:] != g[sel][:-1]]]
        src = ev_rows[firstg]                      # src[g] = chosen old row of group g
        for n in A:
            new[n] = np.zeros((G,) + _PAIR_SPEC[n][1], _PAIR_SPEC[n][0])
            new[n][:] = _PAIR_SPEC[n][2]
        keep_g = success | inherit | restart
        counts = dict(success=int(success.sum()), inherit=int(inherit.sum()), restart=int(restart.sum()))
        for n in A:
            if n not in ('i', 'j'):
                new[n][inherit] = A[n][src[inherit]]
        new['i'], new['j'] = gMi, gMj
        new['state'][success] = np.where(glower[success], ST_ORBIT_LOWER, ST_ORBIT)
        new['peri'][success] = gperi[success]
        new['reason'][success] = greason[success]
        new['infall'][success] = np.where(ginfall[success] >= 0, ginfall[success], gperi[success])
        new['last_eval'][success] = -2
        new['state'][restart], new['carry'][restart] = ST_FAR, ginfall[restart]
        new['rk'][restart], new['last_eval'][restart] = k, -2
        new = {n: v[keep_g] for n, v in new.items()}
    tb.keep(~aff)
    if new and len(new['i']):
        tb.append(**new)
    return counts


def _propose_pairs(tb, T, k, alive_idx, rank, radius_fn, search_factor, box, workers, chunk=200_000):
    """Add the (member, host) pairs found by a periodic cKDTree of the tracks alive at epoch k."""
    N = len(rank)
    p = np.mod(T.pos[alive_idx, k].astype(np.float64), box)
    p[p >= box] = 0.0
    tree = cKDTree(p, boxsize=box)
    R = radius_fn(T.mult[alive_idx, k].astype(np.float64), T.redshift[k])
    hosts, members = [], []
    for s in range(0, len(p), chunk):
        nb = tree.query_ball_point(p[s:s + chunk], search_factor * R[s:s + chunk], workers=workers)
        cnt = np.fromiter(map(len, nb), np.int64, count=len(nb))
        if not cnt.sum():
            continue
        J = np.repeat(np.arange(s, s + len(nb)), cnt)
        I = np.concatenate([np.asarray(x, np.int64) for x in nb])
        m = rank[alive_idx[J]] < rank[alive_idx[I]]
        hosts.append(alive_idx[J[m]])
        members.append(alive_idx[I[m]])
    if not hosts:
        return 0
    h, m = np.concatenate(hosts), np.concatenate(members)
    if len(tb):
        fresh = ~np.isin(m * N + h, tb.a['i'] * N + tb.a['j'])
        h, m = h[fresh], m[fresh]
    tb.append(i=m, j=h)
    return len(m)


@dataclass
class EpochScan:
    """Output of ``scan_epochs``: per (track, epoch) classification, all indexed like ``T`` (N tracks, K epochs)."""
    status: np.ndarray       # (N,K) int8: -1 not formed, 0 central, 1 infalling, 2 orbiting (satellite), 3 absorbed
    host: np.ndarray         # (N,K) int32: direct host (most massive object orbited, else infalled-toward); the
                             #   track itself for a central; the object absorbed into for status 3; -1 if none
    infall_epoch: np.ndarray # (N,K) int16: epoch index the pair with ``host`` started infalling (-1: none)
    peri_epoch: np.ndarray   # (N,K) int16: pericentre epoch w.r.t. ``host`` (status 2; a bound if lower_reason>0)
    lower_reason: np.ndarray # (N,K) int8: 0 or LOWER_* for the pair with ``host``
    recent_merger: np.ndarray# (N,K) bool: the track absorbed something fewer than n_settle epochs ago
    r_over_R: np.ndarray     # (N,K) float16: separation / R_host to ``host`` (NaN for centrals); large for an
                             #   orbiting track = flung out since its pericentre (backsplash candidate)
    ever_satellite: np.ndarray  # (N,K) bool: has ever had status 'orbiting' (persists through resets; see ever_rule)
    ever_epoch: np.ndarray   # (N,) int16: epoch it first became a satellite (-1: never), final value
    ever_host: np.ndarray    # (N,) int32: its host then (-1: never)
    rank: np.ndarray         # (N,) peak-multiplicity rank (0 = heaviest)
    params: dict
    diag: dict               # per-epoch arrays: n_pairs, n_new, n_suppressed_flips, n_restart, n_inherit, n_orbit_merge


def scan_epochs(T, mass_factor=4.5, search_factor=3.0, so_delta=200.0, track_factor=2.0,
                phi_first=math.cos(math.radians(45)), phi_lower=-0.8, host_velocity='raw',
                f_min=0.2, n_settle=3, leave_factor=3.0, prune_after=None, ever_rule='or', workers=8, verbose=False):
    """Causal central / infalling / orbiting / absorbed status of EVERY track at EVERY epoch.

    T must come from ``read_forest``. Per epoch: periodic cKDTree of the alive tracks proposes new
    (member, host) pairs within ``search_factor`` * R_host; every pair in the table is advanced with the
    ``first_pericentre`` logic (tracking starts within ``track_factor`` * R_host; orbit = v_r flip with
    phi < 45 deg, or the lower-limit cases). Choices that are not in ``find_orbit_groups``:

    f_min: a secondary with fewer than this fraction of the merged object's (pre-merge) multiplicity is
        'minor'; see ``_merge_surgery`` for the rule.
    n_settle: pericentre / lower-limit detection is skipped for a pair for this many epochs from the
        epoch at which either object absorbed something (merged centre of mass + unsettled velocity).
        KNOWN BIAS: a pericentre inside this window is missed (counted in diag['n_suppressed_flips']).
    leave_factor: an infalling pair reverts to 'far' once r >= leave_factor * R_host while receding with
        no pericentre pending (a fly-by cannot be told from an infaller except causally); None disables.
    prune_after: drop never-orbiting pairs after this many consecutive epochs beyond ``search_factor``
        * R_host (None: keep every pair).
    ever_rule: 'or' (default): at a merger the merged object is flagged ever_satellite if ANY non-minor
        constituent was (with the earliest such epoch/host); 'primary': only the primary's own flag counts.
        The orbit state itself can still reset at a merger (see ``_merge_surgery``); the ever flag cannot.
        ``diag['n_ever_gained_or']`` counts the merged objects whose flag the OR rule alone would set.
    """
    if ever_rule not in ('or', 'primary'):
        raise ValueError("ever_rule must be 'or' or 'primary'")
    if T.parent_track is None:
        raise ValueError("scan_epochs needs read_forest Trajectories")
    if host_velocity == 'corrected' and T.rel_vel is None:
        raise ValueError("host_velocity='corrected' needs read_forest(..., with_rel_vel=True)")
    box = float(T.header['BoxSize'])
    N, K = T.mult.shape
    z = np.asarray(T.redshift, np.float64)
    om = float(T.header['Omega_M'])
    hubble_a = 100 * np.sqrt(om * (1 + z) ** 3 + 1 - om) / (1 + z)
    radius_fn = lambda m, zz: halo_radius(m, zz, T.header, mass_factor, so_delta)
    rank = peak_rank(T)
    alive_all = np.isfinite(T.pos[:, :, 0])
    status = np.full((N, K), STATUS_NONE, np.int8)
    host_o = np.full((N, K), -1, np.int32)
    infall_o = np.full((N, K), -1, np.int16)
    peri_o = np.full((N, K), -1, np.int16)
    lower_o = np.zeros((N, K), np.int8)
    recent_o = np.zeros((N, K), bool)
    xr_o = np.full((N, K), np.nan, np.float16)
    ever_o = np.zeros((N, K), bool)
    ever_now = np.zeros(N, bool)
    ever_ep = np.full(N, -1, np.int16)
    ever_host = np.full(N, -1, np.int32)
    last_merge = np.full(N, -10 ** 6, np.int64)
    diag = {n: np.zeros(K, np.int64) for n in
            ('n_pairs', 'n_new', 'n_suppressed_flips', 'n_restart', 'n_inherit', 'n_orbit_merge', 'n_merged',
             'n_ever_gained_or')}
    tb = _PairTable()
    absorbed = (T.parent_track >= 0) & (T.merge_k < K)
    cons_of = np.full(N, -1, np.int64)
    for k in range(K):
        # ---- 1. mergers that complete at this epoch -------------------------------------------------
        S = np.flatnonzero(absorbed & (T.merge_k == k))
        if S.size:
            surv = survivor_at(T, k)
            M = surv[S]
            ok = alive_all[M, k] & (M != S)
            S, M = S[ok], M[ok]
        if S.size:
            Mu = np.unique(M)
            lastm = T.mult[S, T.last_k[S]].astype(np.float64)
            kprev = max(k - 1, 0)
            pm = T.mult[Mu, kprev].astype(np.float64)
            pm = np.where(np.isfinite(pm), pm, T.mult[Mu, k])
            total = pm.copy()
            np.add.at(total, np.searchsorted(Mu, M), lastm)
            frac = lastm / total[np.searchsorted(Mu, M)]
            minor = np.zeros(N, bool)
            minor[S] = frac < f_min
            ncnt = np.ones(N, np.int64)
            np.add.at(ncnt, M[~minor[S]], 1)
            cons_of[S], cons_of[Mu] = M, Mu
            c = _merge_surgery(tb, cons_of, minor, ncnt, rank, k, N)
            diag['n_orbit_merge'][k], diag['n_inherit'][k], diag['n_restart'][k] = c['success'], c['inherit'], c['restart']
            cons_of[S], cons_of[Mu] = -1, -1
            last_merge[Mu] = k
            diag['n_merged'][k] = len(Mu)
            # persistent satellite flag: earliest epoch/host over the non-minor constituents
            src = S[~minor[S]]
            gain = src[ever_now[src]]
            if gain.size:
                tgt = M[np.searchsorted(S, gain)]
                diag['n_ever_gained_or'][k] = int(np.unique(tgt[~ever_now[tgt]]).size)
                if ever_rule == 'or':
                    o = np.lexsort((ever_ep[gain], tgt))
                    gain, tgt = gain[o], tgt[o]
                    f1 = np.r_[True, tgt[1:] != tgt[:-1]]
                    gain, tgt = gain[f1], tgt[f1]
                    better = ~ever_now[tgt] | (ever_ep[gain] < ever_ep[tgt])
                    ever_ep[tgt[better]], ever_host[tgt[better]] = ever_ep[gain[better]], ever_host[gain[better]]
                    ever_now[tgt[better]] = True
        if len(tb):                                    # drop pairs with a dead end (absorbed / ended)
            tb.keep((T.last_k[tb.a['i']] >= k) & (T.last_k[tb.a['j']] >= k))
        # ---- 2. new candidate pairs from this epoch's tree ------------------------------------------
        alive_idx = np.flatnonzero(alive_all[:, k])
        if alive_idx.size > 1:
            diag['n_new'][k] = _propose_pairs(tb, T, k, alive_idx, rank, radius_fn, search_factor, box, workers)
        # ---- 3. advance every pair with both ends alive ---------------------------------------------
        A = tb.a
        e = np.flatnonzero(alive_all[A['i'], k] & alive_all[A['j'], k]) if len(tb) else np.empty(0, np.int64)
        if e.size:
            i, j = A['i'][e], A['j'][e]
            d = (T.pos[i, k] - T.pos[j, k]).astype(np.float64)
            d -= box * np.round(d / box)
            r = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
            rhat = d / r[:, None]
            Rh = radius_fn(T.mult[j, k].astype(np.float64), z[k])
            vj = T.vel[j, k].astype(np.float64)
            if host_velocity == 'corrected':
                vj = vj + np.nan_to_num(T.rel_vel[j, k].astype(np.float64))
            vr = ((T.vel[i, k] - vj) * rhat).sum(1) + hubble_a[k] * r
            st, peri, infall, reason = A['state'][e].copy(), A['peri'][e].copy(), A['infall'][e].copy(), A['reason'][e].copy()
            pending, nflip, ref = A['pending'][e].copy(), A['nflip'][e].copy(), A['ref'][e].copy()
            vr_prev, phi_prev, last_eval = A['vr_prev'][e], A['phi_prev'][e], A['last_eval'][e]
            nsup, far = A['nsup'][e].copy(), A['far'][e].copy()
            carry, rk = A['carry'][e].copy(), A['rk'][e]
            # born inside the host at the pair's first common epoch: orbiting lower limit
            kc = np.maximum(T.first_k[i], T.first_k[j])
            born = (last_eval == -2) & (st == ST_FAR) & (k == kc) & (r < Rh) & (rk < 0)
            st[born], reason[born], peri[born], infall[born] = ST_ORBIT_LOWER, LOWER_BORN_INSIDE, k, k
            # infalling pairs: the pericentre test
            sup = (k - last_merge[i] < n_settle) | (k - last_merge[j] < n_settle)
            live = (st == ST_INFALL) & (last_eval == k - 1)
            phi = (rhat * ref).sum(1)
            flip = live & (vr_prev < 0) & (vr >= 0)
            diag['n_suppressed_flips'][k] = int((flip & sup).sum())
            nsup += (flip & sup)
            live &= ~sup
            flip &= live
            angle_ok = (phi < phi_first) | (phi_prev < phi_first)
            fire = (flip & angle_ok) | (live & pending & (vr > 0) & (phi < phi_first))
            nflip += (flip & (r < Rh)).astype(np.int8)
            swept = live & ~fire & (phi < phi_lower)
            three = live & ~fire & ~swept & (nflip >= 3)
            pending_new = live & ~fire & ~swept & ~three & ((flip & ~angle_ok) | (pending & (vr > 0)))
            pending = np.where(live, pending_new, pending)
            hit = fire | swept | three
            st[hit] = np.where(fire[hit], ST_ORBIT, ST_ORBIT_LOWER)
            peri[hit] = k
            reason[swept] = LOWER_SWEPT
            reason[three] = LOWER_SWITCHES
            # a receding infaller that is already far has left (fly-by)
            if leave_factor is not None:
                gone = (st == ST_INFALL) & ~hit & ~pending & (r >= leave_factor * Rh) & (vr > 0)
                st[gone], infall[gone], carry[gone], nflip[gone] = ST_FAR, -1, -1, 0
                ref[gone] = 0
            # start tracking: far pairs within track_factor * R_host (reference direction = now)
            start = (st == ST_FAR) & (r < track_factor * Rh)
            st[start] = ST_INFALL
            infall[start] = np.where((rk[start] == k) & (carry[start] >= 0), carry[start], k)
            carry[start] = -1
            ref[start], pending[start], nflip[start] = rhat[start], False, 0
            phi = np.where(start, 1.0, phi)
            # optional pruning counter
            far = np.where((st < ST_ORBIT) & (r >= search_factor * Rh), far + 1, 0).astype(np.int16)
            for name, v in dict(state=st, peri=peri, infall=infall, reason=reason, pending=pending, nflip=nflip,
                                ref=ref, vr_prev=vr, phi_prev=phi, last_eval=np.full(e.size, k), nsup=nsup,
                                far=far, carry=carry).items():
                A[name][e] = v
            # ---- 4. per-track status at this epoch ----------------------------------------------------
            status[alive_idx, k] = STATUS_CENTRAL
            host_o[alive_idx, k] = alive_idx
            cls_rows = np.flatnonzero(st >= ST_INFALL)
            if cls_rows.size:
                ci_ = i[cls_rows]
                cls = np.where(st[cls_rows] >= ST_ORBIT, 2, 1)
                s = np.lexsort((rank[j[cls_rows]], -cls, ci_))
                s = s[np.r_[True, ci_[s][1:] != ci_[s][:-1]]]
                pick, mem = cls_rows[s], ci_[s]
                status[mem, k] = np.where(cls[s] == 2, STATUS_ORBITING, STATUS_INFALLING)
                host_o[mem, k] = j[pick]
                infall_o[mem, k] = infall[pick]
                peri_o[mem, k] = np.where(cls[s] == 2, peri[pick], -1)
                lower_o[mem, k] = np.where(st[pick] == ST_ORBIT_LOWER, reason[pick], 0)
                xr_o[mem, k] = (r[pick] / Rh[pick]).astype(np.float16)
                newsat = mem[(cls[s] == 2) & ~ever_now[mem]]
                if newsat.size:
                    sel_ = np.isin(mem, newsat)
                    ever_now[mem[sel_]] = True
                    ever_ep[mem[sel_]], ever_host[mem[sel_]] = k, j[pick][sel_]
            if prune_after is not None:
                A2 = tb.a
                tb.keep(~((A2['state'] < ST_ORBIT) & (A2['far'] >= prune_after)))
        else:
            status[alive_idx, k] = STATUS_CENTRAL
            host_o[alive_idx, k] = alive_idx
        recent_o[:, k] = alive_all[:, k] & (k - last_merge < n_settle)
        # absorbed tracks belong to the object that absorbed them
        gone_ = np.flatnonzero(absorbed & (T.merge_k <= k) & ~alive_all[:, k])
        if gone_.size:
            status[gone_, k] = STATUS_ABSORBED
            host_o[gone_, k] = survivor_at(T, k)[gone_]
        ever_o[:, k] = ever_now & (status[:, k] != STATUS_NONE)
        diag['n_pairs'][k] = len(tb)
        if verbose:
            print(f"epoch {k}/{K}: alive {alive_idx.size}, pairs {len(tb)}, new {diag['n_new'][k]}, "
                  f"merged {diag['n_merged'][k]}", flush=True)
    return EpochScan(status=status, host=host_o, infall_epoch=infall_o, peri_epoch=peri_o,
                     lower_reason=lower_o, recent_merger=recent_o, r_over_R=xr_o, ever_satellite=ever_o,
                     ever_epoch=ever_ep, ever_host=ever_host, rank=rank, diag=diag,
                     params=dict(mass_factor=mass_factor, search_factor=search_factor, so_delta=so_delta,
                                 track_factor=track_factor, phi_first=phi_first, phi_lower=phi_lower,
                                 host_velocity=host_velocity, f_min=f_min, n_settle=n_settle,
                                 leave_factor=leave_factor, prune_after=prune_after, ever_rule=ever_rule))


def top_host(sc, k):
    """(N,) the top-level host at epoch k: follow ``host`` while the host is itself a satellite or an
    infaller; a central (or a track not alive then) maps to itself, absorbed tracks to their object's top host."""
    h = sc.host[:, k].astype(np.int64)
    own = np.arange(len(h))
    h = np.where(h >= 0, h, own)
    for _ in range(64):
        nxt = h[h]
        if np.array_equal(nxt, h):
            break
        h = nxt
    return h


def scan_masks(sc, k, lower_limits='include'):
    """Population masks at epoch k from an ``EpochScan``, like ``population_masks``: dict(central, infalling,
    orbiting, exists, absorbed). ``lower_limits``: 'include' counts lower-limit satellites as orbiting,
    'exclude' drops them from every population."""
    s = sc.status[:, k]
    low = (sc.lower_reason[:, k] > 0) & (s == STATUS_ORBITING)
    ex = (s >= STATUS_CENTRAL) & (s <= STATUS_ORBITING)
    if lower_limits == 'exclude':
        ex &= ~low
    return dict(central=ex & (s == STATUS_CENTRAL), infalling=ex & (s == STATUS_INFALLING),
                orbiting=ex & (s == STATUS_ORBITING), exists=ex, absorbed=s == STATUS_ABSORBED)


def host_distance_samples(T, sc, k, search_factor=3.0, mass_factor=4.5, so_delta=200.0, min_age=4, workers=8):
    """One sample per well-characterised MAP at epoch k near a central: where it sits relative to the most
    massive central within ``search_factor`` * R_host, and whether the pericentre test calls it orbiting that
    host's halo. This is the calibration set for a distance-only satellite rule (x_sat(M_host, z, env)).

    Members: tracks alive at k with age (epochs since first_k, inclusive) >= ``min_age``, status central,
    infalling or orbiting, lighter (peak rank) than the host. Returns dict of equal-length arrays: member,
    host, x = r / R_host, orbiting (status 2 and top host == host), status, host_mult, member_mult,
    density (member's), age, z.
    """
    box = float(T.header['BoxSize'])
    st = sc.status[:, k]
    alive = np.flatnonzero((st >= STATUS_CENTRAL) & (st <= STATUS_ORBITING))
    cen = alive[st[alive] == STATUS_CENTRAL]
    pos = np.mod(T.pos[alive, k].astype(np.float64), box)
    pos[pos >= box] = 0.0
    tree = cKDTree(pos, boxsize=box)
    z = float(T.redshift[k])
    Rc = halo_radius(T.mult[cen, k].astype(np.float64), z, T.header, mass_factor, so_delta)
    nb = tree.query_ball_point(np.mod(T.pos[cen, k].astype(np.float64), box) % box, search_factor * Rc, workers=workers)
    cnt = np.fromiter(map(len, nb), np.int64, count=len(nb))
    empty = dict(member=np.empty(0, np.int64), host=np.empty(0, np.int64), x=np.empty(0), orbiting=np.empty(0, bool),
                 status=np.empty(0, np.int8), host_mult=np.empty(0), member_mult=np.empty(0),
                 density=np.empty(0), age=np.empty(0, np.int64), z=z)
    if not cnt.sum():
        return empty
    J = np.repeat(cen, cnt)
    Rj = np.repeat(Rc, cnt)
    I = alive[np.concatenate([np.asarray(x, np.int64) for x in nb])]
    m = (sc.rank[I] > sc.rank[J]) & (k - T.first_k[I] + 1 >= min_age)
    I, J, Rj = I[m], J[m], Rj[m]
    if not I.size:
        return empty
    s = np.lexsort((sc.rank[J], I))                 # most massive central per member
    I, J, Rj = I[s], J[s], Rj[s]
    f = np.r_[True, I[1:] != I[:-1]]
    I, J, Rj = I[f], J[f], Rj[f]
    d = (T.pos[I, k] - T.pos[J, k]).astype(np.float64)
    d -= box * np.round(d / box)
    r = np.linalg.norm(d, axis=1)
    top = top_host(sc, k)
    return dict(member=I, host=J, x=r / Rj, orbiting=(st[I] == STATUS_ORBITING) & (top[I] == J), status=st[I],
                host_mult=T.mult[J, k].astype(np.float64), member_mult=T.mult[I, k].astype(np.float64),
                density=T.density[I].astype(np.float64), age=k - T.first_k[I] + 1, z=z)


def p_crossing(x_mid, p, level):
    """Smallest x at which the orbiting fraction p(x) falls to ``level`` (linear interpolation; NaN if it
    never crosses; the first bin's x if it starts below)."""
    ok = np.isfinite(p)
    x_mid, p = x_mid[ok], p[ok]
    below = np.flatnonzero(p < level)
    if not below.size:
        return np.nan
    b = below[0]
    if b == 0:
        return x_mid[0]
    return x_mid[b - 1] + (level - p[b - 1]) * (x_mid[b] - x_mid[b - 1]) / (p[b] - p[b - 1])


# ======================================================================================================
# Row-matched labels: every row of a maplog file (formation, merger, light-cone and timeslice nodes) gets
# the classification of the track it belongs to. Only timeslice epochs run the algorithm (scan_epochs);
# the other rows INHERIT: a row at timestep t takes the track's label at the last epoch strictly before t
# (causal); rows before the track's first timeslice (formation rows) are BACKFILLED from its first epoch.
# ======================================================================================================
PROV_EXACT, PROV_FORWARD, PROV_BACKFILL, PROV_NONE = 0, 1, 2, 3


def label_rows(T, sc, file_index, nt, ts, row_track, track_offset):
    """Row-matched labels for one maplog file. ``nt``/``ts`` are the file's node_type / timestep columns,
    ``row_track`` the file-local track of each row (from ``forest_from_nodes(..., layout_only=True)``, -1: not a
    kept track), ``track_offset`` the index of the file's first track in ``T``. Returns a dict of (n_rows,) arrays:
    track (global T row, -1 none), epoch (grid index the label was read at, -1 none), provenance
    (PROV_EXACT timeslice node / PROV_FORWARD inherited from an earlier epoch / PROV_BACKFILL formation rows
    before the first epoch / PROV_NONE), status, host_track, host_file, host_row (the host's timeslice row at
    that epoch in the host's own file), ever_satellite, ever_epoch, infall_epoch, peri_epoch, lower_reason,
    r_over_R, recent_merger."""
    n = len(nt)
    K = len(T.grid)
    has = row_track >= 0
    g = np.where(has, row_track + track_offset, 0)
    kprev = np.searchsorted(T.grid, ts, side='left') - 1            # last epoch strictly before the node
    is_ts = np.asarray(nt) == NODE_TIMESLICE
    kexact = np.searchsorted(T.grid, ts)
    fk, lk = T.first_k[g], T.last_k[g]
    k = np.where(is_ts, kexact, kprev)
    kuse = np.clip(k, fk, lk)
    prov = np.where(is_ts, PROV_EXACT, np.where(kprev < fk, PROV_BACKFILL, PROV_FORWARD)).astype(np.int8)
    ok = has & (lk >= 0)
    prov[~ok] = PROV_NONE
    kuse = np.where(ok, kuse, 0)
    st = np.where(ok, sc.status[g, kuse], -1).astype(np.int8)
    ok &= st >= 0                                    # a track missing a sample at kuse: nothing to inherit
    prov[~ok] = PROV_NONE
    host = np.where(ok, sc.host[g, kuse], -1).astype(np.int64)
    hrow = np.where(host >= 0, T.row[np.maximum(host, 0), kuse], -1)
    out = dict(
        track=np.where(ok, g, -1).astype(np.int32), epoch=np.where(ok, kuse, -1).astype(np.int16), provenance=prov,
        status=np.where(ok, st, -1).astype(np.int8), host_track=host.astype(np.int32),
        host_file=np.where(host >= 0, T.file_index[np.maximum(host, 0)], -1).astype(np.int16),
        host_row=hrow.astype(np.int64),
        ever_satellite=np.where(ok, sc.ever_satellite[g, kuse], False),
        ever_epoch=np.where(ok, sc.ever_epoch[g], -1).astype(np.int16),
        infall_epoch=np.where(ok, sc.infall_epoch[g, kuse], -1).astype(np.int16),
        peri_epoch=np.where(ok, sc.peri_epoch[g, kuse], -1).astype(np.int16),
        lower_reason=np.where(ok, sc.lower_reason[g, kuse], 0).astype(np.int8),
        r_over_R=np.where(ok, sc.r_over_R[g, kuse], np.nan).astype(np.float16),
        recent_merger=np.where(ok, sc.recent_merger[g, kuse], False))
    return out


def _layout_one_file(f, min_mult):
    import asdf
    from abacusnbody.data.read_abacus import read_asdf
    asdf.get_config().validate_on_read = False
    t = read_asdf(f, load=['node_type', 'timestep', 'mult'])
    nt, ts = np.asarray(t['node_type']), np.asarray(t['timestep'])
    lay = forest_from_nodes(nt, ts, np.asarray(t['mult']), None, None, None, None, min_mult, layout_only=True)
    return nt, ts, lay


def write_row_labels(T, sc, mdir, out_dir, min_mult=50, files=None):
    """Write ``<out_dir>/<maplog file stem>.rowlabels.npz`` (arrays of ``label_rows``, same row order as the maplog
    file) for every maplog file under ``mdir`` (or only the file indices in ``files``). ``min_mult`` must be the
    value ``read_forest`` used. Refuses to write anywhere under /flare/Abacus/AbacusAurora (the sims)."""
    out_dir = Path(out_dir).resolve()
    if str(out_dir).startswith('/flare/Abacus/AbacusAurora'):
        raise ValueError("refusing to write under the simulation tree")
    out_dir.mkdir(parents=True, exist_ok=True)
    flist = sorted(glob(str(Path(mdir) / "maplogs/*/maplog_z*_*.asdf")))
    ntr = np.bincount(T.file_index, minlength=len(flist))
    if len(ntr) != len(flist):
        raise ValueError("T and the maplog directory disagree on the number of files")
    offs = np.concatenate([[0], np.cumsum(ntr)])[:-1]
    paths = []
    for i in (range(len(flist)) if files is None else files):
        nt, ts, lay = _layout_one_file(flist[i], min_mult)
        if lay['n_tracks'] != ntr[i]:
            raise ValueError(f"file {i}: {lay['n_tracks']} tracks in the layout but {ntr[i]} in T (min_mult mismatch?)")
        lab = label_rows(T, sc, i, nt, ts, lay['row_track'], offs[i])
        p = out_dir / (Path(flist[i]).stem + '.rowlabels.npz')
        np.savez_compressed(p, **lab)
        paths.append(str(p))
    return paths


def load_scan(path):
    """Load what ``run_group_scan.py --save-scan`` wrote (``scan.npz``, ``diag.npz``, ``summary.json`` in ``path``)
    as an ``EpochScan`` (arrays indexed like the ``read_forest`` Trajectories of the same run)."""
    import json
    path = Path(path)
    z = np.load(path / 'scan.npz')
    diag = dict(np.load(path / 'diag.npz'))
    diag.pop('redshift', None)
    params = json.loads((path / 'summary.json').read_text())['params']
    return EpochScan(status=z['status'], host=z['host'], infall_epoch=z['infall_epoch'], peri_epoch=z['peri_epoch'],
                     lower_reason=z['lower_reason'], recent_merger=z['recent_merger'], r_over_R=z['r_over_R'],
                     ever_satellite=z['ever_satellite'], ever_epoch=z['ever_epoch'], ever_host=z['ever_host'],
                     rank=z['rank'], params=params, diag=diag)
