#!/usr/bin/env python
"""Match MAPs to TNG halos (FoF groups) by position, at one or more redshifts.

This is a reimplementation of ``make_map_tng_galaxy_dict_pos`` from
``illustris_utilities.py`` for the *new* MAP interface (the ``maplog_*.asdf``
maplog files, see ``analysis.ipynb``).

For every MAP, the nearest TNG halo is found (minimizing the periodic
distance normalized by the halo's ``Group_R_Mean200`` radius, matching the
original ``compare_pos`` metric ``dist / R``).  The result is a dictionary
keyed by halo (group) index whose values record the halo size and the pids of
the MAPs that matched it.

Single process, no MPI: the nearest-neighbour search (``compare_pos``) is
numba ``@njit(parallel=True)``, i.e. thread-parallel over every core via
``prange`` -- there is no benefit to also splitting MAPs across MPI ranks, and
dropping MPI means only one process ever imports ``abacus``, and the whole
match only ever needs a single node (thread pools don't span nodes). See
``match_maps_to_tng`` for the multi-redshift entry point meant to be called
directly from a run script/.pbs file.

Run with, e.g.::

    python -m acj.match_maps_to_tng \
        --basePath /flare/Abacus/ajohnson/TNG300/TNG300-2/output/ \
        --mapdir   /flare/Abacus/ajohnson/TNG300-3-Dark-Abacus_ne/ \
        --redshifts 0.2 0.5 1.0 \
        --out_dir  Images/
"""

import os
import argparse
import pickle
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
from astropy import units as u
from numba import njit, prange
import illustris_python as il

# `abacus` is an editable meson-python install whose import *finder* rebuilds
# the C++ extension (build/cp313) on any `import abacus.*` -- even pure-Python
# submodules. This script is single-process/single-node (no MPI), but a run
# script may still launch several of these concurrently (e.g. one per test in
# a .pbs TEST list), and those separate processes rebuilding build/cp313 at the
# same time would corrupt it ("ninja: error: rebuilding 'build.ninja'"). The
# only abacus dependency in this code path is abacus.param.InputFile, which is
# pure Python (io + shlex), so load it straight from source and register it in
# sys.modules. Seeding the module cache makes Python resolve `from abacus.param
# import InputFile` from cache without ever consulting the meson-python finder
# -- both here and in acj.utilities below, which imports it at module scope.
# No rebuild, no editable build needed at all.
import sys
import types
import importlib.util
from pathlib import Path

if "abacus.param" not in sys.modules:
    _if_src = Path(__file__).resolve().parents[1] / "py/abacus/param/input_file.py"
    _spec = importlib.util.spec_from_file_location("abacus.param.input_file", _if_src)
    _ifmod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_ifmod)
    _abacus = sys.modules.setdefault("abacus", types.ModuleType("abacus"))
    if not hasattr(_abacus, "__path__"):
        _abacus.__path__ = []  # mark as a package so `abacus.param` is a valid child
    _param = types.ModuleType("abacus.param")
    _param.InputFile = _ifmod.InputFile
    _abacus.param = _param
    sys.modules["abacus.param"] = _param

# Centralized maplog reader: change reading behavior in one place (acj.utilities)
# and it propagates here. Relative import works under `python -m acj.match_maps_to_tng`;
# fall back to an absolute import for standalone loads (e.g. the notebook doing
# `from match_maps_to_tng import *` with acj/ on sys.path but no parent package).
try:
    from .utilities import (read_maplog_dir, get_threshold, get_sci_not, get_xi,
                             get_available_tng_snapshots, map_step_for_redshift,
                             _xi_plot_values, _ratio_suffix, read_gm_dict_for_maplog,
                             map_satellite_mask, rel_vel_healpix_to_vector)
except ImportError:
    from utilities import (read_maplog_dir, get_threshold, get_sci_not, get_xi,
                            get_available_tng_snapshots, map_step_for_redshift,
                             _xi_plot_values, _ratio_suffix, read_gm_dict_for_maplog,
                             map_satellite_mask, rel_vel_healpix_to_vector)


# --------------------------------------------------------------------------- #
# Output record (replaces HaloMapRec; stores MAP pids as hashed by
# ``read_maplog_dir(..., compute_pids=True)``).
# --------------------------------------------------------------------------- #
class MapGalaxyRec(object):
    """Records the MAP galaxies matched to a single TNG halo.

    Like the original ``HaloMapRec``, the matched MAPs are recorded by their
    pid (the new maplog interface exposes ``pid``, so we keep that rather than
    bare array indices).
    """

    def __init__(self, N, map_pid):
        self.N = N                  # halo size (dm particle count)
        self.map_pids = [map_pid]   # pids of the MAPs matched to this halo

    def n_map(self):
        return len(self.map_pids)

    def append(self, map_pid):
        self.map_pids.append(map_pid)


# --------------------------------------------------------------------------- #
# Position matching (njit, periodic minimum image, parallel over MAPs)
# --------------------------------------------------------------------------- #
@njit(nogil=True, parallel=True)
def compare_pos(map_pos, halo_pos, halo_R, BoxSize):
    """For each MAP return the index of the nearest halo.

    The metric is the periodic distance divided by the halo radius
    ``halo_R`` (same selection as the original ``compare_pos``, which used
    ``argmin(dist / halo_R)``).
    """
    Nmap = map_pos.shape[0]
    Nhalo = halo_pos.shape[0]
    halo_ids = np.empty(Nmap, dtype=np.int64)
    half = 0.5 * BoxSize
    for im in prange(Nmap):
        best = np.inf
        best_ih = -1
        for ih in range(Nhalo):
            d2 = 0.0
            for k in range(3):
                dx = abs(map_pos[im, k] - halo_pos[ih, k])
                if dx > half:
                    dx = BoxSize - dx
                d2 += dx * dx
            R = halo_R[ih]
            metric = d2 / (R * R)
            if metric < best:
                best = metric
                best_ih = ih
        halo_ids[im] = best_ih
    return halo_ids


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def _map_pos_pids_for_step(maplog, step, BoxSize):
    """MAP positions/pids for one timeslice step, wrapped into ``[0, BoxSize)``.

    ``control = 4*timestep + node_type``; timeslice nodes have ``node_type == 3``,
    and a single maplog (``read_maplog_dir``) Log tree spans many steps, so this
    just selects the one requested here (see ``map_step_for_redshift`` for
    resolving a target redshift to a step).
    """
    control_val = int(step) * 4 + 3
    msk = maplog['control'] == control_val
    print("Selected {:d} MAPs (control == {:d}).".format(int(np.count_nonzero(msk)),
                                                          control_val))
    map_pos = np.ascontiguousarray(maplog['pos'][msk, :], dtype=np.float64)
    map_pids = np.ascontiguousarray(maplog['pid'][msk], dtype=np.uint64)
    # wrap into [0, BoxSize) to share TNG's frame (as the original did)
    map_pos = (map_pos + BoxSize) % BoxSize
    return map_pos, map_pids


def load_tng_halos(basePath, snapno, mass_min=1e5):
    """Load TNG halos (FoF groups) used as match targets.

    Returns the original group index, positions [Mpc], a per-halo radius [Mpc]
    used to normalize the match metric (``Group_R_Mean200``), and a per-halo
    size N (dm particle count).  Unlike subhalos there is no ``SubhaloFlag``
    equivalent, so flag filtering is not applied.

    Only halos whose stellar mass exceeds ``mass_min`` (in solar masses) are
    kept, to avoid matching against an unwieldy number of tiny halos.  TNG
    group masses are reported in units of 1e10 Msun, so the cut is
    ``GroupMassType[:, star_type] > mass_min / 1e10``.
    """
    dm_type = il.snapshot.partTypeNum('dm')
    star_type = il.snapshot.partTypeNum('star')
    fields = ['GroupPos', 'Group_R_Mean200', 'GroupLenType', 'GroupMassType']
    grp = il.groupcat.loadHalos(basePath, snapno, fields=fields)

    keep = grp['GroupMassType'][:, star_type] > (mass_min / 1e10)
    halo_idx = np.ascontiguousarray(np.nonzero(keep)[0], dtype=np.int64)
    halo_pos = np.ascontiguousarray(grp['GroupPos'][keep] / 1e3, dtype=np.float64)
    halo_R = np.ascontiguousarray(grp['Group_R_Mean200'][keep] / 1e3, dtype=np.float64)
    halo_N = np.ascontiguousarray(grp['GroupLenType'][keep, dm_type], dtype=np.int64)
    print("Keeping {:d}/{:d} halos with stellar mass > {:.1e} Msun.".format(
        halo_idx.size, grp['count'], mass_min))
    # guard against zero-radius halos (would blow up the dist/R metric)
    halo_R[halo_R <= 0.0] = np.median(halo_R[halo_R > 0.0])
    return halo_idx, halo_pos, halo_R, halo_N


# --------------------------------------------------------------------------- #
# Main matcher (single process; compare_pos is thread-parallel over all cores)
# --------------------------------------------------------------------------- #
def make_map_tng_galaxy_dict_pos(basePath, snapno, map_pos, map_pids, BoxSize,
                                 mass_min=1e5, out_path=None, recalc=False):
    """Build a {group_id: MapGalaxyRec} dict matching every MAP to a TNG halo.

    Parameters
    ----------
    basePath, snapno : TNG simulation base path and snapshot number.
    map_pos : (Nmap, 3) array of MAP positions (correct control value, already
        wrapped into the box).
    map_pids : (Nmap,) array of the corresponding MAP pids.
    BoxSize : box size in Mpc (matching ``map_pos``/TNG units).
    mass_min : keep only halos above this mass [Msun] as match targets.
    out_path : optional pickle path for the resulting dict.
    recalc : if False and ``out_path`` exists, load and return it instead.
    """
    if out_path is not None and os.path.isfile(out_path) and not recalc:
        print("Loading existing dict from", out_path)
        with open(out_path, 'rb') as fb:
            return pickle.load(fb)

    print("Loading TNG halos...")
    halo_idx, halo_pos, halo_R, halo_N = load_tng_halos(basePath, snapno, mass_min)

    Nmap = map_pos.shape[0]
    print("Searching {:d} MAPs against {:d} halos...".format(Nmap, halo_pos.shape[0]))
    map_pos = np.ascontiguousarray(map_pos, dtype=np.float64)
    ids = compare_pos(map_pos, halo_pos, halo_R, float(BoxSize))

    gm_dict = dict()
    for map_pid, ih in zip(map_pids, ids):
        # ih indexes the (mass-cut) catalog; key the dict by the true group id
        group_id = int(halo_idx[ih])
        if group_id in gm_dict:
            gm_dict[group_id].append(map_pid)
        else:
            gm_dict[group_id] = MapGalaxyRec(int(halo_N[ih]), map_pid)
    print("Matched {:d} MAPs onto {:d} distinct halos.".format(Nmap, len(gm_dict)))

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with open(out_path, 'wb') as fp:
            pickle.dump(gm_dict, fp, protocol=2)
        print("Wrote", out_path)
    return gm_dict


# --------------------------------------------------------------------------- #
# Multi-redshift entry point (meant to be called once from a run script)
# --------------------------------------------------------------------------- #
def match_maps_to_tng(mapdir, basePath, redshifts, mass_min=1e5, out_dir=None,
                      recalc=False, ztol=5e-3):
    """Match MAPs to TNG halos at one or more redshifts in a single call.

    The maplog directory is read *once* (``read_maplog_dir``, whose Log tree
    spans the whole run's history) and reused for every requested redshift,
    so a run script only needs one process/one ``abacus`` import regardless
    of how many redshifts it matches at.

    Parameters
    ----------
    mapdir : str
        Simulation maplog directory (as passed to ``read_maplog_dir``).
    basePath : str
        TNG group-catalog basePath.
    redshifts : float or sequence of float
        Target redshift(s) to match at; each is resolved to its nearest MAP
        timeslice step (``map_step_for_redshift``) and nearest available TNG
        snapshot (``get_available_tng_snapshots``).
    mass_min : float
        Passed to ``load_tng_halos``: only match against halos above this
        stellar mass [Msun].
    out_dir : str, optional
        If given, one pickle per redshift is written to
        ``<out_dir>/gm_dict_z<redshift:.3f>.pkl``.
    recalc : bool
        If False and a redshift's pickle already exists under ``out_dir``, it
        is loaded instead of recomputed.
    ztol : float
        Max redshift difference allowed between a requested redshift and the
        closest available TNG snapshot (as in ``plot_correlation_functions``).

    Returns
    -------
    dict mapping each requested redshift (float) to its
    ``{group_id: MapGalaxyRec}`` gm_dict.
    """
    redshifts = np.atleast_1d(redshifts).astype(float)

    maplog = read_maplog_dir(mapdir, compute_pids=True, nworkers=8)
    BoxSize = maplog['header']['BoxSize']
    tng_snapnos, tng_zs = get_available_tng_snapshots(basePath)

    gm_dicts = {}
    for z_req in redshifts:
        step, map_z = map_step_for_redshift(maplog, z_req)
        k = int(np.argmin(np.abs(tng_zs - z_req)))
        if abs(tng_zs[k] - z_req) > ztol:
            raise ValueError(
                "requested redshift {:.3f} has no TNG snapshot within {:g} "
                "(2 decimal places); available TNG redshifts are {}".format(
                    z_req, ztol, ', '.join('{:.2f}'.format(z) for z in tng_zs)))
        snapno = int(tng_snapnos[k])

        print("z={:.3f}: MAP step {:d} (z={:.3f}), TNG snapshot {:d} (z={:.3f})".format(
            z_req, step, map_z, snapno, tng_zs[k]))
        map_pos, map_pids = _map_pos_pids_for_step(maplog, step, BoxSize)

        out_path = None
        if out_dir is not None:
            out_path = os.path.join(out_dir, 'gm_dict_z{:.3f}.pkl'.format(z_req))

        gm_dicts[float(z_req)] = make_map_tng_galaxy_dict_pos(
            basePath, snapno, map_pos, map_pids, BoxSize,
            mass_min=mass_min, out_path=out_path, recalc=recalc)

    return gm_dicts


def make_HOD_plot(gm_dict_dir, maplog, basePath, redshifts, str_identifier="",
                  abundances=(5.01e-3,), figsize=(12, 9), save=False, ztol=5e-3,
                  plot_ratios=False, save_columns=False):
    """Plot the HOD (``<N>(M_h)``) of MAPs vs TNG galaxies, one figure set per redshift.

    This is the new-data-model replacement for the old ``hm_dict``/``maps_tup``/
    ``map_dict`` interface, made redshift-aware to match ``match_maps_to_tng``:
    a single ``maplog`` (``read_maplog_dir(..., compute_pids=True)``, whose Log tree spans every
    requested redshift) and a single TNG ``basePath`` are reused for every
    redshift, and each entry in ``redshifts`` is independently resolved to its
    MAP timeslice step (``map_step_for_redshift``) and nearest TNG snapshot
    (``get_available_tng_snapshots``, within ``ztol``) -- exactly as
    ``match_maps_to_tng`` resolves them when producing the per-redshift
    ``gm_dict`` pickles this function reads back in.

    ``gm_dict_dir`` : str
        Directory of ``gm_dict_z<redshift:.3f>.pkl`` pickles as written by
        ``match_maps_to_tng`` (each a ``{group_id: MapGalaxyRec}`` dict, keyed
        by the *true* TNG group/FoF-halo id; ``rec.map_pids`` gives the MAPs
        matched to that halo).
    ``maplog`` : dict
        The dict returned by ``acj.utilities.read_maplog_dir`` -- provides
        ``header['BoxSize']`` plus the per-MAP multiplicity (``mult``) of whichever
        timeslice each requested redshift resolves to, used to impose the
        abundance cut. (``hm_dict`` no longer exists; the per-MAP masses that
        used to live in ``maps_tup``/``map_dict`` are now just the maplog
        ``mult`` column.)
    ``redshifts`` : float or sequence of float
        Target redshift(s); each gets its own pair of figures.

    Galaxy selection is done by *abundance matching* rather than fixed mass cuts:
    for each entry in ``abundances`` (one panel-column each) the MAP multiplicity
    threshold and the TNG stellar-mass threshold are derived with
    ``get_threshold`` so both samples share the same number density -- the same
    convention used in ``plot_window`` and ``analysis.ipynb``.

    For each redshift, two figures are produced: one for the means (``<N_cen>``,
    ``<N_sat>``, ``<N_cen>+<N_sat>``) and one for the satellite second moment
    (``sat_sig``). If ``plot_ratios`` is True, each figure also gets a narrow
    ratio subplot (1/4 the height of the main panel) directly underneath
    showing MAP / TNG for every curve in the panel above it; otherwise each
    figure is just the single top panel.

    ``plot_ratios`` : bool
        If True, add the MAP/TNG ratio subplot described above beneath each
        panel. If False (default), only the main panel is drawn.

    ``save_columns`` : bool
        If True, also make a separate, standalone figure for every abundance
        (its main panel plus the ratio subplot, if any) and save it to
        ``../Images/HOD_{mean,sig}_z<z>_n<abundance>[<str_identifier>].png``.
        Independent of ``save``.

    Returns
    -------
    dict mapping each requested redshift (float) to its
    ``(centrals, satellites, centrals_tng, satellites_tng)`` for the last
    entry in ``abundances``.
    """
    redshifts = np.atleast_1d(redshifts).astype(float)

    if 'pid' not in maplog:
        raise KeyError("maplog has no 'pid'; read it with read_maplog_dir(..., compute_pids=True)")
    BoxSize   = maplog['header']['BoxSize'] * u.Mpc
    dm_type  = il.snapshot.partTypeNum('dm')
    star_type = il.snapshot.partTypeNum('star')

    tng_snapnos, tng_zs = get_available_tng_snapshots(basePath)

    def binned(centrals, satellites, bn, idcs):
        """Mean centrals/satellites and satellite second moment per mass bin."""
        cen_n   = np.zeros_like(bn)
        sat_n   = np.zeros_like(bn)
        sat_sig = np.zeros_like(bn)
        for i in range(len(bn)):
            msk = idcs == i
            if np.count_nonzero(msk) > 0:
                cen_n[i]   = np.mean(centrals[msk])
                sat_n[i]   = np.mean(satellites[msk])
                sat_sig[i] = np.mean(satellites[msk] * (satellites[msk] - 1)) ** 0.5
        sat_sig[sat_n != 0] /= sat_n[sat_n != 0]
        return cen_n, sat_n, sat_sig

    lw, ms = 2.5, 11.2
    Nplt = len(abundances)
    results = {}
    for z_req in redshifts:
        step, map_z = map_step_for_redshift(maplog, z_req)
        k = int(np.argmin(np.abs(tng_zs - z_req)))
        if abs(tng_zs[k] - z_req) > ztol:
            raise ValueError(
                "requested redshift {:.3f} has no TNG snapshot within {:g} "
                "(2 decimal places); available TNG redshifts are {}".format(
                    z_req, ztol, ', '.join('{:.2f}'.format(z) for z in tng_zs)))
        snapno = int(tng_snapnos[k])

        gm_dict_path = os.path.join(gm_dict_dir, 'gm_dict_z{:.3f}.pkl'.format(z_req))
        with open(gm_dict_path, "rb") as f:
            gm_dict = pickle.load(f)

        # ---- MAPs: build pid -> multiplicity for this redshift's timeslice ----
        # control = 4*step + node_type; node_type 3 -> timeslice nodes.
        timeslice = maplog['control'] == step * 4 + 3
        map_mult  = maplog['mult'][timeslice]
        pid2mult  = dict(zip(maplog['pid'][timeslice], map_mult))

        # ---- TNG halos (FoF groups): dm masses + subhalo bookkeeping ----
        # gm_dict is keyed by the true group id, so we load the *full* catalog and
        # leave unmatched halos at zero galaxies (they still anchor the <N> average).
        halo_masses = il.groupcat.loadHalos(basePath, snapno,
                                            fields=['GroupMassType'])[:, dm_type] * 1e10
        gfs = il.groupcat.loadHalos(basePath, snapno, fields=['GroupFirstSub'])
        gns = il.groupcat.loadHalos(basePath, snapno, fields=['GroupNsubs'])
        Nhalo = len(halo_masses)

        # full subhalo list (GroupFirstSub/GroupNsubs index into it, flagged ones included)
        sub_flag = il.groupcat.loadSubhalos(basePath, snapno, fields=['SubhaloFlag'])
        sub_mass = il.groupcat.loadSubhalos(basePath, snapno,
                                            fields=['SubhaloMassType'])[:, star_type]

        nrows_fig = 2 if plot_ratios else 1
        gridspec_kw = {'wspace': 0.05, 'hspace': 0.}
        if plot_ratios:
            gridspec_kw['height_ratios'] = [3, 1]
        fig_mean, ax_mean = plt.subplots(nrows_fig, Nplt, figsize=figsize, sharex=True, squeeze=False,
                                         gridspec_kw=gridspec_kw)#, layout='constrained)
        fig_sig, ax_sig = plt.subplots(nrows_fig, Nplt, figsize=figsize, sharex=True, squeeze=False,
                                       gridspec_kw=gridspec_kw)#, layout='constrained')
        fig_mean.suptitle('z={:.2f}'.format(map_z), y=1.0, va='bottom')   # above the subplot titles
        fig_sig.suptitle('z={:.2f}'.format(map_z), y=1.0, va='bottom')   # above the subplot titles

        # mass binning over all (positive-mass) halos in this redshift's catalog
        bins = np.logspace(np.log10(halo_masses[halo_masses > 0].min()),
                           np.log10(halo_masses.max()), 25)
        bn   = bins[:-1] + np.diff(bins) / 2.
        idcs = np.digitize(halo_masses, bins) - 1

        for nplt, abundance in enumerate(abundances):
            # abundance-matched thresholds (same guesses as plot_window)
            map_thresh = get_threshold(map_mult, BoxSize, guess=1e1, abundance=abundance)
            tng_thresh = get_threshold(sub_mass, BoxSize, guess=1e-3, abundance=abundance)
            tng_valid  = (sub_flag == 1) & (sub_mass >= tng_thresh)

            # ---- MAP centrals/satellites per halo (keyed by true group id) ----
            centrals   = np.zeros(Nhalo)
            satellites = np.zeros(Nhalo)
            for gid, rec in gm_dict.items():
                mult = np.array([pid2mult[mp] for mp in rec.map_pids])
                nmap = np.count_nonzero(mult >= map_thresh)
                if nmap >= 1:
                    centrals[gid]   = 1
                    satellites[gid] = nmap - 1

            # ---- TNG centrals/satellites per halo ----
            centrals_tng   = np.zeros(Nhalo)
            satellites_tng = np.zeros(Nhalo)
            for i in range(Nhalo):
                if gfs[i] > -1:
                    msk = tng_valid[gfs[i]:gfs[i] + gns[i]]
                    if len(msk) > 0:
                        centrals_tng[i]   = msk[0]
                        satellites_tng[i] = np.count_nonzero(msk[1:])

            cen_n,   sat_n,   sat_sig     = binned(centrals, satellites, bn, idcs)
            cen_tng, sat_tng, sat_tng_sig = binned(centrals_tng, satellites_tng, bn, idcs)
            tot_n, tot_tng = cen_n + sat_n, cen_tng + sat_tng

            man, exp = get_sci_not(abundance)
            n_title = r'$n={:.2f}\times 10^{{{:d}}}\;[h/\mathrm{{Mpc}}]^3$'.format(man, exp)

            def draw(am, asg, first, title):
                """Plot this abundance onto a means column ``am`` and a sigma column ``asg``
                (each a [main, ratio] axes list); ``first`` puts the y labels on."""
                # ==== means figure ====
                # ---- top panel: <N_cen>, <N_sat> and their sum ----
                am[0].plot(bn, cen_tng, '.-', lw=lw, ms=ms, c='indianred')
                am[0].plot(bn, sat_tng, '*-', lw=lw, ms=ms, c='indianred')
                am[0].plot(bn, tot_tng, 'r-', lw=lw, ms=ms, label='TNG')
                am[0].plot(bn, cen_n, '.-', lw=lw, ms=ms, c='tab:blue')
                am[0].plot(bn, sat_n, '*-', lw=lw, ms=ms, c='tab:blue')
                am[0].plot(bn, tot_n, 'b-', lw=lw, ms=ms, label='MAP')

                # ==== sigma figure ====
                # ---- top panel: satellite second moment ----
                asg[0].plot(bn, sat_tng_sig, '*-', lw=lw, ms=ms, c='indianred', label='TNG')
                asg[0].plot(bn, sat_sig, '.-', lw=lw, ms=ms, c='tab:blue', label='MAP')

                if plot_ratios:
                    # MAP / TNG ratios for every curve, used in the ratio subplots below
                    with np.errstate(divide='ignore', invalid='ignore'):
                        cen_ratio = cen_n / cen_tng
                        sat_ratio = sat_n / sat_tng
                        tot_ratio = tot_n / tot_tng
                        sig_ratio = sat_sig / sat_tng_sig

                    # ---- ratio subplot: MAP / TNG for cen, sat, and their sum ----
                    am[1].axhline(1.0, color='gray', ls=':', lw=1)
                    am[1].plot(bn, cen_ratio, '.--', ms=ms, lw=lw * 0.8, c='gray', label='cen')
                    am[1].plot(bn, sat_ratio, '*--', ms=ms, lw=lw * 0.8, c='gray', label='sat')
                    am[1].plot(bn, tot_ratio, '--', ms=ms, lw=lw * 0.8, c='gray', label='total')

                    # ---- ratio subplot: MAP / TNG for the satellite second moment ----
                    asg[1].axhline(1.0, color='gray', ls=':', lw=1)
                    asg[1].plot(bn, sig_ratio, '--', ms=ms, lw=lw * 0.8, c='k')

                # ---- format: means figure ----
                am[0].grid()
                am[0].set(xscale='log', yscale='log', ylim=[1e-2, 1e2], xlim=[1e11, 1e15],
                                    xticks=[1e12, 1e14])
                am[0].set_title(title, fontsize=14)
                am[0].legend(loc='upper left')
                if plot_ratios:
                    am[0].tick_params(labelbottom=False)
                    am[1].set(xlabel=r'$M_h \;\; [h^{-1}M_{\odot}]$', xscale='log')
                    am[1].grid()
                    am[1].set_ylim([0.85, 1.15])
                else:
                    am[0].set(xlabel=r'$M_h \;\; [h^{-1}M_{\odot}]$')
                if first:
                    am[0].set_ylabel(r'$\langle N\rangle$')
                    #am[1].set_ylabel('MAP / TNG')
                    #am[1].legend(loc='upper left', fontsize='small')
                else:
                    am[0].tick_params(labelleft=False)
                    if plot_ratios:
                        am[1].tick_params(labelleft=False)

                # ---- format: sigma figure ----
                asg[0].grid()
                asg[0].set(xscale='log', ylim=[-0.15, 3.75], xlim=[1e11, 1e15],
                                   xticks=[1e12, 1e14])
                asg[0].legend(loc='upper left')
                if plot_ratios:
                    asg[0].tick_params(labelbottom=False)
                    asg[1].set(xlabel=r'$M_h \;\; [h^{-1}M_{\odot}]$', xscale='log')
                    asg[1].grid()
                    asg[1].set_ylim([0.85, 1.15])
                else:
                    asg[0].set(xlabel=r'$M_h \;\; [h^{-1}M_{\odot}]$')
                if first:
                    asg[0].set_ylabel(r'$\langle N^2-N \rangle/\langle N\rangle$')
                    #asg[1].set_ylabel('MAP / TNG')
                else:
                    asg[0].tick_params(labelleft=False)
                    if plot_ratios:
                        asg[1].tick_params(labelleft=False)

            draw(ax_mean[:, nplt], ax_sig[:, nplt], nplt == 0, n_title)
            if save_columns:
                # standalone figures: one abundance each, same panels as its grid column
                col_kw = {k: v for k, v in gridspec_kw.items() if k != 'wspace'}
                col_size = (figsize[0] / Nplt + 1, figsize[1])
                fig_cm, ax_cm = plt.subplots(nrows_fig, 1, figsize=col_size, sharex=True, squeeze=False,
                                             gridspec_kw=col_kw)
                fig_cs, ax_cs = plt.subplots(nrows_fig, 1, figsize=col_size, sharex=True, squeeze=False,
                                             gridspec_kw=col_kw)
                draw(ax_cm[:, 0], ax_cs[:, 0], True, f'z={map_z:.2f}, ' + n_title)
                for kind, fig_c in (('mean', fig_cm), ('sig', fig_cs)):
                    fig_c.savefig(f"../Images/HOD_{kind}_z{map_z:.2f}_n{abundance:.2e}" + str_identifier + '.png',
                                  bbox_inches='tight', transparent=True)

        if save:
            fig_mean.savefig(f"../Images/HOD_mean_z{map_z:.2f}" + str_identifier + '.png',
                             bbox_inches='tight', transparent=True)
            fig_sig.savefig(f"../Images/HOD_sig_z{map_z:.2f}" + str_identifier + '.png',
                            bbox_inches='tight', transparent=True)

        results[float(z_req)] = (centrals, satellites, centrals_tng, satellites_tng)

    plt.show()

    return results


def halowise_map_galaxies(maplog, gm_dict, halo_masses, bin_edges, abundance=5.01e-3,
                          redshift=0.2):
    """Central / satellite MAP positions per parent-halo mass bin, for one maplog.

    New-data-model replacement for the ``gal_map_dict``/``map_dict``/``maps_tup``
    bookkeeping of the old ``halo_wise_corr``: the MAP <-> halo association comes
    from ``gm_dict`` (``{group_id: MapGalaxyRec}`` produced by
    ``make_map_tng_galaxy_dict_pos``) and the per-MAP multiplicity / position
    come from the timeslice nodes of ``maplog`` (``read_maplog_dir(...,
    compute_pids=True)``) at the step nearest ``redshift`` (a single maplog spans
    every step of the run); ``gm_dict`` must have been matched at that redshift.

    Following the original, a MAP threshold is abundance-matched on multiplicity
    (``get_threshold``); within each matched halo the surviving MAPs are split
    into one *central* (the highest-multiplicity MAP) and the remaining
    *satellites*. Halos are binned by ``halo_masses[group_id]`` into ``bin_edges``.

    Returns ``(cen, sat)``: two lists of length ``len(bin_edges) - 1``; ``cen[b]``
    / ``sat[b]`` are lists of ``(3,)`` central / satellite positions in mass bin
    ``b``.
    """
    if 'pid' not in maplog:
        raise KeyError("maplog has no 'pid'; read it with read_maplog_dir(..., compute_pids=True)")
    BoxSize   = maplog['header']['BoxSize'] * u.Mpc
    step, _   = map_step_for_redshift(maplog, redshift)
    timeslice = maplog['control'] == step * 4 + 3
    mult = maplog['mult'][timeslice]
    pos, pids = _map_pos_pids_for_step(maplog, step, maplog['header']['BoxSize'])

    map_thresh = get_threshold(mult, BoxSize, guess=1e1, abundance=abundance)
    pid2idx    = dict(zip(pids, range(len(pids))))
    nbins      = len(bin_edges) - 1
    hbin       = np.digitize(halo_masses, bin_edges) - 1

    cen = [[] for _ in range(nbins)]
    sat = [[] for _ in range(nbins)]
    for gid, rec in gm_dict.items():
        b = hbin[gid]
        if b < 0 or b >= nbins:
            continue
        idx = np.array([pid2idx[mp] for mp in rec.map_pids if mp in pid2idx])
        if idx.size == 0:
            continue
        m    = mult[idx]
        keep = m >= map_thresh
        if np.count_nonzero(keep) < 1:
            continue
        idx, m = idx[keep], m[keep]
        i_cen  = np.argmax(m)
        cen[b].append(pos[idx[i_cen]])
        sat_idx = idx[np.arange(idx.size) != i_cen]
        if sat_idx.size:
            sat[b].extend(pos[sat_idx])
    return cen, sat


def halowise_tng_galaxies(basePath, snapno, halo_masses, bin_edges, BoxSize,
                          abundance=5.01e-3):
    """Central / satellite TNG-galaxy positions per parent-halo mass bin.

    The TNG analog of ``halowise_map_galaxies``: subhalos are abundance-matched
    on stellar mass (``get_threshold``), the ``GroupFirstSub`` subhalo of each
    group is its central and the remaining valid subhalos are satellites
    (resolved via ``SubhaloGrNr``). Positions are returned in Mpc/h. Returns
    ``(cen, sat)`` with the same shape as ``halowise_map_galaxies``.
    """
    star_type = il.snapshot.partTypeNum('star')
    gfs      = il.groupcat.loadHalos(basePath, snapno, fields=['GroupFirstSub'])
    sub_flag = il.groupcat.loadSubhalos(basePath, snapno, fields=['SubhaloFlag'])
    sub_mass = il.groupcat.loadSubhalos(basePath, snapno,
                                        fields=['SubhaloMassType'])[:, star_type]
    sub_grnr = il.groupcat.loadSubhalos(basePath, snapno, fields=['SubhaloGrNr'])
    sub_pos  = il.groupcat.loadSubhalos(basePath, snapno, fields=['SubhaloPos']) / 1e3

    tng_thresh = get_threshold(sub_mass, BoxSize, guess=1e-3, abundance=abundance)
    valid      = (sub_flag == 1) & (sub_mass >= tng_thresh)

    # the GroupFirstSub of every group is that group's central subhalo
    is_central = np.zeros(len(sub_flag), dtype=bool)
    is_central[gfs[gfs > -1]] = True

    nbins    = len(bin_edges) - 1
    sub_bin  = np.digitize(halo_masses, bin_edges)[sub_grnr] - 1  # each subhalo's halo bin
    cen_mask = valid & is_central
    sat_mask = valid & ~is_central

    cen = [[] for _ in range(nbins)]
    sat = [[] for _ in range(nbins)]
    for b in range(nbins):
        cen[b] = list(sub_pos[cen_mask & (sub_bin == b)])
        sat[b] = list(sub_pos[sat_mask & (sub_bin == b)])
    return cen, sat


def compare_halowise_correlation_functions(gm_tuples,
                                            basePath='/flare/Abacus/ajohnson/TNG300/TNG300-2/output/',
                                            snapno=None, redshift=0.2, labels=None,
                                            plot_tng=False,
                                            rmin=2e-2, rmax=30.0, nbins=30,
                                            type_str='gg', mass_min=8e11, mass_nbins=6,
                                            abundance=5.01e-3, min_size=10,
                                            max_p=np.inf, nthreads=26,
                                            cmaps=('Blues_r', 'Greens_r', 'Purples_r', 'Oranges_r'),
                                            clo=0., chi=0.7, panel_size=(4, 3),
                                            save=False, str_identifier='',
                                            map_default=None, default_label=''):
    """Compare halo-wise MAP correlation functions, one figure per halo mass bin.

    New-data-model adaptation of ``halo_wise_corr`` (scratch.py) cast in the mould
    of ``compare_correlation_functions``. Parent halos (TNG FoF groups) are binned
    by mass and *each mass bin gets its own figure*. Within a figure the grid is
    ``nrows x 1``: one row per tuple (exactly as ``compare_correlation_functions``),
    each showing the ``centrals & satellites`` galaxy sample.

    Inputs take the ``(gm_dict, maplog)`` pair form of ``plot_map_halo_mass_frac``
    (the ``gt`` of analysis.ipynb): ``gm_tuples[row][i]`` is a pair where

    ``gm_dict`` : ``{group_id: MapGalaxyRec}``
        Produced by ``make_map_tng_galaxy_dict_pos`` (keyed by the true TNG group
        id); ``rec.map_pids`` are the pids of the MAPs matched to that halo.
    ``maplog`` : dict
        ``read_maplog_dir`` output, whose final-timeslice nodes supply the per-MAP
        multiplicity and position.

    Each pair's MAP galaxies in a mass bin are abundance-matched on multiplicity,
    split into a central (highest-multiplicity MAP per halo) and satellites, and
    correlated; curves are drawn as ``r^2 xi(r)`` with a Poisson band, shaded per
    row from ``cmaps`` just like ``compare_correlation_functions``.

    Unlike ``compare_correlation_functions`` there is no redshift axis: a single
    ``redshift`` is plotted. Each maplog (a single one spans every step of the run,
    and must be read with ``compute_pids=True``) is taken at the timeslice nearest
    ``redshift``, and the TNG snapshot is the one nearest ``redshift`` unless
    ``snapno`` is given; the ``gm_dict``s must have been matched at that redshift
    (``match_maps_to_tng``).

    Parameters
    ----------
    gm_tuples : sequence of tuple of (gm_dict, maplog)
        One tuple per row; each tuple is a set of ``(gm_dict, maplog)`` pairs to
        overplot in that row's subplots.
    basePath, snapno :
        TNG group-catalog base path and snapshot number (parent halos + optional
        TNG overlay). ``snapno=None`` resolves it from ``redshift``; either way it
        must correspond to the snapshot the ``gm_dict``s were built against.
    redshift : float
        Redshift at which the maplogs' timeslices are taken.
    plot_tng : bool
        If True, overplot the TNG-galaxy correlation function (red dashed) in
        every subplot.
    rmin, rmax, nbins :
        Log-spaced separation binning (Mpc/h).
    mass_min, mass_nbins : float, int
        Parent-halo mass binning runs log-spaced from ``mass_min`` to the maximum
        TNG group mass in ``mass_nbins`` bins; one figure per bin.
    abundance : float
        Target number density used by ``get_threshold`` for both the MAP
        multiplicity and TNG stellar-mass selections.
    min_size : int
        Minimum number of galaxies in a (bin, sample) before its correlation
        function is drawn.
    map_default : (gm_dict, maplog), optional
        Reference ``(gm_dict, maplog)`` pair drawn as a gray dashed curve in every
        subplot (the new-data-model analogue of the ``map_default`` argument of
        ``compare_correlation_functions``).
    default_label : str
        Legend label for the gray reference.

    Other parameters (``labels``, ``cmaps``, ``clo``, ``chi``, ``panel_size``,
    ``max_p``, ``nthreads``, ``type_str``, ``save``, ``str_identifier``) behave as
    in ``compare_correlation_functions``.

    Returns
    -------
    (figs, axes_list) : lists with one ``(fig, axes)`` entry per mass bin.
    """
    nrows = len(gm_tuples)

    if snapno is None:
        tng_snapnos, tng_zs = get_available_tng_snapshots(basePath)
        snapno = int(tng_snapnos[np.argmin(np.abs(tng_zs - redshift))])

    if labels is not None:
        assert len(labels) == nrows, "labels must have one tuple per gm tuple (row)"
        for row, (pairs, labs) in enumerate(zip(gm_tuples, labels)):
            assert len(labs) == len(pairs), \
                "labels[{}] must match the number of pairs in row {}".format(row, row)

    BoxSize = gm_tuples[0][0][1]['header']['BoxSize'] * u.Mpc
    dm_type = il.snapshot.partTypeNum('dm')

    r_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
    r       = r_edges[:-1] + np.diff(r_edges) / 2.0

    # parent-halo masses (shared across all maplogs: every gm_dict keys the same
    # TNG groups) define the mass bins, one figure each.
    halo_masses = il.groupcat.loadHalos(basePath, snapno,
                                        fields=['GroupMassType'])[:, dm_type] * 1e10
    bin_edges = np.logspace(np.log10(mass_min), np.log10(halo_masses.max()), mass_nbins + 1)

    def _compute_xi(positions):
        if len(positions) <= min_size:
            return None
        twopt   = get_xi(np.asarray(positions, dtype=float), r_edges, BoxSize,
                         max_p=max_p, nthreads=nthreads)
        xi      = twopt.get_corr(return_sep=False)
        poisson = np.sqrt(twopt.D1D2.wcounts) / twopt.R1R2.wcounts
        return xi, poisson

    def _plot_xi(ax, xi_poisson, color, style, label):
        if xi_poisson is None:
            return
        xi, poisson = xi_poisson
        ax.plot(r, r**2 * xi, style, lw=2, color=color, label=label)
        ax.fill_between(r, r**2 * (xi - poisson), r**2 * (xi + poisson),
                        color=color, alpha=0.3)

    # ---- precompute central/satellite positions per mass bin (once each) ----
    map_cs = [[halowise_map_galaxies(maplog, gm_dict, halo_masses,
                                     bin_edges, abundance=abundance, redshift=redshift)
               for (gm_dict, maplog) in pairs]
              for pairs in gm_tuples]

    default_cs = None
    if map_default is not None:
        gm_d, ml_d = map_default
        default_cs = halowise_map_galaxies(ml_d, gm_d, halo_masses,
                                           bin_edges, abundance=abundance, redshift=redshift)

    tng_cs = None
    if plot_tng:
        tng_cs = halowise_tng_galaxies(basePath, snapno, halo_masses, bin_edges,
                                       BoxSize, abundance=abundance)

    def _sample(cs, b):
        """centrals + satellites for bin b."""
        cen, sat = cs
        return cen[b] + sat[b]

    # ---- one figure per mass bin, plus a single figure with a single axes
    # collecting the sqrt(xi_MAP / xi_TNG) ratio curves across all mass bins ----
    figs = [None] * mass_nbins
    axes_list = [None] * mass_nbins

    # (label, xi_map, xi_tng) entries accumulated across every mass bin
    all_ratio_entries = []

    for b in range(mass_nbins):
        fig, axes = plt.subplots(nrows, 1, sharex=True, sharey=True, squeeze=False,
                                 figsize=(panel_size[0], panel_size[1] * nrows))
        fig.subplots_adjust(wspace=0, hspace=0)
        figs[b], axes_list[b] = fig, axes

        default_xi = _compute_xi(_sample(default_cs, b)) if map_default is not None else None
        tng_xi     = _compute_xi(_sample(tng_cs, b)) if plot_tng else None

        # curves to compare against TNG in the ratio figure: (label, xi_poisson)
        ratio_curves = []
        if default_xi is not None:
            ratio_curves.append((default_label, default_xi))

        for row, pairs in enumerate(gm_tuples):
            cmap  = colormaps[cmaps[row % len(cmaps)]]
            n_map = len(pairs)
            fracs = clo + (chi - clo) * np.linspace(0, 1, max(n_map, 1))
            ax = axes[row][0]

            _plot_xi(ax, default_xi, 'gray', '--', default_label)

            for m in range(n_map):
                label = str(labels[row][m]) if labels is not None else ''
                map_xi = _compute_xi(_sample(map_cs[row][m], b))
                _plot_xi(ax, map_xi, cmap(fracs[m]), '-', label)
                if map_xi is not None:
                    ratio_curves.append((label, map_xi))

            _plot_xi(ax, tng_xi, 'red', '--', 'TNG')

            ax.set(xscale='log')
            ax.grid()
            ax.legend(fontsize='small')

        fig.suptitle(r'$M_h \in [{:.2e}, {:.2e}]\;M_\odot$'.format(bin_edges[b], bin_edges[b + 1]))
        for ax in axes[-1, :]:
            ax.set_xlabel(r'$r\;\;[h^{-1}Mpc]$')
        for ax in axes[:, 0]:
            ax.set_ylabel(r'$r^2\xi_{%s}(r)$' % type_str)

        if save:
            fig.savefig('../Images/halo_xi_{:d}_'.format(b) + str_identifier
                        + '.png', bbox_inches='tight', transparent=True)

        # stash this bin's curves (with mass range baked into the label) for the
        # shared single-axes ratio figure built after the loop
        if tng_xi is not None and len(ratio_curves) > 0:
            xi_tng, _ = tng_xi
            bin_str = r'$M_h \in [{:.1e}, {:.1e}]$'.format(bin_edges[b], bin_edges[b + 1])
            for label, (xi_map, _) in ratio_curves:
                full_label = '{} ({})'.format(label, bin_str) if label else bin_str
                all_ratio_entries.append((full_label, xi_map, xi_tng))

    rfig = None
    if len(all_ratio_entries) > 0:
        rfig, rax = plt.subplots(figsize=panel_size)

        inferno = colormaps['inferno']
        rfracs  = np.linspace(0.05, 0.95, len(all_ratio_entries))
        for (label, xi_map, xi_tng), frac in zip(all_ratio_entries, rfracs):
            rax.plot(r, np.sqrt(xi_map / xi_tng), '-', lw=2,
                    color=inferno(frac), label=label)

        rax.axhline(1.0, color='gray', ls=':', lw=1)
        rax.set(xscale='log', ylim=(0.85, 1.15))
        rax.set_xlabel(r'$r\;\;[h^{-1}Mpc]$')
        rax.grid()
        rax.set_ylabel(r'$\sqrt{\dfrac{\xi_{gg,MAP}}{\xi_{gg,TNG}}}$',fontsize='small')
        #rax.text(0.03, 0.95, r'$\sqrt{\dfrac{\xi_{gg,MAP}}{\xi_{gg,TNG}}}$',
        #         transform=rax.transAxes, ha='left', va='top', fontsize='small')

        if save:
            rfig.savefig('../Images/halo_xi_err_' + str_identifier
                        + '.png', bbox_inches='tight', transparent=True)

    plt.show()
    return figs, axes_list, rfig


def compare_velocity_distributions(maplog, redshift=0.2, label='MAP',
                                   basePath='/flare/Abacus/ajohnson/TNG300/TNG300-2/output/',
                                   gm_template='../Images/{sim}_tng/gm_dict_z{z:.3f}.pkl',
                                   abundance=5.01e-3, nbins=60, ztol=5e-3,
                                   panel_size=(5, 2.4), save=False, str_identifier=''):
    """Histograms of MAP and TNG central / satellite velocities, per component and speed.

    One abundance-matched sample of each, at one redshift (``redshift`` resolved to the
    nearest MAP timeslice and TNG snapshot):

    - TNG galaxies: stellar-mass-thresholded subhalos; a group's ``GroupFirstSub``
      is its central, every other valid subhalo a satellite. ``SubhaloVel`` is
      used as is.
    - MAPs: multiplicity-thresholded timeslice nodes, split with the MAP -> TNG halo
      match (``gm_template``, see ``map_satellite_mask``): in each halo the
      highest-multiplicity MAP is the central, the rest satellites. Needs
      ``compute_pids=True``.

    Layout: column 0 centrals, column 1 satellites; rows ``v_x, v_y, v_z`` and the speed
    ``|v|``, all as probability densities. With ``save``, the grid is saved and each
    column is also made and saved as its own figure
    (``compare_velocity_distributions_{central,satellite}_...png``). Each panel overplots TNG (red), the MAPs'
    reported velocity (blue) and their reconstructed dark-matter velocity
    ``vel + rel_vel_vector`` (blue dashed; the velocity ``compare_zspace_correlation_functions``
    uses for MAP centrals). Legends give each sample's standard deviation of the
    component (mean speed for ``|v|``).

    Returns ``(fig, axes, data)`` with ``data[sample][population]`` the ``(N, 3)``
    velocity arrays, for samples ``'TNG'``, ``'MAP'``, ``'MAP (DM)'`` and populations
    ``'central'``, ``'satellite'``.
    """
    # ---- MAPs ----
    BoxSize   = maplog['header']['BoxSize'] * u.Mpc
    step, _   = map_step_for_redshift(maplog, redshift)
    timeslice = maplog['control'] == int(step) * 4 + 3
    thresh    = get_threshold(maplog['mult'][timeslice], BoxSize, guess=1e1, abundance=abundance)
    msk       = timeslice & (maplog['mult'] >= thresh)
    vel_map   = maplog['vel'][msk, :]
    vel_dm    = vel_map + rel_vel_healpix_to_vector(maplog['rel_vel_healpix'][msk], maplog['rel_vel'][msk])
    gm_dict   = read_gm_dict_for_maplog(maplog, redshift, gm_template)
    is_sat    = map_satellite_mask(maplog['pid'][msk], maplog['mult'][msk], gm_dict)

    # ---- TNG ----
    snapnos, zs = get_available_tng_snapshots(basePath)
    k = np.argmin(np.abs(zs - redshift))
    if abs(zs[k] - redshift) > ztol:
        raise ValueError(f"requested redshift {redshift:.3f} has no TNG snapshot within {ztol:g}")
    snapno    = int(snapnos[k])
    star_type = il.snapshot.partTypeNum('star')
    gfs       = il.groupcat.loadHalos(basePath, snapno, fields=['GroupFirstSub'])
    sub       = il.groupcat.loadSubhalos(basePath, snapno,
                                         fields=['SubhaloMassType', 'SubhaloFlag', 'SubhaloVel'])
    sub_mass  = sub['SubhaloMassType'][:, star_type]
    tng_vel   = sub['SubhaloVel']
    tng_box   = il.groupcat.loadHeader(basePath, snapno)['BoxSize'] / 1e3 * u.Mpc
    tng_thresh = get_threshold(sub_mass, tng_box, guess=1e-3, abundance=abundance)
    valid     = (sub['SubhaloFlag'] == 1) & (sub_mass >= tng_thresh)
    is_first  = np.zeros(len(sub_mass), dtype=bool)
    is_first[gfs[gfs > -1]] = True

    data = {'TNG':      {'central': tng_vel[valid & is_first],  'satellite': tng_vel[valid & ~is_first]},
            'MAP':      {'central': vel_map[~is_sat],           'satellite': vel_map[is_sat]},
            'MAP (DM)': {'central': vel_dm[~is_sat],            'satellite': vel_dm[is_sat]}}
    styles = [('TNG', dict(color='red', ls='-')), ('MAP', dict(color='tab:blue', ls='-')),
              ('MAP (DM)', dict(color='tab:blue', ls='--'))]
    names  = {'TNG': 'TNG', 'MAP': label, 'MAP (DM)': label + ' DM'}

    allv = np.concatenate([d for pops in data.values() for d in pops.values()])
    vmax = np.percentile(np.abs(allv), 99.5)
    comp_edges  = np.linspace(-vmax, vmax, nbins + 1)
    speed_edges = np.linspace(0, np.sqrt(3) * vmax, nbins + 1)

    comp_labels = (r'$v_x$', r'$v_y$', r'$v_z$', r'$|v|$')

    def draw(axs, pop):
        """Draw one population's four panels (v_x, v_y, v_z, |v|) onto the four axes ``axs``."""
        for col, (ax, name) in enumerate(zip(axs, comp_labels)):
            for key, style in styles:
                v = data[key][pop]
                x, edges = (v[:, col], comp_edges) if col < 3 else (np.linalg.norm(v, axis=1), speed_edges)
                stat = f'{np.mean(x):.0f}' if col == 3 else f'{np.std(x):.0f}'
                sym  = r'\langle|v|\rangle' if col == 3 else r'\sigma'
                ax.hist(x, bins=edges, density=True, histtype='step', lw=2, **style,
                        label=f'{names[key]} (${sym}$={stat})')
            ax.grid()
            ax.set_ylim(0, ax.get_ylim()[1] * 1.7)              # headroom for the legend above the peaks
            ax.legend(fontsize='x-small', loc='upper right', handlelength=1.5)
            ax.set_yticklabels([])
            ax.set_xlabel(name + ' [km/s]')
        axs[0].set_title(f"{pop}s")

    pops = ('central', 'satellite')
    fig, axes = plt.subplots(4, 2, figsize=(panel_size[0] * 2, panel_size[1] * 4), squeeze=False)
    for col, pop in enumerate(pops):
        draw(axes[:, col], pop)
    fig.tight_layout()
    fig.suptitle(f'z = {redshift:.2f}', y=1.01, va='bottom')
    ids = f'_z{redshift:.2f}_n{abundance:.2e}' + str_identifier
    if save:
        fig.savefig('../Images/compare_velocity_distributions' + ids + '.png',
                    bbox_inches='tight', transparent=True)
        for pop in pops:                    # each column again as its own figure
            fig_c, axes_c = plt.subplots(4, 1, figsize=(panel_size[0], panel_size[1] * 4), squeeze=False)
            draw(axes_c[:, 0], pop)
            fig_c.tight_layout()
            fig_c.savefig(f'../Images/compare_velocity_distributions_{pop}' + ids + '.png',
                          bbox_inches='tight', transparent=True)
    plt.show()
    return fig, axes, data


def compare_central_satellite_correlation_functions(
        maplog_tuples, redshift=0.2, labels=None, row_names=None,
        basePath='/flare/Abacus/ajohnson/TNG300/TNG300-2/output/', snapno=None,
        gm_template='../Images/{sim}_tng/gm_dict_z{z:.3f}.pkl',
        abundance=1.01e-2, min_halo_mass=1e12,
        rmin=1e-1, rmax=40.0, nbins=12, type_str=r'cen,sat',
        max_p=np.inf, nthreads=26,
        cmaps=('Blues_r', 'Greens_r', 'Purples_r', 'Oranges_r'), clo=0., chi=0.7,
        panel_size=(5, 2.6), save=False, str_identifier='',
        map_default=None, default_label='', plot_ratio=False,
        xscale='log', ylim=None, yticks=None):
    """Compare MAP central-satellite cross-correlation functions across maplogs.

    Same interface as ``compare_correlation_functions`` (tuples of maplogs, one
    row per tuple, ``labels``/``row_names``, per-row colormap shading,
    ``map_default`` gray-dashed reference, ``plot_ratio``, ``ylim``/``yticks``),
    with a single ``redshift`` instead of a column per redshift and every curve a
    central x satellite cross-correlation ``r^2 xi_{cen,sat}(r)`` instead of an
    autocorrelation.

    For each maplog the MAPs are matched to TNG halos by its ``gm_dict`` (found
    from the maplog's own ``header['SimName']``, see ``read_gm_dict_for_maplog``;
    made beforehand with ``match_maps_to_tng`` at this ``redshift``). Considering
    only halos above ``min_halo_mass`` (dark-matter mass), MAPs are abundance-
    matched on multiplicity (``abundance``) and, within each matched halo, the
    highest-multiplicity MAP is its *central* and the rest *satellites*
    (``halowise_map_galaxies`` with one halo-mass bin ``[min_halo_mass, inf)``).
    The curve is the cross-correlation of all centrals with all satellites, with a
    Poisson band.

    Parameters
    ----------
    maplog_tuples : sequence of tuple of dict
        One tuple per row; ``read_maplog_dir(..., compute_pids=True)`` outputs (the
        pids connect MAPs to ``gm_dict``).
    redshift : float
        Resolves both the MAP timeslice and the TNG snapshot (unless ``snapno``).
    basePath, snapno :
        TNG group catalog supplying the parent-halo masses; must be the one the
        gm_dicts were matched against.
    gm_template : str
        Path template for the gm_dict, formatted with ``sim`` and ``z``.
    abundance, min_halo_mass : float
        MAP number-density cut and minimum parent-halo mass.
    map_default : dict, optional
        A maplog drawn as a gray dashed reference in every row; also the
        denominator when ``plot_ratio`` (which requires it). Its curve is computed once.
    plot_ratio : bool
        Plot each curve divided by ``map_default``'s (reference drawn at 1 with
        +/-10% dotted lines). Saved figure names get ``_ratio`` appended.
    labels, row_names, rmin, rmax, nbins, type_str, max_p, nthreads, cmaps, clo,
    chi, panel_size, save, str_identifier, xscale, ylim, yticks :
        As in ``compare_correlation_functions``. ``yticks`` defaults to automatic
        (three ticks symmetric about 1 in ratio mode, whose default y-range is
        (0.7, 1.3)). Saved to ``Images/compare_central_satellite_correlation_functions_n<abundance>_<str_identifier>[_ratio].png``.

    Returns
    -------
    (fig, axes, data) : ``data['curves'][row][m] = (xi, poisson)`` for each curve
        and ``data['default']`` for ``map_default`` (or None), so results can be re-plotted.
    """
    nrows = len(maplog_tuples)
    if plot_ratio and map_default is None:
        raise ValueError("plot_ratio=True requires map_default to provide the reference curve")
    if labels is not None:
        assert len(labels) == nrows, "labels must have one tuple per maplog tuple (row)"
        for row, (mls, labs) in enumerate(zip(maplog_tuples, labels)):
            assert len(labs) == len(mls), f"labels[{row}] must match the maplogs in row {row}"
    if row_names is not None:
        assert len(row_names) == nrows, "row_names must have one entry per maplog tuple (row)"

    if snapno is None:
        tng_snapnos, tng_zs = get_available_tng_snapshots(basePath)
        snapno = int(tng_snapnos[np.argmin(np.abs(tng_zs - redshift))])
    # gm_dict is keyed by the true TNG group id, so index the full halo catalog by gid
    dm_type     = il.snapshot.partTypeNum('dm')
    halo_masses = il.groupcat.loadHalos(basePath, snapno, fields=['GroupMassType'])[:, dm_type] * 1e10

    r_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
    r       = r_edges[:-1] + np.diff(r_edges) / 2.0

    def _xi(maplog):
        gm_dict  = read_gm_dict_for_maplog(maplog, redshift, gm_template)
        cen, sat = halowise_map_galaxies(maplog, gm_dict, halo_masses, [min_halo_mass, np.inf],
                                         abundance=abundance, redshift=redshift)
        twopt = get_xi(np.array(cen[0]), r_edges, maplog['header']['BoxSize'] * u.Mpc,
                       pos2=np.array(sat[0]), max_p=max_p, nthreads=nthreads)
        return twopt.get_corr(return_sep=False), np.sqrt(twopt.D1D2.wcounts) / twopt.R1R2.wcounts

    default = _xi(map_default) if map_default is not None else None   # computed once
    data    = [[_xi(m) for m in maplogs] for maplogs in maplog_tuples]

    fig, axes = plt.subplots(nrows, 1, sharex=True, sharey=True, squeeze=False,
                             figsize=(panel_size[0], panel_size[1] * nrows))
    fig.subplots_adjust(hspace=0)
    axes = axes[:, 0]
    ymax = 0.0
    for row, maplogs in enumerate(maplog_tuples):
        ax    = axes[row]
        cmap  = colormaps[cmaps[row % len(cmaps)]]
        fracs = clo + (chi - clo) * np.linspace(0, 1, max(len(maplogs), 1))
        if map_default is not None:
            if plot_ratio:
                ax.axhline(1.0, ls='--', lw=2, color='gray', label=default_label or '_nolegend_')
                ax.axhline(0.9, ls=':', lw=2.5, color='gray', alpha=0.8)
                ax.axhline(1.1, ls=':', lw=2.5, color='gray', alpha=0.8)
            else:
                y_d, yerr_d = _xi_plot_values(r, *default)
                ax.plot(r, y_d, '--', lw=2, color='gray', label=default_label or '_nolegend_')
                ax.fill_between(r, y_d - yerr_d, y_d + yerr_d, color='gray', alpha=0.3)
                ymax = max(ymax, np.nanmax(y_d + yerr_d))
        for m in range(len(maplogs)):
            xi, poisson = data[row][m]
            y, yerr = _xi_plot_values(r, xi, poisson, plot_ratio, default if plot_ratio else None)
            label = str(labels[row][m]) if labels is not None else '_nolegend_'
            ax.plot(r, y, '-', lw=2, color=cmap(fracs[m]), label=label)
            ax.fill_between(r, y - yerr, y + yerr, color=cmap(fracs[m]), alpha=0.3)
            if not plot_ratio:
                ymax = max(ymax, np.nanmax(y + yerr))
        ax.set(xscale=xscale)
        ax.grid()
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize='small')
        ylab = (r'$\xi_{%s}/\xi_{%s,\rm ref}$' % (type_str, type_str)) if plot_ratio \
               else r'$r^2\xi_{%s}$ [$h^{-2}Mpc^2$]' % type_str
        if row_names is not None:
            ylab = f'{row_names[row]}\n{ylab}'
        ax.set_ylabel(ylab, fontsize='small' if row_names is not None else None)
    # y is shared across all rows, so one limit set covers them
    if plot_ratio:
        axes[0].set_ylim(0.7, 1.3)
    else:
        axes[0].set_ylim(0, ymax * 1.05)
    if ylim is not None:
        axes[0].set_ylim(ylim)
    lo, hi = axes[0].get_ylim()
    if yticks is not None:
        axes[0].set_yticks(yticks)
    elif plot_ratio:
        d = 0.2 if (hi - lo) / 2 > 0.2 else 0.1 if (hi - lo) / 2 > 0.1 else 0.05
        axes[0].set_yticks([1 - d, 1.0, 1 + d])
    axes[0].set_title(f'z = {redshift:.2f}')
    axes[-1].set_xlabel(r'r [$h^{-1}Mpc$]')

    if save:
        fig.savefig('../Images/compare_central_satellite_correlation_functions_n{:.2e}_'.format(abundance)
                    + str_identifier + _ratio_suffix(plot_ratio) + '.png',
                    bbox_inches='tight', transparent=True)
    plt.show()
    return fig, axes, {'curves': data, 'default': default}


def _map_halo_mass_frac_curves(gm_dict, maplog, halo_masses, edges, Mp_val,
                               abundance=None, BoxSize=None, redshift=0.2):
    """Per-mass-bin central and total MAP-mass fractions for one (gm_dict, maplog).

    ``central`` is the largest MAP "mass" (``mult * Mp``) in a halo divided by the
    halo mass; ``total`` is the sum over all the halo's MAPs divided by the halo
    mass. Returns ``(cen_mean, cen_std, tot_mean, tot_std)``, each an array over
    the ``len(edges) - 1`` mass bins (NaN where a bin has no matched halo).
    """
    # pid -> multiplicity at the timeslice nearest `redshift` (control =
    # 4*step + 3 selects that step's timeslice nodes; needs compute_pids=True).
    if 'pid' not in maplog:
        raise KeyError("maplog has no 'pid'; read it with read_maplog_dir(..., compute_pids=True)")
    step, _   = map_step_for_redshift(maplog, redshift)
    timeslice = maplog['control'] == step * 4 + 3
    pid2mult  = dict(zip(maplog['pid'][timeslice], maplog['mult'][timeslice]))

    nbin = len(edges) - 1
    idcs = np.digitize(halo_masses, edges)

    # gather each in-range halo's MAP masses (Msun) keyed by its mass bin
    per_halo = []  # (bin_index, masses, halo_mass)
    for gid, rec in gm_dict.items():
        b = idcs[gid]
        if 0 < b <= nbin:
            masses = np.array([pid2mult[mp] for mp in rec.map_pids], float) * Mp_val
            per_halo.append((b - 1, masses, halo_masses[gid]))

    # optional abundance cut: a single MAP-mass threshold (cf. make_HOD_plot).
    # When abundance is None the original code's threshold was disabled (==0).
    thresh = 0.0
    if abundance is not None:
        if BoxSize is None:
            raise ValueError("abundance matching requires BoxSize")
        pooled = (np.concatenate([m for _, m, _ in per_halo])
                  if per_halo else np.array([0.]))
        thresh = get_threshold(pooled, BoxSize, guess=0., abundance=abundance)[0]

    centrals = [[] for _ in range(nbin)]
    totals   = [[] for _ in range(nbin)]
    for b, masses, mh in per_halo:
        masses = masses[masses > thresh]
        if len(masses) > 0:
            centrals[b].append(masses.max() / mh)  # central = largest MAP
            totals[b].append(masses.sum() / mh)    # total   = all MAPs

    cen_mean = np.array([np.mean(c) if c else np.nan for c in centrals])
    cen_std  = np.array([np.std(c)  if c else np.nan for c in centrals])
    tot_mean = np.array([np.mean(t) if t else np.nan for t in totals])
    tot_std  = np.array([np.std(t)  if t else np.nan for t in totals])
    return cen_mean, cen_std, tot_mean, tot_std


def plot_map_halo_mass_frac(gm_tuples, Mp, basePath="/flare/Abacus/ajohnson/TNG300/TNG300-2/output", snapno=None,
                            redshift=0.2, labels=None, default=None, default_label='',
                            mass_min=1e11, mass_max=1e15, nbins=30,
                            abundance=None, BoxSize=None,
                            cmaps=('Blues_r', 'Greens_r', 'Purples_r', 'Oranges_r'),
                            clo=0., chi=0.7, panel_size=(7, 14 / 6.),
                            save=False, str_identifier=''):
    """Compare central/total MAP "mass" relative to parent halo mass across runs.

    Output analogue of ``acj.utilities.compare_map_fractions`` and
    ``compare_map_correlation_functions``: a column of subplots (one per tuple)
    sharing both axes, with the curves of each tuple drawn in shades of a
    single-color colormap chosen per tuple from ``cmaps`` (sampled on
    ``[clo, chi]`` to avoid the near-white end), and an optional gray reference
    overplotted in every subplot. Returns ``(fig, axes)``.

    Each curve is a ``(gm_dict, maplog)`` pair -- the new-data-model inputs of
    ``make_HOD_plot``:

    ``gm_dict`` : ``{group_id: MapGalaxyRec}``
        Produced by ``make_map_tng_galaxy_dict_pos`` (keyed by the true TNG group
        id); ``rec.map_pids`` are the pids of the MAPs matched to that halo. This
        is the ``gal_map_dict`` of the original inline code.
    ``maplog`` : dict
        ``read_maplog_dir(..., compute_pids=True)`` output; its ``mult`` column at
        the timeslice nearest ``redshift`` supplies the per-MAP multiplicity.

    For every halo-mass bin two quantities are plotted in the curve's color:
    the **central** fraction ``<M_central_MAP / M_h>`` (largest MAP) as a solid
    line and the **total** fraction ``<sum M_MAP / M_h>`` as a dashed line, each
    with a +/-1 std band.

    Parameters
    ----------
    gm_tuples : sequence of tuple of (gm_dict, maplog)
        One tuple per subplot row; each tuple is a set of ``(gm_dict, maplog)``
        pairs to overplot.
    basePath, snapno :
        TNG group-catalog base path and snapshot; the dm ``GroupMassType`` of the
        full halo catalog is loaded once and indexed by the gm_dict group ids.
        ``snapno=None`` resolves it (and the maplog timeslice) from ``redshift``.
    redshift : float
        Redshift of the MAP timeslice / TNG snapshot (the gm_dicts must have been
        matched at it).
    Mp : float or astropy.units.Quantity
        Particle mass in Msun; MAP "mass" is ``mult * Mp``.
    labels : sequence of tuple of str, optional
        Same shape as ``gm_tuples``; legend label of each pair's (solid) curve.
    default : (gm_dict, maplog), optional
        Reference pair drawn as gray solid (central) / gray dashed (total) in
        every subplot; computed once and reused (the baseline everything is
        compared against, like the old ``idx == 0`` curve).
    default_label : str
        Legend label for the gray reference.
    mass_min, mass_max, nbins : float, float, int
        Log-spaced halo-mass binning (Msun) for the x-axis.
    abundance : float, optional
        If given, impose a shared MAP-mass threshold matching this number density
        (requires ``BoxSize``); if None, no cut (matching the original code).
    BoxSize : astropy.units.Quantity, optional
        Box size, required only when ``abundance`` is set.
    cmaps, clo, chi, panel_size, save, str_identifier :
        Same roles as in ``compare_map_fractions``.

    Returns
    -------
    (fig, axes) : the Figure and the 1D ndarray of Axes.
    """
    if snapno is None:
        tng_snapnos, tng_zs = get_available_tng_snapshots(basePath)
        snapno = int(tng_snapnos[np.argmin(np.abs(tng_zs - redshift))])
    dm_type     = il.snapshot.partTypeNum('dm')
    halo_masses = il.groupcat.loadHalos(basePath, snapno,
                                        fields=['GroupMassType'])[:, dm_type] * 1e10
    Mp_val      = getattr(Mp, 'value', Mp)

    edges = np.logspace(np.log10(mass_min), np.log10(mass_max), nbins + 1)
    bn    = edges[:-1] + np.diff(edges) / 2.0

    nrows = len(gm_tuples)
    if labels is not None:
        assert len(labels) == nrows, "labels must have one tuple per gm tuple (row)"
        for row, (pairs, labs) in enumerate(zip(gm_tuples, labels)):
            assert len(labs) == len(pairs), \
                "labels[{}] must match the number of pairs in row {}".format(row, row)

    fig, axes = plt.subplots(nrows, 1, sharex=True, sharey=True, squeeze=False,
                             figsize=(panel_size[0], panel_size[1] * nrows))
    axes = axes[:, 0]
    fig.subplots_adjust(hspace=0)

    # gray reference, computed once and reused in every subplot
    default_curves = None
    if default is not None:
        gm_d, ml_d = default
        default_curves = _map_halo_mass_frac_curves(
            gm_d, ml_d, halo_masses, edges, Mp_val,
            abundance=abundance, BoxSize=BoxSize, redshift=redshift)

    for row, pairs in enumerate(gm_tuples):
        ax    = axes[row]
        cmap  = colormaps[cmaps[row % len(cmaps)]]
        n     = len(pairs)
        # one shade per pair, held constant; matches compare_map_fractions
        fracs = clo + (chi - clo) * np.linspace(0, 1, max(n, 1))

        if default_curves is not None:
            cen_m, cen_s, tot_m, tot_s = default_curves
            ax.plot(bn, cen_m, '-', lw=2, color='gray', label=default_label)
            ax.fill_between(bn, cen_m - cen_s, cen_m + cen_s, color='gray', alpha=0.15)
            ax.plot(bn, tot_m, '--', lw=2, color='gray')
            ax.fill_between(bn, tot_m - tot_s, tot_m + tot_s, color='gray', alpha=0.15)

        for m, (gm_dict, maplog) in enumerate(pairs):
            cen_m, cen_s, tot_m, tot_s = _map_halo_mass_frac_curves(
                gm_dict, maplog, halo_masses, edges, Mp_val,
                abundance=abundance, BoxSize=BoxSize, redshift=redshift)
            color = cmap(fracs[m])
            label = str(labels[row][m]) if labels is not None else ''
            ax.plot(bn, cen_m, '-', lw=2, color=color, label=label)
            ax.fill_between(bn, cen_m - cen_s, cen_m + cen_s, color=color, alpha=0.12)
            ax.plot(bn, tot_m, '--', lw=2, color=color)
            ax.fill_between(bn, tot_m - tot_s, tot_m + tot_s, color=color, alpha=0.12)

        ax.grid()
        ax.set(xscale='log', yscale='log',
               ylabel=r'$\langle M_{MAP}/M_h \rangle$')
        if labels is not None or default_curves is not None:
            ax.legend(fontsize='small')

    axes[0].set_title(r'MAP "mass" / parent halo mass')
    axes[-1].set_xlabel(r'$M_h\;\;[h^{-1}M_{\odot}]$')

    if save:
        ids = ''
        if abundance:
            ids += f'_n={abundance:.2e}'
        elif len(str_identifier)>0:
            ids += "_"+str_identifier
    
        fig.savefig(f'Images/map_halo_mass_frac' + ids + '.png',
                    bbox_inches='tight')
    plt.show()
    return fig, axes

    
# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    USER = os.getenv("USER")
    p.add_argument('--basePath', required=False,default='/flare/Abacus/'+USER+'/TNG300/TNG300-2/output', help='TNG output basePath')
    p.add_argument('--mapdir', required=True, help='directory of simulation')
    p.add_argument('--redshifts', type=float, nargs='+', required=True,
                   help='one or more target redshifts to match at')
    p.add_argument('--mass_min', type=float, default=1e5,
                   help='only match against halos above this mass [Msun] '
                        '(default 1e5)')
    p.add_argument('--ztol', type=float, default=5e-3,
                   help='max |z_req - z_tng_snapshot| allowed (default 5e-3)')
    p.add_argument('--out_dir', default=None,
                   help='directory to write one gm_dict_z<redshift>.pkl per '
                        'redshift into')
    p.add_argument('--recalc', action='store_true',
                   help='recompute even if a redshift\'s pickle already exists')
    return p.parse_args()


def main():
    args = parse_args()
    match_maps_to_tng(args.mapdir, args.basePath, args.redshifts,
                      mass_min=args.mass_min, out_dir=args.out_dir,
                      recalc=args.recalc, ztol=args.ztol)


if __name__ == '__main__':
    main()
