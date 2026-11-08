#!/usr/bin/env python
"""Correlation functions of the AbacusAurora "emulatorslim0p5" cosmological
parameter variations, compared against the fiducial cosmology.

Standalone script version of the "Emulator Slim" section at the end of
acj/analysis.ipynb. c112 == c113 == c133 == c255 all share the same fiducial
cosmology (H0=67.36, omega_b=0.02237, omega_cdm=0.12, n_s=0.9649); every other
cosmology index perturbs exactly one physical parameter away from that
fiducial (confirmed by grepping abacus.par across every
AbacusAurora_emulatorslim0p5_c*_ph* dir). The Map_* halo-finder parameters are
identical across all of these, so any difference in the correlation functions
traces back to the cosmology alone. w0/wa/Omega_K are fixed (-1, 0, 0)
everywhere.

One row per varied parameter -- every step size (small/mid/large) for a given
parameter is overplotted in that same row, ordered low to high value so the
row's colormap shading runs low -> high too. Omega_M is split into two
separate rows since it's varied two different ways: jointly through
omega_cdm & h (small/mid/large steps), or through h alone at fixed omega_cdm.

Saves two figures (paths relative to the abacus repo root -- see
``utilities.compare_correlation_functions``):
``Images/compare_correlation_functions_n<abundance>_cosmo_variations.png`` (raw
r^2*xi) and
``Images/compare_correlation_functions_n<abundance>_cosmo_variations_ratio.png``
(each curve divided by the fiducial at that redshift).

The actual (slow, memory-heavy) ASDF read and pair-counting happens in
``cache_cosmo_variations_xi.py`` -- run that first (see
``run_cosmo_variations.pbs``); this script only reads each directory's tiny
header/step/redshift arrays (``keys=[]``) and pulls correlation functions out
of the xi cache that script populated. It never touches per-particle data
(``pos``/``mult``/``control``) itself -- a cache miss here means
``cache_cosmo_variations_xi.py`` hasn't (yet) covered that (directory,
redshift), and raises a ``KeyError`` on ``'pos'`` rather than silently paying
for an expensive full read.

Run directly, or via the accompanying ``run_cosmo_variations.pbs``::

    python -m acj.plot_cosmo_variations --nworkers 8 --nthreads 26
"""

import argparse
from pathlib import Path

import numpy as np

from utilities import (read_maplog_dir, get_map_xi_cached,
                       compare_correlation_functions, map_step_for_redshift)
from abacus.param import InputFile

AURORA_BASE = Path("/flare/Abacus/AbacusAurora")
FIDUCIAL_COSMO = "c112"

# Redshift columns compare_correlation_functions is asked to plot. HIGH_Z is
# always available; the low-z column prefers LOW_Z_PREFERRED but falls back
# to LOW_Z_FALLBACK per-directory in cache_cosmo_variations_xi.py (some
# emulatorslim0p5 variations haven't (yet) been run down to z=0.2) -- this
# script's own grid uses LOW_Z_PREFERRED for every row, so a variation stuck
# at LOW_Z_FALLBACK will surface as a clear ztol error from
# compare_correlation_functions rather than being silently mixed into the
# same column as everything else.
HIGH_Z = [4.0, 2.0, 1.0]
LOW_Z_PREFERRED, LOW_Z_FALLBACK = 0.2, 0.5
REDSHIFTS = HIGH_Z + [LOW_Z_PREFERRED]

# Must match compare_correlation_functions' own ztol default -- also used by
# resolve_redshifts below to decide whether LOW_Z_PREFERRED is available.
ZTOL = 5e-2

# Must match compare_correlation_functions' own rmin/rmax/nbins defaults --
# the xi cache is only reused by that final call if the two agree on the
# separation binning.
RMIN, RMAX, NBINS = 2e-2, 40.0, 30

# Where get_map_xi_cached persists its cache across runs -- analysis_data/ is
# a symlink into /flare (see acj/analysis_data), so this survives a
# killed/resubmitted PBS job. No raw maplog data is ever written here, only
# the tiny per-(directory, redshift) xi_*.npz correlation-function results.
CACHE_DIR = Path(__file__).parent / "analysis_data" / "maplog_cache"

# row label -> (abacus.par key used for the % delta, [cosmo indices, low -> high value])
PARAM_VARIATIONS = {
    'omega_b':                   ('omega_b', ['c101', 'c100']),
    'n_s':                       ('n_s',     ['c132', 'c236', 'c105', 'c104', 'c234', 'c135']),
    'Omega_M (via omega_cdm+h)': ('Omega_M', ['c131', 'c232', 'c103', 'c102', 'c230', 'c134']),
    'Omega_M (via H0 only)':     ('Omega_M', ['c238', 'c240']),
}


def cosmo_dir(cosmo, preferred_phase=311):
    """MAP sim directory for a cosmology index, preferring ph<preferred_phase>
    and falling back to whatever phase is actually on disk (only c261 lacks
    ph311, and isn't used above)."""
    preferred = AURORA_BASE / f"AbacusAurora_emulatorslim0p5_{cosmo}_ph{preferred_phase}"
    if preferred.is_dir():
        return preferred
    matches = sorted(AURORA_BASE.glob(f"AbacusAurora_emulatorslim0p5_{cosmo}_ph*"))
    if not matches:
        raise FileNotFoundError(f"no AbacusAurora_emulatorslim0p5_{cosmo}_ph* directory found")
    return matches[0]


def read_par(mdir):
    return InputFile(fn=str(Path(mdir) / "abacus.par"))


def build_dir_and_label_tuples():
    """maplog_tuples/labels shape expected by compare_correlation_functions: one
    tuple per row (per parameter), with every step size for that parameter as a
    curve in the tuple; legend text quotes the value and its % delta from the
    fiducial (map_default), computed directly from abacus.par rather than
    hardcoded so it can't drift out of sync with the sims on disk."""
    fiducial_dir = cosmo_dir(FIDUCIAL_COSMO)
    fiducial_par = read_par(fiducial_dir)

    cosmo_dir_tuples = []
    cosmo_label_tuples = []
    row_names = list(PARAM_VARIATIONS.keys())
    for name, (key, cosmos) in PARAM_VARIATIONS.items():
        fid_val = fiducial_par[key]
        dirs, labels = [], []
        for cosmo in cosmos:
            d = cosmo_dir(cosmo)
            val = read_par(d)[key]
            pct = 100.0 * (val - fid_val) / fid_val
            dirs.append(d)
            labels.append(f'{key}={val:.5g} ({pct:+.1f}%)')
        cosmo_dir_tuples.append(tuple(dirs))
        cosmo_label_tuples.append(tuple(labels))

    for name, dirs, labs in zip(row_names, cosmo_dir_tuples, cosmo_label_tuples):
        print(f'--- {name} ---')
        for d, lab in zip(dirs, labs):
            print(f'  {lab:24s} {d.name}')

    return fiducial_dir, cosmo_dir_tuples, cosmo_label_tuples


def resolve_redshifts(maplog, label=''):
    """HIGH_Z plus whichever of LOW_Z_PREFERRED/LOW_Z_FALLBACK this maplog
    actually reaches (preferring the lower one), within ZTOL. Only used by
    ``cache_cosmo_variations_xi.py`` -- this script's own grid always
    requests plain ``REDSHIFTS`` (see its module docstring)."""
    avail = maplog['redshift']
    for z in (LOW_Z_PREFERRED, LOW_Z_FALLBACK):
        if np.min(np.abs(avail - z)) <= ZTOL:
            return HIGH_Z + [z]
    raise ValueError(
        "{}: no timeslice within {:g} of z={:g} or z={:g}; available "
        "redshifts are {}".format(
            label, ZTOL, LOW_Z_PREFERRED, LOW_Z_FALLBACK,
            ', '.join('{:.3f}'.format(z) for z in sorted(avail))))


def read_and_cache_xi(d, args, r_edges, full_keys=False):
    """Read one directory's header/step/redshift (cheap -- ``keys=[]`` skips
    every per-particle field) and pull its correlation function at each of
    ``REDSHIFTS`` out of the xi cache ``cache_cosmo_variations_xi.py`` is
    responsible for having already populated. Returns the (lightweight)
    maplog dict, which is all ``compare_correlation_functions`` needs as
    long as every entry it asks for is already cached.

    ``full_keys=True`` reads ``read_maplog_dir``'s default per-particle
    fields instead of ``keys=[]``. Needed for the ``map_default`` maplog:
    unlike every other row, ``compare_correlation_functions`` calls
    ``get_map_xi`` on it directly (not the cached wrapper), which needs
    ``maplog['control']``/``['mult']``/``['pos']`` to actually be present.
    """
    print(f'Reading maplog: {d}')
    kwargs = {} if full_keys else {'keys': []}
    maplog = read_maplog_dir(str(d), nworkers=args.nworkers, **kwargs)
    for z in REDSHIFTS:
        ts, _ = map_step_for_redshift(maplog, z)
        # max_p must match compare_correlation_functions' own default (1e7)
        # below, or this cache entry is a miss (and fails on maplog['pos']).
        get_map_xi_cached(maplog, Path(d).name, ts, r_edges, args.cache_dir,
                          abundance=args.abundance, max_p=1e7, nthreads=args.nthreads)
    return maplog


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--nworkers', type=int, default=8,
                         help='parallel workers for read_maplog_dir (ASDF decompression)')
    parser.add_argument('--nthreads', type=int, default=26,
                         help='threads for get_xi pair counting')
    parser.add_argument('--abundance', type=float, default=5.01e-3,
                         help='target number density (Mpc/h)^-3 for the MAP threshold')
    parser.add_argument('--cache-dir', type=Path, default=CACHE_DIR,
                         help='directory the xi cache is read from -- must match '
                              'cache_cosmo_variations_xi.py\'s --cache-dir, or every '
                              'redshift here is a miss (default: %(default)s)')
    args = parser.parse_args()

    r_edges = np.logspace(np.log10(RMIN), np.log10(RMAX), NBINS + 1)

    fiducial_dir, cosmo_dir_tuples, cosmo_label_tuples = build_dir_and_label_tuples()

    # Read the fiducial once -- it's the single reference every row is compared
    # against (map_default), so compare_correlation_functions computes its
    # correlation function once per redshift and reuses it across all rows.
    maplog_fiducial = read_and_cache_xi(fiducial_dir, args, r_edges, full_keys=True)

    # Read every varied-parameter maplog, keeping the same (row, col) shape as
    # cosmo_dir_tuples / cosmo_label_tuples.
    cosmo_maplog_tuples = []
    for dirs in cosmo_dir_tuples:
        row = [read_and_cache_xi(d, args, r_edges) for d in dirs]
        cosmo_maplog_tuples.append(tuple(row))

    cosmo_cache_key_tuples = [tuple(d.name for d in dirs) for dirs in cosmo_dir_tuples]

    row_names = list(PARAM_VARIATIONS.keys())

    compare_correlation_functions(
        cosmo_maplog_tuples,
        redshifts=REDSHIFTS,
        labels=cosmo_label_tuples,
        row_names=row_names,
        map_default=maplog_fiducial,
        default_label='Fiducial',
        cmaps=('Blues', 'Greens', 'Purples', 'Oranges'),
        abundance=args.abundance,
        nthreads=args.nthreads,
        save=True,
        str_identifier='cosmo_variations',
        xi_cache_dir=args.cache_dir,
        xi_cache_keys=cosmo_cache_key_tuples,
        default_cache_key=fiducial_dir.name,
    )
    print('Saved figure -> Images/compare_correlation_functions_'
          f'n{args.abundance:.2e}_cosmo_variations.png')

    # Same grid, but every curve divided by the fiducial at that redshift
    # (map_default doubles as the ratio reference here, same as
    # plot_map_variations.py's ratio-to-fiducial figure).
    compare_correlation_functions(
        cosmo_maplog_tuples,
        redshifts=REDSHIFTS,
        labels=cosmo_label_tuples,
        row_names=row_names,
        map_default=maplog_fiducial,
        default_label='Fiducial',
        cmaps=('Blues', 'Greens', 'Purples', 'Oranges'),
        abundance=args.abundance,
        nthreads=args.nthreads,
        save=True,
        str_identifier='cosmo_variations_ratio',
        plot_ratio=True,
        xi_cache_dir=args.cache_dir,
        xi_cache_keys=cosmo_cache_key_tuples,
        default_cache_key=fiducial_dir.name,
    )
    print('Saved figure -> Images/compare_correlation_functions_'
          f'n{args.abundance:.2e}_cosmo_variations_ratio.png')


if __name__ == '__main__':
    main()
