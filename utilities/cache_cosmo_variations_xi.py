#!/usr/bin/env python
"""Populate the xi cache used by ``plot_cosmo_variations.py``, one directory
at a time.

This is the heavy step: for each cosmology-variation directory, read
``pos``/``mult``/``control`` off the ASDF maplog files, compute the MAP
correlation function at each resolved redshift, and cache it to a few-KB
``xi_*.npz`` file (``utilities.get_map_xi_cached``). The maplog dict itself
goes out of scope as soon as this directory's redshifts are all cached, so
peak memory is one directory's ``pos``/``mult``/``control`` arrays, not
every directory's at once -- holding all ~17 directories' full maplogs
simultaneously (as ``plot_cosmo_variations.py`` used to, before this script
existed) is what caused the ``BrokenProcessPool`` crash this replaces.

The read itself goes through ``read_maplog_dir_cached``, which persists the
trimmed (``pos``/``mult``/``control`` only -- not the ~10-field default,
see ``plot_cosmo_variations.read_and_cache_xi``'s old docstring) maplog to
``args.cache_dir`` the first time a directory is read. ``read_maplog_dir``'s
own docstring notes there's no I/O to save by requesting fewer columns --
blosc2 decompresses each file in full regardless -- so trimming ``keys``
only shrinks the *stored*/in-memory result, not the read time; the actual
walltime savings on a resubmitted (killed) job come entirely from this
on-disk cache letting a directory that's already been read skip ASDF
decompression a second time (a pickle load instead).

Run directly, or via ``run_cosmo_variations.pbs`` (before
``plot_cosmo_variations.py``)::

    python cache_cosmo_variations_xi.py --nworkers 32 --nthreads 32
"""

import argparse
from pathlib import Path

import numpy as np

from utilities import read_maplog_dir_cached, get_map_xi_cached, map_step_for_redshift
from plot_cosmo_variations import (
    build_dir_and_label_tuples, resolve_redshifts,
    RMIN, RMAX, NBINS, CACHE_DIR,
)


def cache_dir_xi(d, args, r_edges):
    """Read one directory's (pos/mult/control) maplog -- from the on-disk
    cache if a previous run already read it, else fresh off ASDF -- and cache
    its correlation function at every resolved redshift. The maplog is local
    to this call and goes out of scope (freed) as soon as it returns, so
    holding one directory's data at a time is all this ever costs in memory.
    """
    label = Path(d).name
    print(f'Reading maplog: {d}')
    maplog = read_maplog_dir_cached(str(d), args.cache_dir, nworkers=args.nworkers,
                                    keys=('pos', 'mult', 'control'),
                                    force=args.force_reread)
    for z in resolve_redshifts(maplog, label=label):
        ts, zm = map_step_for_redshift(maplog, z)
        print(f'  z={z:g} -> step {ts} (z={zm:.4f})')
        # max_p must match compare_correlation_functions' own default (1e7)
        # in plot_cosmo_variations.py, or that script's own reuse of this
        # cache entry is a miss.
        get_map_xi_cached(maplog, label, ts, r_edges, args.cache_dir,
                          abundance=args.abundance, max_p=1e7, nthreads=args.nthreads,
                          force=args.force_reread)


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
                         help='directory for the maplog + xi on-disk caches -- must match '
                              'plot_cosmo_variations.py\'s --cache-dir for that script to see '
                              'the xi entries this one writes (default: %(default)s)')
    parser.add_argument('--force-reread', action='store_true',
                         help='ignore any existing maplog/xi cache entries and recompute '
                              'everything from ASDF')
    args = parser.parse_args()

    r_edges = np.logspace(np.log10(RMIN), np.log10(RMAX), NBINS + 1)

    fiducial_dir, cosmo_dir_tuples, _ = build_dir_and_label_tuples()

    all_dirs = [fiducial_dir] + [d for dirs in cosmo_dir_tuples for d in dirs]
    for d in all_dirs:
        cache_dir_xi(d, args, r_edges)

    print("Done -- all directories' xi caches are up to date.")


if __name__ == '__main__':
    main()
