#!/usr/bin/env python
"""Batch MAP diagnostics for many simulation directories, in parallel via MPI.

For every simulation directory, produces four figures under ``--out_dir/Images/``:

1. ``map_xi_zevolution_<sim>.png`` -- MAP correlation function r^2 xi(r) at
   ``--redshifts`` (default 5, 4, 2, 0.2; each resolved to the nearest
   available MAP step), overlaid in one figure
   (``utilities.plot_map_xi_zevolution``).
2. ``mult_histogram_<sim>.png`` -- log-log histogram of the final-timeslice
   MAP multiplicity (``utilities.plot_mult_histogram``).
3. ``velocities_<sim>.png`` -- mean formation/accretion/merger velocities vs.
   redshift, from that run's ``history.asdf``
   (``utilities.plot_vels``; skipped with a warning if that file is
   missing -- gather it first with ``python -m abacus.post.history <dir>``).
4. ``compare_map_fractions_<sim>.png`` -- N_MAP / N_DM vs. redshift
   (``utilities.compare_map_fractions``).

Simulation directories are split round-robin across MPI ranks
(``sims[rank::size]``), so each rank reads/plots its own disjoint subset with
no cross-rank communication. One rank's failure on one sim (missing files,
corrupt maplog, ...) is logged and skipped rather than aborting that rank's
whole batch.

Run with, e.g. (2 MPI ranks per node, one per socket -- see the accompanying
``run_sim_diagnostics.pbs``)::

    mpiexec -n $NTOTRANKS --ppn 2 --depth=$NTHREADS_PER_RANK --cpu-bind depth \\
            --mem-bind list:0:1 --env OMP_NUM_THREADS=$NTHREADS_PER_RANK \\
            python ~/map_analysis/utilities/run_sim_diagnostics.py \\
                --sims_file sims.txt --out_dir /flare/Abacus/ajohnson/diagnostics \\
                --redshifts 5 4 2 0.2 --nthreads $NTHREADS_PER_RANK
"""

import os
import sys
import glob
import argparse
import traceback
import types
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # headless: every rank just saves figures, never opens a window
import matplotlib.pyplot as plt

# `abacus` is an editable meson-python install whose import *finder* rebuilds
# the C++ extension (build/cp313) on any `import abacus.*` -- even pure-Python
# submodules. This script runs many MPI ranks concurrently (2 per node, one
# per socket, across many nodes -- see module docstring), and those separate
# processes rebuilding build/cp313 at the same time would corrupt it ("ninja:
# error: rebuilding 'build.ninja'"). The only abacus dependency in this code
# path is abacus.param.InputFile, which is pure Python (io + shlex), so load
# it straight from source and register it in sys.modules. Seeding the module
# cache makes Python resolve `from abacus.param import InputFile` from cache
# without ever consulting the meson-python finder -- both here and in
# utilities below, which imports it at module scope. No rebuild, no
# editable build needed at all.
if "abacus.param" not in sys.modules:
    _if_src = Path(os.environ.get("ABACUS_REPO", Path.home() / "abacus")) / "py/abacus/param/input_file.py"
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

from utilities import (read_maplog_dir, plot_map_xi_zevolution,
                       plot_mult_histogram, plot_vels, compare_map_fractions)


def process_sim(simdir, redshifts, abundance, nthreads):
    """Run all four diagnostics for one simulation directory."""
    label = Path(simdir).name or Path(simdir).parent.name  # name of last component (strip trailing '/')

    maplog = read_maplog_dir(simdir)

    fig, _ = plot_map_xi_zevolution(maplog, redshifts=redshifts, abundance=abundance,
                                    nthreads=nthreads, save=True, str_identifier=label)
    plt.close(fig)

    fig, _ = plot_mult_histogram(maplog, save=True, str_identifier=label)
    plt.close(fig)

    del maplog  # done with the (potentially huge) concatenated arrays before the next sim

    try:
        fig, _ = plot_vels(simdir, save=True)
        plt.close(fig)
    except FileNotFoundError as e:
        print(f"  [{label}] skipping plot_vels: {e}", flush=True)

    fig, _ = compare_map_fractions([(simdir,)], labels=[[label]], save=True,
                                   str_identifier=label)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('sims', nargs='*',
                   help='Simulation directories to process (glob patterns are expanded).')
    p.add_argument('--sims_file',
                   help='Text file with one simulation directory per line '
                        '(blank lines and lines starting with # are skipped). '
                        'Combined with any directories given positionally.')
    p.add_argument('--out_dir', required=True,
                   help='Output root; figures are written under <out_dir>/Images/.')
    p.add_argument('--redshifts', type=float, nargs='+', default=[5, 4, 2, 0.2],
                   help='Redshifts for the correlation-function figure (default: 5 4 2 0.2); '
                        'each is resolved to the nearest available MAP step.')
    p.add_argument('--abundance', type=float, default=5.01e-3,
                   help='Target number density (Mpc/h)^-3 for the multiplicity threshold '
                        '(default: 5.01e-3).')
    p.add_argument('--nthreads', type=int, default=8,
                   help='Threads per rank passed to the correlation-function pair counts '
                        '(match --ompthreads/$NTHREADS_PER_RANK in the .pbs; default: 8).')
    return p.parse_args()


def main():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    args = parse_args()

    sims = list(args.sims)
    if args.sims_file:
        with open(args.sims_file) as f:
            sims += [line.strip() for line in f if line.strip() and not line.startswith('#')]
    # expand glob patterns (a plain existing directory is returned unchanged by glob)
    sims = sorted({d for pattern in sims for d in (glob.glob(pattern) or [pattern])})
    if not sims:
        if rank == 0:
            print("No simulation directories given (positional args, --sims_file, "
                  "or glob patterns matched nothing).", file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'Images'), exist_ok=True)
    os.chdir(args.out_dir)  # plot_* functions save to a relative 'Images/...' path

    my_sims = sims[rank::size]
    print(f"[rank {rank}/{size}] {len(my_sims)}/{len(sims)} sim(s) assigned", flush=True)

    n_ok = n_fail = 0
    for simdir in my_sims:
        print(f"[rank {rank}] {simdir}", flush=True)
        try:
            process_sim(simdir, args.redshifts, args.abundance, args.nthreads)
            n_ok += 1
        except Exception:
            n_fail += 1
            print(f"[rank {rank}] FAILED on {simdir}:", file=sys.stderr, flush=True)
            traceback.print_exc()

    print(f"[rank {rank}] done: {n_ok} ok, {n_fail} failed", flush=True)

    total_fail = comm.reduce(n_fail, op=MPI.SUM, root=0)
    if rank == 0:
        status = "OK" if total_fail == 0 else f"{total_fail} sim(s) FAILED"
        print(f"All ranks done ({len(sims)} sim(s) total): {status}")
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main() or 0)
