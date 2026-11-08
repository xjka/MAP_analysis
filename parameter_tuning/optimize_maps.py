#!/usr/bin/env python
"""Optimize MAP merger parameters to match a reference correlation function.

Goal
----
Tune the MAP *merger* parameters so the Abacus (TNG300-3-Dark-Abacus) galaxy
two-point correlation function matches a reference dataset's -- FLAMINGO by
default, TNG optionally (see ``TARGET_DATASET`` below).  Each "function
evaluation" is a *full Abacus run*, so it is extremely expensive (hours,
multiple nodes).  The entire design here is therefore built around three
things:

1. **Resumability / caching.**  Every evaluated parameter point and its loss are
   persisted to a JSON history on disk.  The optimizer state is *reconstructed
   from that history* on every invocation (ask/tell), so the search survives
   PBS-job boundaries: a 10h job that fits 3 runs simply appends 3 points and the
   next job picks up where it left off.  No evaluation is ever repeated.

2. **Sample efficiency.**  Because runs are so costly we minimize the number of
   them.  The default backend is Bayesian optimization (a Gaussian-process
   surrogate, skopt), which is the standard tool for "few, expensive, noisy"
   black boxes and gives an implicit smoothed gradient via the surrogate.  A
   finite-difference / scipy gradient path is also provided for local refinement
   (see ``estimate_gradient`` and the discussion in ``compute_loss``).

3. **A meaningful, low-noise loss.**  The run is deterministic (fixed ZD_Seed),
   and we disable correlation-function subsampling (``max_p=inf``), so the loss
   is reproducible.  Note the caveat in ``compute_loss``: merger criteria are
   *thresholds*, so the loss surface is expected to be locally staircase-like in
   the parameters -- another reason to prefer a surrogate over raw finite
   differences with a tiny step.

This is meant to be driven from ``prod/ParamTuning/run.pbs`` once ready (replace
the ``python -m abacus.run --clean`` line with ``python optimize_maps.py``).

Status: SKETCH.  Search for "TODO" for the spots that need site-specific values
or a decision before this is production-ready.
"""

from __future__ import annotations

import faulthandler
import gc
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# `acj` is a namespace package at the repo root (no __init__.py), so it only
# resolves when the repo root is on sys.path -- true when cwd is $HOME/abacus,
# but not when running from prod/ParamTuning. Put the repo root (two levels up:
# prod/ParamTuning -> prod -> abacus) on the path so the import below works
# regardless of where the script is launched from.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# acj.utilities already centralizes maplog reading + correlation functions
# (read_maplog_dir, get_map_xi, get_threshold, get_xi, load_tng_subhalos,
# get_available_tng_snapshots, map_step_for_redshift, load_flamingo_halos).
# Reuse them so the loss is computed exactly the way the analysis notebook
# visualizes these curves.
from acj.utilities import (
    read_maplog_dir,
    get_map_xi,
    get_threshold,
    get_xi,
    load_tng_subhalos,
    get_available_tng_snapshots,
    map_step_for_redshift,
    load_flamingo_halos,
    FLAMINGO_CACHE_DIR,
)
import illustris_python as il
from astropy import units as u
from skopt import Optimizer
from skopt.space import Real


HERE = Path(__file__).resolve().parent

# Where the optimizer's own artifacts (evaluation history + cached TNG target)
# are written. Kept off the repo tree and on the flare filesystem alongside the
# per-run outputs. Override with the OPTIMIZE_MAPS_DIR env var if desired.
OUTPUT_DIR = Path(os.environ.get("OPTIMIZE_MAPS_DIR",
                                 f"/flare/Abacus/{os.environ.get('USER', '')}/Tuning"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# 1. What we optimize: the parameter space
# --------------------------------------------------------------------------- #
@dataclass
class ParamSpec:
    """One tunable Abacus parameter.

    ``log=True`` means the optimizer searches in log10 space, which is the right
    choice for positive scale-like quantities (so a factor-of-2 step is the same
    "distance" everywhere).  ``low``/``high`` are hard bounds in *native* units.
    """
    name: str
    low: float
    high: float
    log: bool = True

    def to_search(self, value: float) -> float:
        return np.log10(value) if self.log else value

    def from_search(self, s: float) -> float:
        return float(10.0 ** s) if self.log else float(s)

    @property
    def search_bounds(self) -> tuple[float, float]:
        return (self.to_search(self.low), self.to_search(self.high))


# The two merger knobs.  See proepi.cpp:1018-1028 & tracers.cpp:72 for meaning:
#   Map_bmerge_proper -> merge *distance* threshold, in proper Mpc/h. It sets
#       b_merge = min(Map_bmerge_proper/a/BoxSize, Map_bmerge_max_ips/PPD), so
#       the proper term dominates at late times -- exactly where the xi fit
#       matters. Current par2 value: 0.02.
#   Map_Amerge        -> merge *velocity* threshold (dv^2 < Map_Amerge*m).
#       Current par2 value: 57.7
PARAM_SPECS: list[ParamSpec] = [
    #ParamSpec("Map_bmerge_proper", low=0.01, high=0.03, log=False),
    #ParamSpec("Map_Amerge", low=5, high=65, log=False),
    #ParamSpec("Map_bform_ips", low=0.04, high=0.2, log=False),
    #ParamSpec("Map_baccrete_ips", low=0.04, high=0.2, log=False),
    ParamSpec("Map_Aaccrete", low=0.6, high=2, log=False),
    ParamSpec("Map_bmerge_proper", low=0.02, high=0.05, log=False),
    ParamSpec("Map_Amerge", low=50, high=200, log=False)
]


def vector_to_params(x) -> dict[str, float]:
    """Optimizer search-space vector -> {param_name: native_value}."""
    return {spec.name: spec.from_search(xi) for spec, xi in zip(PARAM_SPECS, x)}


def params_to_vector(params: dict[str, float]):
    """{param_name: native_value} -> optimizer search-space vector."""
    return [spec.to_search(params[spec.name]) for spec in PARAM_SPECS]


# --------------------------------------------------------------------------- #
# 2. The expensive part: run Abacus with overridden parameters
# --------------------------------------------------------------------------- #
def run_abacus(params: dict[str, float], par2file: str = "abacus_png.par2",
               sim_name: str | None = None, dry_run: bool = False) -> Path:
    """Run one Abacus simulation with ``params`` overridden; return its run dir.

    Each evaluation gets a *distinct* SimName so its outputs (maplogs, state,
    logs) never clobber another point's -- essential for caching and for keeping
    the history auditable.  Returns the directory to hand to ``compute_loss``
    (the run's WorkingDirectory, which holds abacus.par, write/state, log/, and
    maplogs/ as consumed by acj.utilities.read_maplog_dir).
    """
    if sim_name is None:
        sim_name = _sim_name_for(params)

    # The par2 has `SimName = "$SIMNAME$"`, and `$VAR$` tokens are substituted
    # from os.environ at par2 *parse* time (gen_param.shellreplace), which runs
    # before any `params=` override is applied -- so a SimName override can't
    # prevent the "Variable 'SIMNAME' is not defined" crash. Set the env var to
    # our per-point name; this is the mechanism the par2 is built around and it
    # drives the whole OutputDirectory/WorkingDirectory tree via @SimName@.
    os.environ["SIMNAME"] = sim_name

    # Import here so --help / dry-run don't require the build to be importable.
    from abacus.sim import abacus

    # LogVerbosity=4 overrides the par2's production default (-1, no per-rank
    # logs) so per-rank STDLOG output is written -- needed to diagnose the
    # crashes these runs keep hitting at roughly the same point every time.
    run_params = dict(params)
    run_params["LogVerbosity"] = 4
    # To catch the recurring HeartbeatTimeoutError hang with a live backtrace
    # instead of losing it, wrap every rank in gdbserver via gdb_wrap.sh
    # (uncomment). This stalls ALL ranks at MPI_Init until every one of them
    # is attached and `continue`d -- use attach_all.sh/interrupt_all.sh in
    # this directory to do that. If it's not attached within StepTimeout
    # (900s), the heartbeat watchdog kills the run and raises
    # HeartbeatTimeoutError -- so don't enable this unless you're actively
    # watching the job and ready to run attach_all.sh right after qsub.
    # run_params["exec_prefix"] = str(HERE / "gdb_wrap.sh")
    sim = abacus.Simulation(par2file=par2file, params=run_params)
    workdir = Path(sim.param["WorkingDirectory"])

    if dry_run:
        print(f"[dry-run] would run {sim_name} with {params}")
        return workdir

    # clean=True: fresh run (IC is preserved by _clean unless erase_ic).  The
    # JSON cache already prevents re-running an evaluated point, so we always
    # start clean here rather than trying to resume a half-finished run.
    try:
        ret = sim.run(clean=True)
        print(f"[run_abacus] {sim_name}: sim.run returned {ret} at {time.strftime('%H:%M:%S')}")
        if ret != 0:
            raise RuntimeError(f"Abacus run {sim_name} failed with code {ret}")
    except Exception as e:
        # Let evaluate() find the run dir to preserve before its next retry's
        # clean=True would otherwise wipe it out.
        e.workdir = workdir
        raise
    return workdir


def _output_root() -> Path:
    """$ABACUS_OUTPUT_ROOT (the par2's OutputDirectory base), which every
    history sim_name is relative to."""
    return Path(os.environ.get("ABACUS_OUTPUT_ROOT",
                               f"/flare/Abacus/{os.environ.get('USER', '')}"))


def _sim_name_for(params: dict[str, float], base: str = "MapTune") -> str:
    """Deterministic, filesystem-safe SimName from the parameter values.

    The prefix is OUTPUT_DIR (OPTIMIZE_MAPS_DIR) relative to
    $ABACUS_OUTPUT_ROOT (the par2's OutputDirectory is
    "$ABACUS_OUTPUT_ROOT/" + @SimName@), so the run outputs sit alongside the
    optimizer's history/target in OUTPUT_DIR and moving OPTIMIZE_MAPS_DIR moves
    both. The decimal-point/minus-sign substitution keeps the leaf name
    filesystem-safe while leaving the "/" separators intact.
    """
    out_root = _output_root()
    try:
        prefix = OUTPUT_DIR.resolve().relative_to(out_root.resolve())
    except ValueError:
        raise RuntimeError(
            f"OPTIMIZE_MAPS_DIR ({OUTPUT_DIR}) must be under ABACUS_OUTPUT_ROOT "
            f"({out_root}) so run directories can be nested beside the history")
    parts = [base]
    for spec in PARAM_SPECS:
        parts.append(f"{spec.name.replace('Map_', '')}_{params[spec.name]:.5g}")
    leaf = "_".join(parts).replace(".", "p").replace("-", "m")
    return f"{prefix}/{leaf}"


# --------------------------------------------------------------------------- #
# 3. The loss: discrepancy between Abacus and the reference correlation function
# --------------------------------------------------------------------------- #
# Redshifts at which to compare.  Must be present both in MapLogRedshifts (par2:
# [5.0, 4.0, 3.0, 2.0, 1.0, 0.3]) and as an available snapshot of whichever
# dataset TARGET_DATASET selects.  Fewer redshifts == cheaper loss; the
# late-time ones constrain mergers most.
# TODO: confirm which redshifts you actually want to fit (and the abundance).
COMPARE_REDSHIFTS = (4,1,0.2)
ABUNDANCE = 5.01e-3                     # target number density [h/Mpc]^3

# How the separation bins are spaced between R_MIN and R_MAX.
#   "log"    -- np.logspace, even spacing in log(r). Concentrates bins at
#       small scales, so the loss is dominated by the 1-halo term.
#   "linear" -- np.linspace, even spacing in r (default). Weights bins by
#       physical separation instead, so large scales aren't drowned out by
#       many small-scale bins -- avoids fitting to just the tiny scales.
# Switch via R_SPACING env var, or just edit the default below.
# NOTE: changing this (like changing N_R_BINS/R_MIN/R_MAX) changes what
# compute_loss() means. That's handled automatically: every history record is
# tagged with the loss_config() it was computed under, and optimize() recomputes
# any stale record from its on-disk maplog before the GP sees it.
R_SPACING = os.environ.get("R_SPACING", "linear")
R_MIN = float(os.environ.get("R_MIN", 2e-2))    # [Mpc/h] set via env var, e.g. in run.pbs
R_MAX = float(os.environ.get("R_MAX", 20.0))    # [Mpc/h]
N_R_BINS = int(os.environ.get("N_R_BINS", 20))  # number of R_EDGES (so N_R_BINS-1 bins)
if R_SPACING == "log":
    R_EDGES = np.logspace(np.log10(R_MIN), np.log10(R_MAX), N_R_BINS)
elif R_SPACING == "linear":
    R_EDGES = np.linspace(R_MIN, R_MAX, N_R_BINS)
else:
    raise ValueError(f"Unknown R_SPACING {R_SPACING!r}; expected 'log' or 'linear'")
R_CENT = R_EDGES[:-1] + np.diff(R_EDGES) / 2.0

# Which reference dataset the MAP correlation function is fit against.
#   "flamingo" (default) -- FLAMINGO L1_m8 SOAP halo catalog, thresholded on
#       stellar mass to ABUNDANCE. Bigger box / better statistics than TNG300,
#       and the snapshots at all of COMPARE_REDSHIFTS are already cached
#       locally (see FLAMINGO_SNAPSHOTS / load_flamingo_halos), so no network
#       access is needed at runtime.
#   "tng" -- the original TNG300-2 galaxy target. Switch back to it by setting
#       this to "tng" (or exporting TARGET_DATASET=tng); no other code change
#       is needed.
#   "abacus_fiducial" -- an actual Abacus MAP catalog (not a different
#       simulation/code) read the same way compute_loss() reads the tuning
#       run itself: read_maplog_dir + get_map_xi at each COMPARE_REDSHIFTS
#       epoch, abundance-matched to the same ABUNDANCE. Defaults to
#       AbacusAurora_emulatormax_c000_ph310 -- the c000 comparison sim this
#       whole small-box/reduced-particle-mass tuning setup is meant to track
#       (same cosmology/ZD_Seed as abacus_png.par2, see ABACUS_FIDUCIAL_DIR) -- so
#       the loss isolates the effect of particle mass on the tuned MAP params
#       rather than also picking up TNG/FLAMINGO's own differences from
#       Abacus. Select it with TARGET_DATASET=abacus_fiducial.
#       NOTE: the c000 comparison run has MAP outputs at z = 4.0, 1.0039 and
#       0.3 (its MapLogRedshifts), so COMPARE_REDSHIFTS = (4, 1, 0.3) resolves
#       within ztol (z=1 lands on 1.0039). This par2's FinalRedshift is 0.3 and
#       its MapLogRedshifts/TimeSliceRedshifts_Subsample include 0.3. If you
#       add a redshift the baseline lacks, the ztol check in
#       _abacus_fiducial_xi_at raises ValueError rather than silently
#       comparing against the nearest available epoch.
# NOTE: switching datasets changes what compute_loss() means (same as an
# R_EDGES change). Each dataset caches its target under its own file
# (<dataset>_target.npz), and history records computed under another dataset
# are recomputed automatically at startup (see loss_config / History).
TARGET_DATASET = os.environ.get("TARGET_DATASET", "abacus_fiducial")

# TODO: point this at the TNG300-2 (or -3) reference output used as the target.
TNG_BASEPATH = os.environ.get(
    "TNG_BASEPATH",
    f"/flare/Abacus/{os.environ.get('USER', '')}/TNG300/TNG300-2/output",
)

# z -> FLAMINGO SOAP snapshot step, covering COMPARE_REDSHIFTS. Extend this if
# COMPARE_REDSHIFTS changes; a step not yet cached under FLAMINGO_CACHE_DIR
# requires load_flamingo_halos(..., root_dir=hdfstream.open('cosma', '/')) once
# to fetch and cache it (see acj.utilities.load_flamingo_halos).
FLAMINGO_SNAPSHOTS = {5: 10, 2: 38, 1: 58, 0.2: 74}

# The c000 comparison run backing TARGET_DATASET="abacus_fiducial" -- same
# cosmology (abacus_cosm000) and ZD_Seed (253) abacus_png.par2 is set up to match.
ABACUS_FIDUCIAL_DIR = os.environ.get(
    "ABACUS_FIDUCIAL_DIR",
    "/flare/Abacus/AbacusAurora/AbacusAurora_emulatormax_c000_ph310",
)


# Bump whenever compute_loss()'s *formula* changes (weighting, residual form,
# error model, ...) -- anything loss_config() can't see from the settings alone.
# Every history record computed under an older version is then recomputed.
LOSS_VERSION = 1


def _target_source() -> str:
    """The on-disk source of the active TARGET_DATASET's reference curves."""
    return {"tng": TNG_BASEPATH,
            "flamingo": str(FLAMINGO_CACHE_DIR),
            "abacus_fiducial": ABACUS_FIDUCIAL_DIR}.get(TARGET_DATASET, "")


def loss_config() -> dict:
    """Every hyper-parameter that defines what compute_loss() returns.

    Stored on each history record and in the target cache; a mismatch with the
    current value means the stored loss/target is stale. R_EDGES itself is
    stored (not R_SPACING/R_MIN/R_MAX/N_R_BINS), so a code edit to the binning
    is caught as well as an env-var change.
    """
    return {
        "loss_version": LOSS_VERSION,
        "target_dataset": TARGET_DATASET,
        "target_source": _target_source(),
        "compare_redshifts": [float(z) for z in COMPARE_REDSHIFTS],
        "abundance": float(ABUNDANCE),
        "r_edges": [float(f"{r:.10g}") for r in R_EDGES],
    }


def _tng_xi_at(z: float, ztol: float = 5e-2):
    """TNG galaxy correlation function (abundance-matched) at redshift ``z``.

    Mirrors the TNG branch of utilities.compare_correlation_functions: subhalos
    thresholded on stellar mass to the target abundance, correlated with get_xi.
    Returns (xi, poisson) on R_CENT.  Cached on disk by the caller.
    """
    star_type = il.snapshot.partTypeNum("star")
    snapnos, zs = get_available_tng_snapshots(TNG_BASEPATH)
    k = int(np.argmin(np.abs(zs - z)))
    if abs(zs[k] - z) > ztol:
        raise ValueError(
            f"no TNG snapshot within {ztol} of z={z}; available: {zs}")
    sn = int(snapnos[k])

    boxsize = il.groupcat.loadHeader(TNG_BASEPATH, sn)["BoxSize"] / 1e3 * u.Mpc
    subs = load_tng_subhalos(TNG_BASEPATH, sn)
    m = subs["SubhaloMassType"][:, star_type]
    thresh = get_threshold(m, boxsize, guess=1e-3, abundance=ABUNDANCE)
    msk = m >= thresh
    twopt = get_xi(subs["SubhaloPos"][msk], R_EDGES, boxsize,
                   max_p=np.inf, nthreads=int(os.environ.get("OMP_NUM_THREADS", 26)))
    xi = twopt.get_corr(return_sep=False)
    poisson = np.sqrt(twopt.D1D2.wcounts) / twopt.R1R2.wcounts
    return np.asarray(xi), np.asarray(poisson)


def _flamingo_xi_at(z: float, ztol: float = 5e-2):
    """FLAMINGO halo correlation function (abundance-matched) at redshift ``z``.

    Mirrors the FLAMINGO branch of utilities.compare_correlation_functions: SOAP
    halos thresholded on stellar mass to the target abundance, correlated with
    get_xi.  Returns (xi, poisson) on R_CENT.  Cached on disk by the caller
    (load_flamingo_halos itself also caches the raw halo catalog).
    """
    zs = np.array(sorted(FLAMINGO_SNAPSHOTS.keys()))
    k = int(np.argmin(np.abs(zs - z)))
    if abs(zs[k] - z) > ztol:
        raise ValueError(
            f"no FLAMINGO snapshot within {ztol} of z={z}; available: {zs} "
            f"(add it to FLAMINGO_SNAPSHOTS)")
    step = FLAMINGO_SNAPSHOTS[float(zs[k])]

    flam = load_flamingo_halos(step, cache_dir=FLAMINGO_CACHE_DIR)
    boxsize = flam["boxsize"] * u.Mpc
    thresh = get_threshold(flam["mass"], boxsize, guess=1e-3, abundance=ABUNDANCE)
    msk = flam["mass"] >= thresh
    twopt = get_xi(flam["pos"][msk], R_EDGES, boxsize,
                   max_p=np.inf, nthreads=int(os.environ.get("OMP_NUM_THREADS", 26)))
    xi = twopt.get_corr(return_sep=False)
    poisson = np.sqrt(twopt.D1D2.wcounts) / twopt.R1R2.wcounts
    return np.asarray(xi), np.asarray(poisson)


_fiducial_maplog_cache: dict | None = None

# get_map_xi only touches pos/mult/control (+ header); timestep is added
# internally by read_maplog_dir.  Read nothing else, and only the
# COMPARE_REDSHIFTS epochs.
_XI_KEYS = ["pos", "mult", "control"]


def _read_xi_maplog(path, nworkers=None):
    kw = {} if nworkers is None else {"nworkers": nworkers}
    return read_maplog_dir(str(path), keys=_XI_KEYS,
                           redshifts=COMPARE_REDSHIFTS, **kw)


def _fiducial_maplog() -> dict:
    """Lazily read + cache ABACUS_FIDUCIAL_DIR's maplog (read once, reused for
    every COMPARE_REDSHIFTS entry -- it's ~175GB on disk, so re-reading it per
    redshift the way _tng_xi_at/_flamingo_xi_at re-open their (much smaller)
    catalogs would be wasteful)."""
    global _fiducial_maplog_cache
    if _fiducial_maplog_cache is None:
        _fiducial_maplog_cache = _read_xi_maplog(
            ABACUS_FIDUCIAL_DIR,
            nworkers=int(os.environ.get("OMP_NUM_THREADS", 26)))
    return _fiducial_maplog_cache


def _abacus_fiducial_xi_at(z: float, ztol: float = 5e-2):
    """Flagship Abacus MAP correlation function (ABACUS_FIDUCIAL_DIR) at
    redshift ``z``.

    Unlike the TNG/FLAMINGO branches, this compares MAP to MAP (same object
    definition, same code) rather than Abacus MAP to a different
    simulation's galaxy/halo catalog -- the point of this dataset is to
    isolate the effect of resolution (particle mass) on the tuned MAP
    parameters, not to also pick up a cross-code/cross-catalog difference.
    Reuses get_map_xi exactly as compute_loss() does for the tuning run
    itself.  Returns (xi, poisson) on R_CENT.
    """
    maplog = _fiducial_maplog()
    ts, zm = map_step_for_redshift(maplog, z)
    if abs(zm - z) > ztol:
        raise ValueError(
            f"no {ABACUS_FIDUCIAL_DIR} MAP timeslice within {ztol} of z={z} "
            f"(nearest available: z={zm}); this flagship run's FinalRedshift "
            f"may not reach that far -- check its abacus.par and adjust "
            f"COMPARE_REDSHIFTS")
    xi, poisson = get_map_xi(maplog, ts, R_EDGES, abundance=ABUNDANCE,
                             max_p=np.inf,
                             nthreads=int(os.environ.get("OMP_NUM_THREADS", 26)))
    return np.asarray(xi), np.asarray(poisson)


def _reference_xi_at(z: float, ztol: float = 5e-2):
    """Dispatch to the TARGET_DATASET-selected reference correlation function."""
    if TARGET_DATASET == "flamingo":
        return _flamingo_xi_at(z, ztol=ztol)
    elif TARGET_DATASET == "tng":
        return _tng_xi_at(z, ztol=ztol)
    elif TARGET_DATASET == "abacus_fiducial":
        return _abacus_fiducial_xi_at(z, ztol=ztol)
    raise ValueError(
        f"unknown TARGET_DATASET {TARGET_DATASET!r}; expected 'flamingo', "
        f"'tng', or 'abacus_fiducial'")


def reference_target(cache_path: Path | None = None) -> dict:
    """Compute (once) and cache the reference target curves at COMPARE_REDSHIFTS.

    The target never changes across the whole optimization, so it is computed a
    single time and reused for every loss evaluation.  Cached under a filename
    keyed by TARGET_DATASET, and reused only if the loss_config() it was built
    under matches the current one (binning, abundance, redshifts, source).
    """
    if cache_path is None:
        cache_path = OUTPUT_DIR / f"{TARGET_DATASET}_target.npz"
    cfg = json.dumps(loss_config(), sort_keys=True)
    if cache_path.exists():
        d = np.load(cache_path)
        if "loss_config" in d.files and str(d["loss_config"]) == cfg:
            return {float(z): (d[f"xi_{i}"], d[f"err_{i}"])
                    for i, z in enumerate(COMPARE_REDSHIFTS)}
        print(f"[reference_target] {cache_path} was built under a different "
              f"loss_config; rebuilding")
    target = {}
    save = {"redshifts": np.asarray(COMPARE_REDSHIFTS),
            "r_edges": np.asarray(R_EDGES),   # bins the xi_/err_ arrays live on
            "loss_config": np.asarray(cfg)}
    for i, z in enumerate(COMPARE_REDSHIFTS):
        xi, err = _reference_xi_at(z)
        target[float(z)] = (xi, err)
        save[f"xi_{i}"], save[f"err_{i}"] = xi, err
    np.savez(cache_path, **save)
    # The flagship maplog is huge and only needed to build the target: release
    # it so it doesn't sit in this process while the tuning sims run alongside.
    global _fiducial_maplog_cache
    _fiducial_maplog_cache = None
    gc.collect()
    return target


def compute_loss(rundir: Path | None = None, maplog: dict | None = None,
                 target: dict | None = None) -> float:
    """Loss = chi-square-like distance between Abacus and reference xi over (z, r).

    Pass either a ``rundir`` (the maplog is read from it) or an already-loaded
    ``maplog`` dict (priming reads it once for both params and loss).

    For each comparison redshift, select the maplog timeslice nearest that z,
    abundance-match on MAP multiplicity (get_map_xi), and compare to the cached
    reference target (FLAMINGO or TNG, see TARGET_DATASET).  The residual is on
    ``r^2 * xi`` (how these curves are read by eye), normalized by the combined
    Poisson error so noisy/empty bins don't dominate:

        L = sum_{z,r} ( r^2 (xi_map - xi_ref) )^2 / ( (r^2 sigma)^2_map + ..._ref )

    CAVEAT: merger criteria are hard thresholds, so as a parameter is nudged a
    MAP either does or does not merge -- the loss is expected to be locally
    piecewise-flat with steps.  That is *why* the optimizer is a GP surrogate
    (which tolerates a rough, noisy surface) rather than a gradient method.

    TODO: decide whether to weight redshifts, fit log(xi) instead of r^2 xi, or
    add an abundance/HOD penalty term.  This is the single most important design
    choice for whether the fit is meaningful -- left deliberately simple here.
    """
    if target is None:
        target = reference_target()
    if maplog is None:
        t0 = time.perf_counter()
        print(f"[compute_loss] reading maplog {rundir} ...")
        maplog = _read_xi_maplog(rundir)  # only pos/mult/control at COMPARE_REDSHIFTS
        print(f"[compute_loss] maplog read in {time.perf_counter() - t0:.1f} s")

    loss = 0.0
    nthreads = int(os.environ.get("OMP_NUM_THREADS", 26))
    for z, (xi_ref, err_ref) in target.items():
        ts, _zm = map_step_for_redshift(maplog, z)
        t0 = time.perf_counter()
        xi_map, err_map = get_map_xi(maplog, ts, R_EDGES, abundance=ABUNDANCE,
                                     max_p=np.inf, nthreads=nthreads)
        print(f"[compute_loss] z={z}: get_map_xi in {time.perf_counter() - t0:.1f} s")
        w = R_CENT ** 2
        num = (w * (xi_map - xi_ref)) ** 2
        den = (w * err_map) ** 2 + (w * err_ref) ** 2
        good = np.isfinite(num) & np.isfinite(den) & (den > 0)
        loss += float(np.sum(num[good] / den[good]))
    return loss


# --------------------------------------------------------------------------- #
# 3b. Priming: seed the search from existing runs
# --------------------------------------------------------------------------- #
# Pre-existing run directories to fold into the search before it starts. These
# are runs you already have -- not points the optimizer would have chosen, but
# every (params -> loss) pair is information the GP surrogate can use. Each entry
# is a run root readable by read_maplog_dir (contains abacus.par, write/state,
# log/, maplogs/). List them here, or pass a file of one path per line via
# --prime-file (blank lines and #-comments ignored).
# TODO: fill in (or point --prime-file at) your existing TNG300-3-Dark runs.
PRIME_DIRS: list[str] = [
    #"/flare/Abacus/ajohnson/TNG300-3-Dark-Abacus_e2/",
]


def params_from_maplog(maplog: dict) -> dict[str, float]:
    """Extract the PARAM_SPECS parameter values from a loaded maplog.

    read_maplog_dir normalizes both the new (Map_*) and old (a_merge/b_merge/...)
    parameter names into Map_* keys, so primed runs with legacy parameter names
    still resolve here.
    """
    out = {}
    for spec in PARAM_SPECS:
        if spec.name not in maplog:
            raise KeyError(
                f"{spec.name} not found in maplog params; this primed run does "
                f"not record it (read_maplog_dir keys: see acj.utilities). "
                f"Can't fold it into the search.")
        out[spec.name] = float(getattr(maplog[spec.name], "value", maplog[spec.name]))
    return out


def prime_from_dirs(dirs, history: "History", target: dict) -> None:
    """Fold each existing run directory into the history as a (params, loss).

    Skips runs already in the history (by parameter key) and runs that error out
    (e.g. missing maplogs) with a warning, so one bad directory can't abort the
    whole priming pass.
    """
    for d in dirs:
        d = str(d).strip()
        if not d or d.startswith("#"):
            continue
        try:
            maplog = _read_xi_maplog(d)
            params = params_from_maplog(maplog)
        except Exception as e:                       # noqa: BLE001 - report & skip
            print(f"[prime] SKIP {d}: {e}")
            continue
        if history.lookup(params) is not None:
            print(f"[prime] already have {params}; skipping {d}")
            continue
        loss = compute_loss(maplog=maplog, target=target)
        # Same convention as _sim_name_for: the run dir relative to
        # ABACUS_OUTPUT_ROOT (e.g. "Tuning_png/MapTune_..."), so
        # History.refresh_stale can find it. A dir outside that root keeps its
        # absolute path, which refresh_stale's `base_dir / sim_name` also
        # resolves (pathlib drops base_dir for an absolute right operand).
        rundir = Path(d).resolve()
        try:
            sim_name = str(rundir.relative_to(_output_root().resolve()))
        except ValueError:
            sim_name = str(rundir)
        history.add(params, loss, sim_name=sim_name)
        print(f"[prime] {d} -> {params} loss={loss:.6g}")


# --------------------------------------------------------------------------- #
# 4. Persistent evaluation history (cache + resumability)
# --------------------------------------------------------------------------- #
class History:
    """JSON-backed list of {params, loss, sim_name} evaluations.

    Doubles as the cache (never re-run an evaluated point) and as the optimizer
    state (the optimizer is rebuilt by telling it the whole history each run).

    Each record also carries the ``loss_config`` it was computed under. Records
    whose config differs from the current loss_config() are *stale*: they're kept
    on disk but ignored by lookup/as_XY/best until refresh_stale() recomputes
    them, so two loss definitions can never mix in the GP's training data.
    """

    def __init__(self, path: Path = OUTPUT_DIR / "optimize_maps_history.json"):
        self.path = Path(path)
        self.records: list[dict] = []
        if self.path.exists():
            self.records = json.loads(self.path.read_text())

    def _key(self, params: dict[str, float]) -> tuple:
        # round in search space so tiny numeric jitter maps to the same key
        return tuple(round(v, 8) for v in params_to_vector(params))

    @staticmethod
    def is_current(r: dict) -> bool:
        return r.get("loss_config") == loss_config()

    def current(self) -> list[dict]:
        return [r for r in self.records if self.is_current(r)]

    def lookup(self, params: dict[str, float]):
        key = self._key(params)
        for r in self.current():
            if self._key(r["params"]) == key:
                return r["loss"]
        return None

    def save(self, path: Path | None = None):
        # Atomic write: a walltime kill mid-write can't corrupt the history --
        # os.replace is atomic, so the file is always either the old or new copy.
        dest = Path(path) if path is not None else self.path
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_text(json.dumps(self.records, indent=2))
        os.replace(tmp, dest)

    def add(self, params: dict[str, float], loss: float, sim_name: str):
        # A stale record for the same point (its run dir was missing, so it
        # couldn't be refreshed) is superseded by this fresh evaluation.
        key = self._key(params)
        self.records = [r for r in self.records
                        if self.is_current(r) or self._key(r["params"]) != key]
        self.records.append({"params": params, "loss": loss, "sim_name": sim_name,
                             "loss_config": loss_config()})
        self.save()

    def refresh_stale(self, target: dict, force: bool = False,
                      out_path: Path | None = None) -> None:
        """Recompute the loss of every stale record (all records if ``force``)
        from its on-disk maplog under the current loss_config(), then save.

        Run dirs are ``self.path.parent.parent / sim_name`` (sim_name is
        relative to ABACUS_OUTPUT_ROOT; an absolute sim_name is used as-is).
        A record whose run dir is missing or unreadable stays stale -- excluded
        from the optimizer -- rather than keeping a loss under the old definition.
        """
        todo = self.records if force else [r for r in self.records if not self.is_current(r)]
        if not todo:
            return
        cfg = loss_config()
        print(f"[history] recomputing {len(todo)}/{len(self.records)} losses under "
              f"loss_config={cfg}")
        base_dir = self.path.parent.parent
        n_ok = 0
        for r in todo:
            rundir = base_dir / r["sim_name"]
            old = r["loss"]
            try:
                new = compute_loss(maplog=_read_xi_maplog(str(rundir)), target=target)
            except Exception as e:                   # noqa: BLE001 - report & skip
                print(f"[history] STALE {rundir}: {type(e).__name__}: {e}; "
                      f"excluded from the optimizer until it can be recomputed")
                continue
            r["loss"], r["loss_config"] = new, cfg
            n_ok += 1
            print(f"[history] {r['sim_name']}: {old:.6g} -> {new:.6g}")
        self.save(out_path)
        print(f"[history] recomputed {n_ok}/{len(todo)}; "
              f"{len(self.records) - len(self.current())} record(s) remain stale")

    def as_XY(self):
        """Return (X_search_vectors, Y_losses) for telling the optimizer."""
        cur = self.current()
        X = [params_to_vector(r["params"]) for r in cur]
        Y = [r["loss"] for r in cur]
        return X, Y

    def best(self):
        cur = self.current()
        if not cur:
            return None
        return min(cur, key=lambda r: r["loss"])


def _preserve_failed_rundir(workdir: Path, sim_name: str, attempt: int) -> Path | None:
    """Move a failed attempt's run dir aside so the next retry's ``clean=True``
    doesn't wipe it -- keeps every attempt's logs/core dump/status.log/maplogs
    for post-mortem instead of only whichever attempt runs last. Returns the
    preserved path (or None if there was nothing to move), so a later success
    can clean these up again -- see the ``preserved_dirs`` cleanup in
    ``evaluate``.

    ``GroupDirectory``/``LogDirectory``/``MultipoleDirectory``/``OutputDirectory``/
    ``TaylorDirectory``/``TracersDirectory`` (the other dirs ``Simulation._clean``
    wipes) are all subdirectories of (or equal to) ``WorkingDirectory`` in this
    par2, so moving just this one directory captures everything.
    """
    if not workdir.exists():
        return None
    dest = workdir.parent / f"{workdir.name}_failed_attempt{attempt}"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(workdir), str(dest))
    print(f"[evaluate] {sim_name}: preserved failed attempt {attempt} at {dest}")
    return dest


# Seconds compute_loss may run before faulthandler dumps all thread stacks.
LOSS_HANG_DUMP_S = float(os.environ.get("LOSS_HANG_DUMP_S", 600))


def evaluate(params: dict[str, float], history: History,
             dry_run: bool = False, max_attempts: int = 3) -> float:
    """Cached objective: return loss for ``params``, running Abacus if needed.

    A run (or the maplog read afterward) is retried up to ``max_attempts``
    times on failure before giving up. The failures seen so far -- a SIGSEGV
    inside the GPU driver, a HeartbeatTimeout hang, a corrupted maplog
    multiplicity -- all look like transient GPU/driver flakiness (a fresh
    `mpirun` re-does the GPU bring-up that the crash comment in
    GPUControl.sycl.cpp says Aurora sometimes botches) rather than a genuine
    problem with this parameter point, so we deliberately do NOT record a
    failed attempt as an infinite/placeholder loss -- that would poison the
    GP surrogate with a fake data point for what's probably just bad luck.
    If every attempt fails, the last exception propagates and kills the job
    exactly as before, so a point that's genuinely broken every time still
    surfaces loudly instead of being silently swallowed.

    Each failed attempt's run dir is preserved (see ``_preserve_failed_rundir``)
    before the next attempt's ``clean=True`` would otherwise wipe it, so a
    reproducible failure leaves forensic evidence from every attempt, not just
    the last one. Once some attempt *succeeds*, those preserved dirs are no
    longer useful (the point evaluated fine, so there's nothing to debug) and
    are deleted -- otherwise a point that fails twice then succeeds would
    permanently litter the filesystem with multi-GB failure dirs (core dumps
    included) for a point that turned out to be fine.
    """
    cached = history.lookup(params)
    if cached is not None:
        print(f"[cache hit] {params} -> {cached}")
        return cached

    sim_name = _sim_name_for(params)
    print(f"[evaluate] {sim_name}: {params}")

    rundir = None
    preserved_dirs: list[Path] = []
    for attempt in range(1, max_attempts + 1):
        try:
            rundir = run_abacus(params, sim_name=sim_name, dry_run=dry_run)
            if dry_run:
                loss = 0.0
            else:
                # compute_loss normally takes ~10 s; job 9040160 sat silently in
                # (most likely) here for ~4 h until walltime. If it overruns,
                # dump every thread's Python stack so the next hang is
                # diagnosable from the PBS log. Doesn't abort -- just reports
                # (and repeats every LOSS_HANG_DUMP_S seconds).
                faulthandler.dump_traceback_later(LOSS_HANG_DUMP_S, repeat=True)
                try:
                    loss = compute_loss(rundir)
                finally:
                    faulthandler.cancel_dump_traceback_later()
            break
        except Exception as e:                       # noqa: BLE001 - retry, then re-raise
            print(f"[evaluate] {sim_name} attempt {attempt}/{max_attempts} failed: "
                  f"{type(e).__name__}: {e}")
            # run_abacus tags its exceptions with the run dir even when it
            # failed before returning; compute_loss failures leave `rundir`
            # already set from the successful run_abacus call above.
            failed_dir = getattr(e, "workdir", None) or rundir
            if failed_dir is not None:
                preserved = _preserve_failed_rundir(Path(failed_dir), sim_name, attempt)
                if preserved is not None:
                    preserved_dirs.append(preserved)
            if attempt == max_attempts:
                raise

    # Succeeded (possibly after retries) -- the preserved failure dirs from
    # earlier attempts on this same point are no longer needed.
    for d in preserved_dirs:
        shutil.rmtree(d, ignore_errors=True)
        print(f"[evaluate] {sim_name}: deleted preserved failure dir {d} (later attempt succeeded)")

    history.add(params, loss, sim_name)
    print(f"[evaluate] {sim_name} -> loss={loss:.6g}")
    return loss


# --------------------------------------------------------------------------- #
# 5. The optimizer (ask/tell, rebuilt from history every invocation)
# --------------------------------------------------------------------------- #
def ask_next(history: History, n_initial: int = 8, seed: int = 0, call_index: int = 0):
    """Pick the next parameter point to evaluate (Bayesian-opt ask step).

    Rebuilds a GP surrogate from the entire history (including primed points) and
    returns its suggestion.  For the first ``n_initial`` points skopt returns a
    space-filling (Sobol/Latin) design instead, so the surrogate has something to
    fit.

    Returning *one* point per call keeps the loop simple and maximally
    resumable; skopt's ask(n_points=k) could batch if you want parallel runs.

    ``call_index`` (the loop iteration within this job, from ``optimize()``)
    is folded into the seed -- see the comment below on why a single fixed
    seed across calls is not safe once the initial design is exhausted.
    """
    X_all, Y_all = history.as_XY()

    # The search box is exactly PARAM_SPECS. History points outside it (from
    # earlier, wider boxes, or primed runs) are NOT told to the GP: skopt's Real
    # space rejects out-of-bounds tells, and widening the box to fit them (the
    # old behaviour) meant narrowing PARAM_SPECS silently had no effect. They
    # stay in the history JSON, so widening PARAM_SPECS again brings them back.
    bounds = [spec.search_bounds for spec in PARAM_SPECS]
    eps = 1e-12
    keep = [all(lo - eps <= xi <= hi + eps for xi, (lo, hi) in zip(x, bounds))
            for x in X_all]
    X = [[min(max(xi, lo), hi) for xi, (lo, hi) in zip(x, bounds)]
         for x, k in zip(X_all, keep) if k]
    Y = [y for y, k in zip(Y_all, keep) if k]
    if len(X) < len(X_all):
        print(f"[ask_next] using {len(X)}/{len(X_all)} history points inside the "
              f"PARAM_SPECS box ({len(X_all) - len(X)} outside, ignored)")

    # initial_point_generator="sobol" is load-bearing, not cosmetic. The default
    # ("random") leaves skopt's _initial_samples=None, so during the initial-
    # design phase ask() draws live from self.rng. Because we rebuild a fresh
    # Optimizer every invocation, a fixed seed would make that first draw
    # identical every time -- once it's evaluated and cached, every later call
    # re-asks the same point and gets a permanent cache hit (the sim never runs).
    # A deterministic Sobol design is stored as an indexed sequence instead, so
    # telling N points advances the index to N and ask() returns a *new* point,
    # regardless of what random_state was used to generate that sequence.
    #
    # That freed us to vary random_state by call_index -- and past the initial
    # design (once the GP/EI acquisition is driving ask()), we *must*: skopt's
    # duplicate-point fallback (Optimizer.ask(), "using random point ...") draws
    # from the same self.rng, seeded fresh from random_state on every rebuild.
    # A fixed seed there means: proposed point is a duplicate -> fallback draws
    # the same "random" substitute every call -> that substitute is now cached
    # too -> next call proposes the same duplicate again -> same substitute
    # cache-hits again -> forever, with history never growing (job 8735262:
    # burned all 333 max-evals on this dead loop after only 5 new evals).
    opt = Optimizer(
        dimensions=[Real(lo, hi) for lo, hi in bounds],
        base_estimator="GP",
        acq_func="EI",                     # expected improvement
        n_initial_points=n_initial,
        initial_point_generator="sobol",
        random_state=seed + call_index,
    )
    if X:
        opt.tell(X, Y)                     # reconstruct state from history
    return opt.ask()


# --------------------------------------------------------------------------- #
# 6. Driver
# --------------------------------------------------------------------------- #
def optimize(max_evals: int = 1, dry_run: bool = False, prime_dirs=None):
    """Run up to ``max_evals`` new Abacus evaluations, then stop.

    Designed to be called once per PBS job: it appends however many runs fit in
    the walltime and persists everything, so the next job resumes seamlessly.
    Set ``max_evals`` to how many full runs you expect to fit in the job.

    ``prime_dirs`` (defaults to PRIME_DIRS) are existing runs folded into the
    history before the loop starts; this is idempotent across jobs (already-seen
    parameter points are skipped), so it is safe to leave configured.
    """
    history = History()
    target = None if dry_run else reference_target()  # compute & cache once up front
    if not dry_run:
        # Loss hyper-parameters (R_EDGES, TARGET_DATASET, ...) changed since some
        # records were computed? Recompute them from their maplogs (~1 s each).
        history.refresh_stale(target)

    dirs = PRIME_DIRS if prime_dirs is None else prime_dirs
    if dirs and not dry_run:
        prime_from_dirs(dirs, history, target)

    for i in range(max_evals):
        x = ask_next(history, call_index=i)
        params = vector_to_params(x)
        evaluate(params, history, dry_run=dry_run)
        best = history.best()
        print(f"[optimize] {len(history.current())} evals; "
              f"best loss={best['loss']:.6g} at {best['params']}")
    return history.best()


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-evals", type=int, default=50,
                   help="number of new Abacus runs to do this invocation")
    p.add_argument("--dry-run", action="store_true",
                   help="exercise the loop without running Abacus or computing loss")
    p.add_argument("--prime-file", default=None,
                   help="file of existing run dirs (one per line) to seed the "
                        "search with, in addition to PRIME_DIRS")
    p.add_argument("--prime-only", action="store_true",
                   help="only fold in the primed runs (max-evals forced to 0)")
    args = p.parse_args()

    prime_dirs = list(PRIME_DIRS)
    if args.prime_file:
        prime_dirs += Path(args.prime_file).read_text().splitlines()

    best = optimize(max_evals=0 if args.prime_only else args.max_evals,
                    dry_run=args.dry_run, prime_dirs=prime_dirs)
    print("Best so far:", best)


if __name__ == "__main__":
    main()
