#!/usr/bin/env python
"""Optimize MAP parameters to match a reference correlation function.

Usage
-----
    python optimize_maps.py --config tuning.toml [--max-evals N] [--dry-run]
                            [--prime-file F] [--prime-only]
                            [--import-history OLD.json ...]

Everything that defines a tuning campaign -- the par2 to run, where outputs go,
the parameters being tuned (``[[params]]``), the loss definition, the
reference dataset -- comes from the TOML config passed with ``--config``
(see ``load_config`` and the example ``tuning.toml`` files next to the PBS
scripts). Nothing is hard-coded per campaign.

Goal
----
Tune MAP parameters so the Abacus MAP two-point correlation function matches
a reference dataset's (TNG, FLAMINGO, or an Abacus fiducial MAP catalog; see
``target_dataset``). Each "function evaluation" is a *full Abacus run*, so it
is extremely expensive (hours, multiple nodes). The design is built around:

1. **Resumability / caching.** Every evaluated point and its loss are
   persisted to a JSON history on disk. The optimizer state is
   *reconstructed from that history* on every invocation (ask/tell), so the
   search survives PBS-job boundaries. No evaluation is ever repeated.

2. **One history per parameter spec.** The history lives in
   ``<output_dir>/spec_<hash>/``, where the hash is of the ``[[params]]``
   list (names, bounds, log flags). Changing the spec -- adding/dropping a
   parameter or moving a bound -- therefore starts a fresh history (and a
   fresh run-directory tree) automatically, instead of mixing points from
   different search spaces in one GP. Older points can be carried over
   deliberately with ``--import-history`` (only points whose parameter names
   match the spec exactly and that lie inside its bounds are taken).

3. **One loss definition per number.** Each history record is tagged with
   the ``loss_config()`` it was computed under (binning, redshifts,
   abundance, target dataset + source, LOSS_VERSION). Records computed under
   a different loss_config are *stale*: ignored by the optimizer and
   recomputed from their on-disk maplogs at startup (``History.refresh_stale``).

4. **Sample efficiency.** Bayesian optimization (GP surrogate, skopt) -- the
   standard tool for "few, expensive, noisy" black boxes. Merger criteria
   are thresholds, so the loss surface is locally staircase-like; a surrogate
   tolerates that where raw finite differences would not.

5. **Runs that form no MAPs** (formation criterion too strict) are recorded
   as unscored ``status: "no_maps"`` points and shown to the optimizer at the
   history's current worst real loss (``History.penalty_loss``) rather than
   being retried forever or poisoning the GP with an arbitrary huge number.
"""

from __future__ import annotations

import argparse
import faulthandler
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# utilities.py lives in ../utilities (MAP_analysis repo layout).
UTILITIES_DIR = Path(__file__).resolve().parents[1] / "utilities"
if str(UTILITIES_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITIES_DIR))

# utilities centralizes maplog reading + correlation functions. Reuse them so
# the loss is computed exactly the way the analysis notebooks visualize these
# curves.
from utilities import (  # noqa: E402
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
import illustris_python as il  # noqa: E402
from astropy import units as u  # noqa: E402
from skopt import Optimizer  # noqa: E402
from skopt.space import Real  # noqa: E402


# Bump whenever compute_loss()'s *formula* changes (weighting, residual form,
# error model, ...) -- anything loss_config() can't see from the settings
# alone. Every history record computed under an older version is then
# recomputed.
LOSS_VERSION = 1

HISTORY_NAME = "optimize_maps_history.json"
SPEC_NAME = "param_spec.json"


# --------------------------------------------------------------------------- #
# 1. Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ParamSpec:
    """One tunable Abacus parameter.

    ``log=True`` means the optimizer searches in log10 space, which is the
    right choice for positive scale-like quantities (so a factor-of-2 step is
    the same "distance" everywhere). ``low``/``high`` are hard bounds in
    *native* units.
    """
    name: str
    low: float
    high: float
    log: bool = True

    def __post_init__(self):
        if not self.low < self.high:
            raise ValueError(f"param {self.name}: need low < high, got {self.low}, {self.high}")
        if self.log and self.low <= 0:
            raise ValueError(f"param {self.name}: log=true needs low > 0, got {self.low}")

    def to_search(self, value: float) -> float:
        return float(np.log10(value)) if self.log else float(value)

    def from_search(self, s: float) -> float:
        return float(10.0 ** s) if self.log else float(s)

    @property
    def search_bounds(self) -> tuple[float, float]:
        return (self.to_search(self.low), self.to_search(self.high))

    def as_dict(self) -> dict:
        return {"name": self.name, "low": float(self.low),
                "high": float(self.high), "log": bool(self.log)}


def spec_id(params: list[ParamSpec]) -> str:
    """Short stable hash of a parameter spec (order-independent: records store
    params by name, so reordering [[params]] doesn't change the search space)."""
    canon = json.dumps(sorted((p.as_dict() for p in params), key=lambda d: d["name"]),
                       sort_keys=True)
    return hashlib.sha256(canon.encode()).hexdigest()[:10]


@dataclass
class TuningConfig:
    """Everything that defines one tuning campaign. Built by ``load_config``."""
    config_path: Path
    par2: Path
    output_dir: Path
    params: list[ParamSpec]
    prime_dirs: list[str] = field(default_factory=list)
    # [loss]
    target_dataset: str = "abacus_fiducial"
    compare_redshifts: tuple[float, ...] = (4.0, 1.0, 0.2)
    abundance: float = 5.01e-3
    r_spacing: str = "linear"
    r_min: float = 2e-2
    r_max: float = 20.0
    n_r_bins: int = 20
    # [target]
    tng_basepath: str = f"/flare/Abacus/{os.environ.get('USER', '')}/TNG300/TNG300-2/output"
    flamingo_snapshots: dict[float, int] = field(
        default_factory=lambda: {5.0: 10, 2.0: 38, 1.0: 58, 0.2: 74})
    abacus_fiducial_dir: str = "/flare/Abacus/AbacusAurora/AbacusAurora_emulatormax_c000_ph310"
    # [optimizer]
    n_initial: int = 8
    seed: int = 0
    max_attempts: int = 3
    loss_hang_dump_s: float = 600.0
    # [abacus_overrides]: extra par2 overrides applied to every tuning run
    # (on top of the tuned params). LogVerbosity=4 writes per-rank STDLOG
    # output, needed to diagnose crashes. To catch a HeartbeatTimeoutError
    # hang with a live backtrace, add exec_prefix = "<path>/gdb_wrap.sh" here
    # (stalls all ranks at MPI_Init until attached -- only while watching).
    abacus_overrides: dict = field(default_factory=lambda: {"LogVerbosity": 4})

    def __post_init__(self):
        if not self.params:
            raise ValueError("config needs at least one [[params]] entry")
        names = [p.name for p in self.params]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate [[params]] names: {names}")
        if self.target_dataset not in ("flamingo", "tng", "abacus_fiducial"):
            raise ValueError(f"unknown loss.target_dataset {self.target_dataset!r}; "
                             f"expected 'flamingo', 'tng', or 'abacus_fiducial'")
        if self.r_spacing == "log":
            self.r_edges = np.logspace(np.log10(self.r_min), np.log10(self.r_max), self.n_r_bins)
        elif self.r_spacing == "linear":
            self.r_edges = np.linspace(self.r_min, self.r_max, self.n_r_bins)
        else:
            raise ValueError(f"unknown loss.r_spacing {self.r_spacing!r}; expected 'log' or 'linear'")
        self.r_cent = self.r_edges[:-1] + np.diff(self.r_edges) / 2.0

    @property
    def spec_id(self) -> str:
        return spec_id(self.params)

    @property
    def spec_dir(self) -> Path:
        """History + run directories for this exact parameter spec."""
        return self.output_dir / f"spec_{self.spec_id}"

    @property
    def history_path(self) -> Path:
        return self.spec_dir / HISTORY_NAME

    def target_cache_path(self) -> Path:
        # The target depends only on the loss settings, not the param spec,
        # so it's shared by every spec_* under output_dir.
        return self.output_dir / f"{self.target_dataset}_target.npz"


_SECTIONS = {
    "loss": {"target_dataset", "compare_redshifts", "abundance",
             "r_spacing", "r_min", "r_max", "n_r_bins"},
    "target": {"tng_basepath", "flamingo_snapshots", "abacus_fiducial_dir"},
    "optimizer": {"n_initial", "seed", "max_attempts", "loss_hang_dump_s"},
}
_TOP_LEVEL = {"par2", "output_dir", "prime_dirs", "params", "abacus_overrides", *_SECTIONS}


def load_config(path) -> TuningConfig:
    """Read a tuning TOML config. Unknown keys are an error (catches typos).

    Required: ``par2``, ``output_dir``, and at least one ``[[params]]`` table
    (name, low, high, optional log -- default true). Relative ``par2`` and
    ``prime_dirs`` paths are resolved against the config file's directory.
    ``[loss]``, ``[target]``, ``[optimizer]`` and ``[abacus_overrides]`` are
    optional; see TuningConfig for their defaults.
    """
    path = Path(path).resolve()
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    unknown = set(raw) - _TOP_LEVEL
    if unknown:
        raise ValueError(f"{path}: unknown top-level key(s) {sorted(unknown)}")
    for req in ("par2", "output_dir", "params"):
        if req not in raw:
            raise ValueError(f"{path}: missing required key {req!r}")

    def resolve(p: str) -> Path:
        p = Path(os.path.expandvars(os.path.expanduser(p)))
        return p if p.is_absolute() else (path.parent / p)

    kw: dict = {}
    for sec, allowed in _SECTIONS.items():
        tbl = raw.get(sec, {})
        bad = set(tbl) - allowed
        if bad:
            raise ValueError(f"{path}: unknown key(s) in [{sec}]: {sorted(bad)}")
        kw.update(tbl)
    if "compare_redshifts" in kw:
        kw["compare_redshifts"] = tuple(float(z) for z in kw["compare_redshifts"])
    if "flamingo_snapshots" in kw:      # TOML keys are strings
        kw["flamingo_snapshots"] = {float(z): int(s) for z, s in kw["flamingo_snapshots"].items()}

    params = []
    for p in raw["params"]:
        bad = set(p) - {"name", "low", "high", "log"}
        if bad:
            raise ValueError(f"{path}: unknown key(s) in [[params]]: {sorted(bad)}")
        params.append(ParamSpec(p["name"], float(p["low"]), float(p["high"]),
                                bool(p.get("log", True))))

    cfg = TuningConfig(
        config_path=path,
        par2=resolve(raw["par2"]),
        output_dir=resolve(raw["output_dir"]),
        params=params,
        prime_dirs=[str(resolve(d)) for d in raw.get("prime_dirs", [])],
        **kw,
    )
    if "abacus_overrides" in raw:
        cfg.abacus_overrides = dict(raw["abacus_overrides"])
    if not cfg.par2.is_file():
        raise FileNotFoundError(f"{path}: par2 {cfg.par2} not found")
    return cfg


# The active configuration. Set once by configure(); every function below
# reads its settings from here.
CFG: TuningConfig | None = None


def configure(cfg: TuningConfig) -> TuningConfig:
    global CFG
    CFG = cfg
    return cfg


def vector_to_params(x) -> dict[str, float]:
    """Optimizer search-space vector -> {param_name: native_value}."""
    return {spec.name: spec.from_search(xi) for spec, xi in zip(CFG.params, x)}


def params_to_vector(params: dict[str, float]):
    """{param_name: native_value} -> optimizer search-space vector."""
    return [spec.to_search(params[spec.name]) for spec in CFG.params]


def _nthreads() -> int:
    return int(os.environ.get("OMP_NUM_THREADS", 26))


# --------------------------------------------------------------------------- #
# 2. The expensive part: run Abacus with overridden parameters
# --------------------------------------------------------------------------- #
def run_abacus(params: dict[str, float], sim_name: str | None = None,
               dry_run: bool = False) -> Path:
    """Run one Abacus simulation of CFG.par2 with ``params`` overridden;
    return its run dir.

    Each evaluation gets a *distinct* SimName so its outputs (maplogs, state,
    logs) never clobber another point's -- essential for caching and for
    keeping the history auditable. Returns the run's WorkingDirectory (holds
    abacus.par, write/state, log/, and maplogs/ as read by read_maplog_dir).
    """
    if sim_name is None:
        sim_name = _sim_name_for(params)

    # The par2 has `SimName = "$SIMNAME$"`, and `$VAR$` tokens are substituted
    # from os.environ at par2 *parse* time (gen_param.shellreplace), which runs
    # before any `params=` override is applied -- so a SimName override can't
    # prevent the "Variable 'SIMNAME' is not defined" crash. Set the env var to
    # our per-point name; it drives the whole OutputDirectory/WorkingDirectory
    # tree via @SimName@.
    os.environ["SIMNAME"] = sim_name

    # Import here so --help / dry-run don't require the build to be importable.
    from abacus.sim import abacus

    run_params = dict(params)
    run_params.update(CFG.abacus_overrides)
    sim = abacus.Simulation(par2file=str(CFG.par2), params=run_params)
    workdir = Path(sim.param["WorkingDirectory"])

    if dry_run:
        print(f"[dry-run] would run {sim_name} with {params}")
        return workdir

    # clean=True: fresh run (IC is preserved by _clean unless erase_ic). The
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


def _rundir_for(sim_name: str) -> Path:
    """Run dir of a history record (an absolute sim_name is used as-is)."""
    return _output_root() / sim_name


def _sim_name_for(params: dict[str, float], base: str = "MapTune") -> str:
    """Deterministic, filesystem-safe SimName from the parameter values.

    The prefix is CFG.spec_dir relative to $ABACUS_OUTPUT_ROOT (the par2's
    OutputDirectory is "$ABACUS_OUTPUT_ROOT/" + @SimName@), so each spec's
    runs sit beside its history. The decimal-point/minus-sign substitution
    keeps the leaf filesystem-safe while leaving the "/" separators intact.
    """
    out_root = _output_root()
    try:
        prefix = CFG.spec_dir.resolve().relative_to(out_root.resolve())
    except ValueError:
        raise RuntimeError(
            f"output_dir ({CFG.output_dir}) must be under ABACUS_OUTPUT_ROOT "
            f"({out_root}) so run directories can be nested beside the history")
    parts = [base]
    for spec in CFG.params:
        parts.append(f"{spec.name.replace('Map_', '')}_{params[spec.name]:.5g}")
    leaf = "_".join(parts).replace(".", "p").replace("-", "m")
    return f"{prefix}/{leaf}"


# --------------------------------------------------------------------------- #
# 3. The loss: discrepancy between Abacus and the reference correlation function
# --------------------------------------------------------------------------- #
def _target_source() -> str:
    """The on-disk source of the active target dataset's reference curves."""
    return {"tng": CFG.tng_basepath,
            "flamingo": str(FLAMINGO_CACHE_DIR),
            "abacus_fiducial": CFG.abacus_fiducial_dir}[CFG.target_dataset]


def loss_config() -> dict:
    """Every hyper-parameter that defines what compute_loss() returns.

    Stored on each history record and in the target cache; a mismatch with
    the current value means the stored loss/target is stale. The bin edges
    themselves are stored (not r_spacing/r_min/r_max/n_r_bins), so any change
    to the binning is caught.
    """
    return {
        "loss_version": LOSS_VERSION,
        "target_dataset": CFG.target_dataset,
        "target_source": _target_source(),
        "compare_redshifts": [float(z) for z in CFG.compare_redshifts],
        "abundance": float(CFG.abundance),
        "r_edges": [float(f"{r:.10g}") for r in CFG.r_edges],
    }


def _tng_xi_at(z: float, ztol: float = 5e-2):
    """TNG galaxy correlation function (abundance-matched) at redshift ``z``.

    Mirrors the TNG branch of utilities.compare_correlation_functions:
    subhalos thresholded on stellar mass to the target abundance, correlated
    with get_xi. Returns (xi, poisson) on r_cent.
    """
    star_type = il.snapshot.partTypeNum("star")
    snapnos, zs = get_available_tng_snapshots(CFG.tng_basepath)
    k = int(np.argmin(np.abs(zs - z)))
    if abs(zs[k] - z) > ztol:
        raise ValueError(f"no TNG snapshot within {ztol} of z={z}; available: {zs}")
    sn = int(snapnos[k])

    boxsize = il.groupcat.loadHeader(CFG.tng_basepath, sn)["BoxSize"] / 1e3 * u.Mpc
    subs = load_tng_subhalos(CFG.tng_basepath, sn)
    m = subs["SubhaloMassType"][:, star_type]
    thresh = get_threshold(m, boxsize, guess=1e-3, abundance=CFG.abundance)
    msk = m >= thresh
    twopt = get_xi(subs["SubhaloPos"][msk], CFG.r_edges, boxsize,
                   max_p=np.inf, nthreads=_nthreads())
    xi = twopt.get_corr(return_sep=False)
    poisson = np.sqrt(twopt.D1D2.wcounts) / twopt.R1R2.wcounts
    return np.asarray(xi), np.asarray(poisson)


def _flamingo_xi_at(z: float, ztol: float = 5e-2):
    """FLAMINGO halo correlation function (abundance-matched) at redshift ``z``.

    SOAP halos thresholded on stellar mass to the target abundance. A
    snapshot not yet cached under FLAMINGO_CACHE_DIR needs
    load_flamingo_halos(..., root_dir=hdfstream.open('cosma', '/')) once.
    """
    zs = np.array(sorted(CFG.flamingo_snapshots))
    k = int(np.argmin(np.abs(zs - z)))
    if abs(zs[k] - z) > ztol:
        raise ValueError(f"no FLAMINGO snapshot within {ztol} of z={z}; available: {zs} "
                         f"(add it to [target] flamingo_snapshots)")
    step = CFG.flamingo_snapshots[float(zs[k])]

    flam = load_flamingo_halos(step, cache_dir=FLAMINGO_CACHE_DIR)
    boxsize = flam["boxsize"] * u.Mpc
    thresh = get_threshold(flam["mass"], boxsize, guess=1e-3, abundance=CFG.abundance)
    msk = flam["mass"] >= thresh
    twopt = get_xi(flam["pos"][msk], CFG.r_edges, boxsize,
                   max_p=np.inf, nthreads=_nthreads())
    xi = twopt.get_corr(return_sep=False)
    poisson = np.sqrt(twopt.D1D2.wcounts) / twopt.R1R2.wcounts
    return np.asarray(xi), np.asarray(poisson)


_fiducial_maplog_cache: dict | None = None

# get_map_xi only touches pos/mult/control (+ header); timestep is added
# internally by read_maplog_dir. Read nothing else, and only the
# compare_redshifts epochs.
_XI_KEYS = ["pos", "mult", "control"]


def _read_xi_maplog(path, nworkers=None):
    kw = {} if nworkers is None else {"nworkers": nworkers}
    return read_maplog_dir(str(path), keys=_XI_KEYS,
                           redshifts=CFG.compare_redshifts, **kw)


def _fiducial_maplog() -> dict:
    """Lazily read + cache abacus_fiducial_dir's maplog (read once, reused for
    every redshift -- it's ~175GB on disk)."""
    global _fiducial_maplog_cache
    if _fiducial_maplog_cache is None:
        _fiducial_maplog_cache = _read_xi_maplog(CFG.abacus_fiducial_dir, nworkers=_nthreads())
    return _fiducial_maplog_cache


def _abacus_fiducial_xi_at(z: float, ztol: float = 5e-2):
    """Fiducial Abacus MAP correlation function at redshift ``z``.

    Compares MAP to MAP (same object definition, same code), isolating the
    effect of resolution (particle mass) on the tuned parameters rather than
    also picking up a cross-code/cross-catalog difference. Reuses get_map_xi
    exactly as compute_loss() does for the tuning run itself.
    """
    maplog = _fiducial_maplog()
    ts, zm = map_step_for_redshift(maplog, z)
    if abs(zm - z) > ztol:
        raise ValueError(
            f"no {CFG.abacus_fiducial_dir} MAP timeslice within {ztol} of z={z} "
            f"(nearest available: z={zm}); check its MapLogRedshifts/FinalRedshift "
            f"and adjust loss.compare_redshifts")
    xi, poisson = get_map_xi(maplog, ts, CFG.r_edges, abundance=CFG.abundance,
                             max_p=np.inf, nthreads=_nthreads())
    return np.asarray(xi), np.asarray(poisson)


def _reference_xi_at(z: float, ztol: float = 5e-2):
    return {"flamingo": _flamingo_xi_at,
            "tng": _tng_xi_at,
            "abacus_fiducial": _abacus_fiducial_xi_at}[CFG.target_dataset](z, ztol=ztol)


def reference_target(cache_path: Path | None = None) -> dict:
    """Compute (once) and cache the reference target curves at compare_redshifts.

    Cached per target dataset under output_dir and reused only if the
    loss_config() it was built under matches the current one.
    """
    if cache_path is None:
        cache_path = CFG.target_cache_path()
    cfg = json.dumps(loss_config(), sort_keys=True)
    if cache_path.exists():
        d = np.load(cache_path)
        if "loss_config" in d.files and str(d["loss_config"]) == cfg:
            return {float(z): (d[f"xi_{i}"], d[f"err_{i}"])
                    for i, z in enumerate(CFG.compare_redshifts)}
        print(f"[reference_target] {cache_path} was built under a different "
              f"loss_config; rebuilding")
    target = {}
    save = {"redshifts": np.asarray(CFG.compare_redshifts),
            "r_edges": np.asarray(CFG.r_edges),   # bins the xi_/err_ arrays live on
            "loss_config": np.asarray(cfg)}
    for i, z in enumerate(CFG.compare_redshifts):
        xi, err = _reference_xi_at(z)
        target[float(z)] = (xi, err)
        save[f"xi_{i}"], save[f"err_{i}"] = xi, err
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **save)
    # The fiducial maplog is huge and only needed to build the target: release
    # it so it doesn't sit in this process while the tuning sims run.
    global _fiducial_maplog_cache
    _fiducial_maplog_cache = None
    gc.collect()
    return target


class NoMapsFormed(RuntimeError):
    """The run completed but wrote no maplog files because no MAP ever formed.

    Abacus creates maplogs/<col>/ up front, but ASDF_Output::Aggregate_and_Write
    skips creating a file when it has no input buffers -- so a parameter point
    whose formation criterion is too strict for any pair to pass (e.g.
    Map_Aform=0.05; job 9044600) leaves an empty maplogs/col*/ tree. That's a
    deterministic property of the parameters, not a flaky run: evaluate()
    records it as a penalty point instead of retrying.
    """


def _final_forest_usage(rundir) -> int | None:
    """IncomingMapForestUsage (MPI-summed MAP node count) from the run's final
    state file, or None if it can't be found."""
    for state in sorted((Path(rundir) / "log" / "last").glob("step*.state")):
        for line in state.read_text().splitlines():
            key, _, val = line.partition("=")
            if key.strip() == "IncomingMapForestUsage":
                return int(val.strip())
    return None


def _check_maps_formed(rundir) -> None:
    """Raise NoMapsFormed if the run positively formed no MAPs.

    An empty maplogs/ alone is ambiguous -- zero-MAP output and a lost write
    look identical on disk -- so also require the final state to report zero
    MAP forest usage. Anything else (no maplogs/ dir, missing state, nonzero
    usage) falls through to read_maplog_dir's FileNotFoundError and
    evaluate()'s normal retry path.
    """
    mdir = Path(rundir) / "maplogs"
    if not mdir.is_dir() or any(mdir.glob("*/maplog_z*_*.asdf")):
        return
    usage = _final_forest_usage(rundir)
    if usage == 0:
        raise NoMapsFormed(f"{mdir} holds no maplog files and the final state "
                           f"reports IncomingMapForestUsage=0: no MAPs formed")
    print(f"[check] {mdir} holds no maplog files but IncomingMapForestUsage="
          f"{usage}; treating as lost output, not a no-MAP point")


def compute_loss(rundir: Path | None = None, maplog: dict | None = None,
                 target: dict | None = None) -> float:
    """Loss = chi-square-like distance between Abacus and reference xi over (z, r).

    Pass either a ``rundir`` (the maplog is read from it) or an already-loaded
    ``maplog`` dict. Raises NoMapsFormed if ``rundir`` positively formed no MAPs.

    For each comparison redshift, select the maplog timeslice nearest that z,
    abundance-match on MAP multiplicity (get_map_xi), and compare to the
    cached reference target. The residual is on ``r^2 * xi`` (how these
    curves are read by eye), normalized by the combined Poisson error so
    noisy/empty bins don't dominate:

        L = sum_{z,r} ( r^2 (xi_map - xi_ref) )^2 / ( (r^2 sigma)^2_map + ..._ref )

    CAVEAT: merger criteria are hard thresholds, so the loss is expected to
    be locally piecewise-flat with steps -- why the optimizer is a GP
    surrogate rather than a gradient method.

    TODO: decide whether to weight redshifts, fit log(xi) instead of r^2 xi,
    or add an abundance/HOD penalty term. Bump LOSS_VERSION if you do.
    """
    if target is None:
        target = reference_target()
    if maplog is None:
        _check_maps_formed(rundir)
        t0 = time.perf_counter()
        print(f"[compute_loss] reading maplog {rundir} ...")
        maplog = _read_xi_maplog(rundir)
        print(f"[compute_loss] maplog read in {time.perf_counter() - t0:.1f} s")

    loss = 0.0
    w = CFG.r_cent ** 2
    for z, (xi_ref, err_ref) in target.items():
        ts, _zm = map_step_for_redshift(maplog, z)
        t0 = time.perf_counter()
        xi_map, err_map = get_map_xi(maplog, ts, CFG.r_edges, abundance=CFG.abundance,
                                     max_p=np.inf, nthreads=_nthreads())
        print(f"[compute_loss] z={z}: get_map_xi in {time.perf_counter() - t0:.1f} s")
        num = (w * (xi_map - xi_ref)) ** 2
        den = (w * err_map) ** 2 + (w * err_ref) ** 2
        good = np.isfinite(num) & np.isfinite(den) & (den > 0)
        loss += float(np.sum(num[good] / den[good]))
    return loss


# --------------------------------------------------------------------------- #
# 3b. Priming: seed the search from existing runs
# --------------------------------------------------------------------------- #
def params_from_maplog(maplog: dict) -> dict[str, float]:
    """Extract the [[params]] values from a loaded maplog.

    read_maplog_dir normalizes both the new (Map_*) and old
    (a_merge/b_merge/...) parameter names into Map_* keys, so primed runs
    with legacy parameter names still resolve here.
    """
    out = {}
    for spec in CFG.params:
        if spec.name not in maplog:
            raise KeyError(f"{spec.name} not found in maplog params; this primed run "
                           f"does not record it. Can't fold it into the search.")
        out[spec.name] = float(getattr(maplog[spec.name], "value", maplog[spec.name]))
    return out


def prime_from_dirs(dirs, history: "History", target: dict) -> None:
    """Fold each existing run directory into the history as a (params, loss).

    Skips runs already in the history, runs outside the spec's bounds, and
    runs that error out (e.g. missing maplogs) with a warning, so one bad
    directory can't abort the whole priming pass.
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
        if not in_bounds(params):
            print(f"[prime] SKIP {d}: {params} is outside the [[params]] bounds")
            continue
        if history.lookup(params) is not None:
            print(f"[prime] already have {params}; skipping {d}")
            continue
        loss = compute_loss(maplog=maplog, target=target)
        # sim_name convention: run dir relative to ABACUS_OUTPUT_ROOT. A dir
        # outside that root keeps its absolute path, which _rundir_for also
        # resolves (pathlib drops the root for an absolute right operand).
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
def in_bounds(params: dict[str, float], tol: float = 1e-9) -> bool:
    """True if ``params`` has exactly the [[params]] names, each within bounds."""
    if set(params) != {s.name for s in CFG.params}:
        return False
    return all(s.low - tol <= params[s.name] <= s.high + tol for s in CFG.params)


class History:
    """JSON-backed list of {params, loss, sim_name, loss_config[, status]}.

    Doubles as the cache (never re-run an evaluated point) and as the
    optimizer state (the optimizer is rebuilt by telling it the whole history
    each run). One History per parameter spec: see TuningConfig.spec_dir.

    Records whose ``loss_config`` differs from the current loss_config() are
    *stale*: kept on disk but ignored by lookup/as_XY/best until
    refresh_stale() recomputes them, so two loss definitions never mix in the
    GP's training data.

    A record may instead carry ``loss: null`` plus ``status: "no_maps"`` (see
    NoMapsFormed): the run completed but produced nothing to score. That
    doesn't depend on the loss definition, so such records are never stale.
    The optimizer sees them at ``penalty_loss()`` -- the worst current real
    loss -- computed fresh each time rather than frozen into the JSON, so it
    tracks loss-definition changes and never dwarfs the real losses enough to
    flatten the GP surrogate.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.records: list[dict] = []
        if self.path.exists():
            self.records = json.loads(self.path.read_text())

    @staticmethod
    def _key(params: dict[str, float]) -> tuple:
        # Relative rounding so tiny numeric jitter (e.g. 10**log10(x)) maps to
        # the same key; keyed by name so it doesn't depend on [[params]] order.
        return tuple(sorted((k, float(f"{float(v):.10g}")) for k, v in params.items()))

    @staticmethod
    def is_scored(r: dict) -> bool:
        return r["loss"] is not None and math.isfinite(r["loss"])

    @staticmethod
    def is_current(r: dict) -> bool:
        return r.get("status") == "no_maps" or r.get("loss_config") == loss_config()

    def current(self) -> list[dict]:
        return [r for r in self.records if self.is_current(r)]

    def penalty_loss(self) -> float | None:
        """Loss reported for unscored (no-MAP) points: the worst current real
        loss, or None if there isn't one yet."""
        losses = [r["loss"] for r in self.current() if self.is_scored(r)]
        return max(losses) if losses else None

    def _effective_loss(self, r: dict) -> float | None:
        return r["loss"] if self.is_scored(r) else self.penalty_loss()

    def lookup(self, params: dict[str, float]):
        """Effective loss for ``params`` if already evaluated, else None.

        An unscored record is still a cache hit (returns the penalty, or NaN
        if no real loss exists yet to derive one from) -- it must never
        trigger a re-run.
        """
        key = self._key(params)
        for r in self.current():
            if self._key(r["params"]) == key:
                eff = self._effective_loss(r)
                return float("nan") if eff is None else eff
        return None

    def save(self, path: Path | None = None):
        # Atomic write: a walltime kill mid-write can't corrupt the history.
        dest = Path(path) if path is not None else self.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_text(json.dumps(self.records, indent=2))
        os.replace(tmp, dest)

    def add(self, params: dict[str, float], loss: float | None, sim_name: str,
            status: str | None = None, save: bool = True):
        # A stale record for the same point (its run dir was missing, so it
        # couldn't be refreshed) is superseded by this fresh evaluation.
        key = self._key(params)
        self.records = [r for r in self.records
                        if self.is_current(r) or self._key(r["params"]) != key]
        rec = {"params": params, "loss": loss, "sim_name": sim_name,
               "loss_config": loss_config()}
        if status is not None:
            rec["status"] = status
        self.records.append(rec)
        if save:
            self.save()

    def refresh_stale(self, target: dict, force: bool = False,
                      out_path: Path | None = None) -> None:
        """Recompute the loss of every stale record (every scored record if
        ``force``) from its on-disk maplog under the current loss_config(),
        then save (to ``out_path`` if given).

        A record whose run dir is missing or unreadable stays stale --
        excluded from the optimizer -- rather than keeping a loss under the
        old definition.
        """
        todo = [r for r in self.records
                if r.get("status") != "no_maps" and (force or not self.is_current(r))]
        if not todo:
            return
        cfg = loss_config()
        print(f"[history] recomputing {len(todo)}/{len(self.records)} losses under "
              f"loss_config={cfg}")
        n_ok = 0
        for r in todo:
            rundir = _rundir_for(r["sim_name"])
            old = r["loss"]
            try:
                new = compute_loss(maplog=_read_xi_maplog(rundir), target=target)
            except Exception as e:                   # noqa: BLE001 - report & skip
                print(f"[history] STALE {rundir}: {type(e).__name__}: {e}; "
                      f"excluded from the optimizer until it can be recomputed")
                continue
            r["loss"], r["loss_config"] = new, cfg
            n_ok += 1
            print(f"[history] {r['sim_name']}: {old} -> {new:.6g}")
        self.save(out_path)
        print(f"[history] recomputed {n_ok}/{len(todo)}; "
              f"{len(self.records) - len(self.current())} record(s) remain stale")

    def as_XY(self):
        """Return (X_search_vectors, Y_losses) for telling the optimizer.

        Unscored records get ``penalty_loss()``, or are left out while no
        real loss exists to derive it from.
        """
        penalty = self.penalty_loss()
        X, Y = [], []
        for r in self.current():
            y = r["loss"] if self.is_scored(r) else penalty
            if y is None:
                continue
            X.append(params_to_vector(r["params"]))
            Y.append(y)
        return X, Y

    def best(self):
        recs = [r for r in self.current() if self.is_scored(r)]
        return min(recs, key=lambda r: r["loss"]) if recs else None


def open_history() -> History:
    """The active spec's History. Creates spec_dir (with a param_spec.json
    describing it) on first use, and reports when the spec is new so a
    PARAM_SPEC change is visible in the job log."""
    spec_dir = CFG.spec_dir
    spec_file = spec_dir / SPEC_NAME
    desc = {"spec_id": CFG.spec_id, "params": [p.as_dict() for p in CFG.params],
            "par2": str(CFG.par2), "config": str(CFG.config_path)}
    if not spec_file.exists():
        others = sorted(p for p in CFG.output_dir.glob("spec_*") if p != spec_dir)
        print(f"[history] new parameter spec {CFG.spec_id}: starting a fresh history "
              f"in {spec_dir}")
        for o in others:
            h = o / HISTORY_NAME
            n = len(json.loads(h.read_text())) if h.exists() else 0
            print(f"[history]   (other spec under {CFG.output_dir}: {o.name}, {n} records; "
                  f"carry points over with --import-history {h})")
        legacy = CFG.output_dir / HISTORY_NAME
        if legacy.exists():
            print(f"[history]   (pre-spec history {legacy} exists; carry its points over "
                  f"with --import-history {legacy})")
        spec_dir.mkdir(parents=True, exist_ok=True)
        spec_file.write_text(json.dumps(desc, indent=2) + "\n")
    hist = History(CFG.history_path)
    print(f"[history] spec {CFG.spec_id} ({', '.join(p.name for p in CFG.params)}): "
          f"{len(hist.records)} record(s) in {CFG.history_path}")
    return hist


def import_history(src: Path, history: History) -> None:
    """Copy records from another history JSON into ``history``.

    Only records whose parameter names exactly match [[params]] and that lie
    inside its bounds are taken; points already present are skipped. Each
    record keeps its sim_name (so its run dir is still found) and its
    loss_config (so refresh_stale recomputes it if the loss definition
    differs). Idempotent, so safe to leave on the command line.
    """
    src = Path(src)
    recs = json.loads(src.read_text())
    have = {History._key(r["params"]) for r in history.records}
    n_new = n_dup = n_out = 0
    for r in recs:
        if not in_bounds(r["params"]):
            n_out += 1
            continue
        if History._key(r["params"]) in have:
            n_dup += 1
            continue
        history.records.append(dict(r, imported_from=str(src)))
        have.add(History._key(r["params"]))
        n_new += 1
    if n_new:
        history.save()
    print(f"[import] {src}: imported {n_new}, already present {n_dup}, "
          f"skipped {n_out} (different parameter names or outside bounds)")


# --------------------------------------------------------------------------- #
# 4b. The cached objective
# --------------------------------------------------------------------------- #
def _preserve_failed_rundir(workdir: Path, sim_name: str, attempt: int) -> Path | None:
    """Move a failed attempt's run dir aside so the next retry's ``clean=True``
    doesn't wipe it -- keeps every attempt's logs/core dump/status.log/maplogs
    for post-mortem. Returns the preserved path (or None if there was nothing
    to move), so a later success can clean these up again.

    The other dirs ``Simulation._clean`` wipes are all subdirectories of (or
    equal to) ``WorkingDirectory`` in these par2s, so moving just this one
    directory captures everything.
    """
    if not workdir.exists():
        return None
    dest = workdir.parent / f"{workdir.name}_failed_attempt{attempt}"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(workdir), str(dest))
    print(f"[evaluate] {sim_name}: preserved failed attempt {attempt} at {dest}")
    return dest


def evaluate(params: dict[str, float], history: History, dry_run: bool = False) -> float:
    """Cached objective: return loss for ``params``, running Abacus if needed.

    A run (or the maplog read afterward) is retried up to
    ``optimizer.max_attempts`` times on failure before giving up. The
    failures seen so far -- a SIGSEGV inside the GPU driver, a
    HeartbeatTimeout hang, a corrupted maplog multiplicity -- look like
    transient GPU/driver flakiness rather than a problem with this parameter
    point, so a failed attempt is NOT recorded as a placeholder loss (that
    would poison the GP with a fake data point). If every attempt fails, the
    last exception propagates and kills the job.

    The one exception is NoMapsFormed: the run completed fine but the
    parameters were too strict for any MAP to form. That's deterministic, so
    it is NOT retried; it's recorded as an unscored ``status: "no_maps"``
    point that the optimizer sees at the history's penalty loss.

    Each failed attempt's run dir is preserved (``_preserve_failed_rundir``)
    and deleted again once a later attempt succeeds.
    """
    cached = history.lookup(params)
    if cached is not None:
        print(f"[cache hit] {params} -> {cached}")
        return cached

    sim_name = _sim_name_for(params)
    print(f"[evaluate] {sim_name}: {params}")

    max_attempts = CFG.max_attempts
    rundir = None
    preserved_dirs: list[Path] = []
    status = None
    for attempt in range(1, max_attempts + 1):
        try:
            rundir = run_abacus(params, sim_name=sim_name, dry_run=dry_run)
            if dry_run:
                loss = 0.0
            else:
                # compute_loss normally takes ~10 s; job 9040160 sat silently
                # in (most likely) here for ~4 h until walltime. If it
                # overruns, dump every thread's Python stack (repeating) so a
                # hang is diagnosable from the PBS log. Doesn't abort.
                faulthandler.dump_traceback_later(CFG.loss_hang_dump_s, repeat=True)
                try:
                    loss = compute_loss(rundir)
                finally:
                    faulthandler.cancel_dump_traceback_later()
            break
        except NoMapsFormed as e:
            print(f"[evaluate] {sim_name}: {e}; recording as a penalty point (no retry)")
            loss, status = None, "no_maps"
            break
        except Exception as e:                       # noqa: BLE001 - retry, then re-raise
            print(f"[evaluate] {sim_name} attempt {attempt}/{max_attempts} failed: "
                  f"{type(e).__name__}: {e}")
            # run_abacus tags its exceptions with the run dir even when it
            # failed before returning; compute_loss failures leave `rundir`
            # set from the successful run_abacus call above.
            failed_dir = getattr(e, "workdir", None) or rundir
            if failed_dir is not None:
                preserved = _preserve_failed_rundir(Path(failed_dir), sim_name, attempt)
                if preserved is not None:
                    preserved_dirs.append(preserved)
            if attempt == max_attempts:
                raise

    # Succeeded (possibly after retries): earlier failure dirs aren't needed.
    for d in preserved_dirs:
        shutil.rmtree(d, ignore_errors=True)
        print(f"[evaluate] {sim_name}: deleted preserved failure dir {d} (later attempt succeeded)")

    history.add(params, loss, sim_name, status=status)
    if status is None:
        print(f"[evaluate] {sim_name} -> loss={loss:.6g}")
        return loss
    eff = history.lookup(params)
    print(f"[evaluate] {sim_name} -> {status}; penalty loss={eff:.6g}")
    return eff


# --------------------------------------------------------------------------- #
# 5. The optimizer (ask/tell, rebuilt from history every invocation)
# --------------------------------------------------------------------------- #
def ask_next(history: History, call_index: int = 0):
    """Pick the next parameter point to evaluate (Bayesian-opt ask step).

    Rebuilds a GP surrogate from the entire current history (including primed
    and imported points) and returns its suggestion. For the first
    ``optimizer.n_initial`` points skopt returns a space-filling Sobol design
    instead, so the surrogate has something to fit.

    ``call_index`` (the loop iteration within this job) is folded into the
    seed -- see the comment below on why a single fixed seed across calls is
    not safe once the initial design is exhausted.
    """
    X, Y = history.as_XY()
    bounds = [spec.search_bounds for spec in CFG.params]
    # Every history point is in bounds (the history is per-spec and imports
    # are bounds-checked); clip only to absorb float round-off at the edges,
    # which skopt's Real space would otherwise reject.
    X = [[min(max(xi, lo), hi) for xi, (lo, hi) in zip(x, bounds)] for x in X]

    # initial_point_generator="sobol" is load-bearing, not cosmetic. The
    # default ("random") leaves skopt's _initial_samples=None, so during the
    # initial-design phase ask() draws live from self.rng. Because we rebuild
    # a fresh Optimizer every invocation, a fixed seed would make that first
    # draw identical every time -- once it's evaluated and cached, every later
    # call re-asks the same point and gets a permanent cache hit. A
    # deterministic Sobol design is an indexed sequence instead, so telling N
    # points advances the index to N and ask() returns a *new* point.
    #
    # Past the initial design we must also vary random_state by call_index:
    # skopt's duplicate-point fallback (Optimizer.ask(), "using random point
    # ...") draws from the same self.rng, seeded fresh on every rebuild. A
    # fixed seed there means the same "random" substitute every call -> cached
    # -> same duplicate proposed again -> forever (job 8735262 burned all 333
    # max-evals on this dead loop after only 5 new evals).
    opt = Optimizer(
        dimensions=[Real(lo, hi) for lo, hi in bounds],
        base_estimator="GP",
        acq_func="EI",                     # expected improvement
        n_initial_points=CFG.n_initial,
        initial_point_generator="sobol",
        random_state=CFG.seed + call_index,
    )
    if X:
        opt.tell(X, Y)                     # reconstruct state from history
    return opt.ask()


# --------------------------------------------------------------------------- #
# 6. Driver
# --------------------------------------------------------------------------- #
def optimize(max_evals: int = 1, dry_run: bool = False, prime_dirs=None,
             import_paths=()):
    """Run up to ``max_evals`` new Abacus evaluations, then stop.

    Designed to be called once per PBS job: it appends however many runs fit
    in the walltime and persists everything, so the next job resumes
    seamlessly. ``prime_dirs`` (default: the config's prime_dirs) and
    ``import_paths`` are folded in first; both are idempotent across jobs.
    """
    history = open_history()
    for src in import_paths:
        import_history(src, history)

    target = None if dry_run else reference_target()  # compute & cache once up front
    if not dry_run:
        # Loss settings changed since some records were computed? Recompute
        # them from their maplogs (~1 s each).
        history.refresh_stale(target)

    dirs = CFG.prime_dirs if prime_dirs is None else prime_dirs
    if dirs and not dry_run:
        prime_from_dirs(dirs, history, target)

    for i in range(max_evals):
        x = ask_next(history, call_index=i)
        params = vector_to_params(x)
        evaluate(params, history, dry_run=dry_run)
        best = history.best()
        if best is None:
            print(f"[optimize] {len(history.current())} evals; no scored point yet")
        else:
            print(f"[optimize] {len(history.current())} evals; "
                  f"best loss={best['loss']:.6g} at {best['params']}")
    return history.best()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True,
                   help="tuning TOML config (par2, output_dir, [[params]], [loss], ...)")
    p.add_argument("--max-evals", type=int, default=50,
                   help="number of new Abacus runs to do this invocation")
    p.add_argument("--dry-run", action="store_true",
                   help="exercise the loop without running Abacus or computing loss")
    p.add_argument("--prime-file", default=None,
                   help="file of existing run dirs (one per line) to seed the "
                        "search with, in addition to the config's prime_dirs")
    p.add_argument("--prime-only", action="store_true",
                   help="only fold in primed/imported runs (max-evals forced to 0)")
    p.add_argument("--import-history", action="append", default=[], metavar="JSON",
                   help="copy matching, in-bounds points from another history file "
                        "(e.g. an older spec's or a pre-spec history); repeatable")
    args = p.parse_args()

    cfg = configure(load_config(args.config))
    print(f"[config] {cfg.config_path}: par2={cfg.par2} output_dir={cfg.output_dir} "
          f"spec={cfg.spec_id}")

    prime_dirs = list(cfg.prime_dirs)
    if args.prime_file:
        prime_dirs += Path(args.prime_file).read_text().splitlines()

    best = optimize(max_evals=0 if args.prime_only else args.max_evals,
                    dry_run=args.dry_run, prime_dirs=prime_dirs,
                    import_paths=args.import_history)
    print("Best so far:", best)


if __name__ == "__main__":
    main()
