import os
import multiprocessing
import hashlib
import pickle
from glob import glob
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import re
import random
import time

#matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
from matplotlib import colormaps
from matplotlib.colors import Normalize, LogNorm

#scipy/sklearn
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
from scipy.optimize import minimize_scalar, curve_fit

#pycorr
from pycorr import TwoPointCorrelationFunction
from pycorr import project_to_multipoles
from pycorr import project_to_wp

#astropy
from astropy import constants as const
from astropy import units as u
import illustris_python as il
import h5py as h5

#abacus
import asdf
from abacusnbody.data.read_abacus import read_asdf
from abacusnbody.data.output_particle import rel_vel_healpix_lookup


def InputFile(*args, **kwargs):
    """Lazy ``abacus.param.InputFile``. Not imported at module level because
    ``abacus`` is a meson-python editable install whose import runs ``ninja``
    on the build dir: ``read_maplog_dir``'s forkserver workers re-import this
    module, so several workers would run ninja concurrently in the same build
    dir and race on ``.ninja_log`` -> ImportError -> BrokenProcessPool."""
    from abacus.param import InputFile as _InputFile
    return _InputFile(*args, **kwargs)


#Useful constants:
k_B = const.k_B.cgs
G = const.G.to(u.Mpc**3/u.M_sun*(u.km/u.s/u.Mpc)**2)
m_e = const.m_e
m_p = const.m_p
m_n = const.m_n
c = const.c.cgs
e = const.e.esu
hbar = const.hbar.to(u.GeV*u.s)
m_H = m_p+m_e - 13.6*u.eV/c**2 #unecessarily exact
mu=0.59

plt.rcParams.update({'font.size':16})
plt.rcParams['animation.ffmpeg_path'] = '/home/ajohnson/miniconda3/bin/ffmpeg'
plt.rcParams.update({'animation.embed_limit':200})
ABACUS_PERSIST = os.environ.get('ABACUS_PERSIST')
rng = np.random.default_rng()
  

def H(z, H0=67.74*u.km/u.s/u.Mpc, Omega_M=0.3089, Omega_DE=0.6911):
    return H0.value*np.sqrt((1+z)**3*Omega_M + Omega_DE)

def get_threshold(mass,BoxSize,guess=1e2,abundance=[5.01e-3,]):
    abundances = np.atleast_1d(abundance)
    Nmass = len(mass)
    mass    = np.copy(mass)
    mass    = np.sort(mass[mass>guess])
    threshs = np.empty_like(abundances)
    for i,n in enumerate(abundances):
        n = np.min([Nmass/BoxSize.value**3, n])
        N = int(n*BoxSize.value**3)
        threshs[i]        = mass[-N]
    return threshs


_NODE_TYPE_BITS = {'timeslice': 3, 'lightcone': 2}
_NODE_TYPE_NAMES = {bits: name for name, bits in _NODE_TYPE_BITS.items()}

_MAP_PARAM_KEYS = (
    'Map_bform_ips', 'Map_baccrete_ips', 'Map_bmerge_proper',
    'Map_Aform', 'Map_Aaccrete', 'Map_Amerge', 'Map_MinMultiplicity',
)

class Maplog:
    def __init__(self, tbl=None,params=None,state=None):
        self.table = tbl
        self.params = params
        self.state = state
        
    def __getitem__(self, key):
        return self.table[key]

    def __setitem__(self, key, value):
        self.table[key] = value

    def __delitem__(self, key):
        if key in self.table:
            del self.table[key]

    def hashable_pids(self):
        # each pid component is 16 bits (uint16 in maplog.cpp), so pack them at 0/16/32 bits: injective
        pids = maplog['pid'].astype(np.int64) & 0xFFFF
        return pids[:,0] + (pids[:,1] << 16) + (pids[:,2] << 32)
    
    def __index(self,ctrl):
        return table['control']==ctrl
    def __index(self,ts,nd):
        return table['control']==3*ts+nd
    
    def pos(self,ctrl):
        return table['pos'][__index(ctrl),:]
    def pos(self, ts, nd):
        return table['pos'][__index(ts,nd),:]
    
    def vel(self,ctrl):
        return table['vel'][__index(ctrl),:]
    def vel(self, ts, nd):
        return table['vel'][__index(ts,nd),:]

    def pid(self,ctrl):
        return table['pid'][__index(ctrl),:]
    def pid(self,ctrl):
        return table['pid'][__index(ts,nd),:]

    def mult(self,ctrl):
        return table['mult'][__index(ctrl),:]
    def mult(self,ctrl):
        return table['mult'][__index(ts,nd),:]

def dominant_formation_pid(node_type, pid, length=None, check=True):
    """Dominant-branch Formation pid for every row of a flat MapLog node stream.

    A MapLog tree is built by prepending, and a Merger node's secondary
    (lower-multiplicity) branch is appended *after* the primary's, so the
    dominant branch (the largest progenitor at every merger) is exactly the
    sequential run from any node forward to the *first* ``node_type==0``
    (``NODE_FORMATION``) encountered -- no stack needed (``maplog.cpp:96-108``).

    This is *not* the same node as ``head + length - 1``: that jump lands on
    the *final* Formation, i.e. the tail of the whole tree, which (whenever
    the MAP has ever been through a merger) is the recursively-deepest
    "always the secondary" progenitor buried at the end -- not the dominant
    lineage's own identity. The two coincide only for a MAP with no merger
    anywhere in its history (a single, unbranched Formation node).

    Because the dominant branch never branches, "nearest Formation at or
    after row i" is exactly what we want and needs no explicit tree walk --
    it's a single reverse running-min over the whole array.

    Parameters
    ----------
    node_type : (N,) array   0=Formation, 1=Merger, 2=LightCone, 3=TimeSlice
    pid       : (N,3) array  meaningful only on Formation rows (zero elsewhere)
    length    : (N,) array, optional -- if given, sanity-checks that every
                epoch row's dominant Formation is found within its own tree's
                declared ``length`` (a violation means a corrupt/truncated
                file, or that non-contiguous data got concatenated).
    check     : bool, whether to run that sanity check (raises if it fails)

    Returns
    -------
    dominant_pid       : (N,3) array, pid at the first Formation >= i. Only
                          meaningful where node_type[i] is an epoch node
                          (LightCone/TimeSlice); harmless garbage elsewhere.
    next_formation_idx : (N,) int64 array of indices used, for debugging.
    """
    node_type = np.asarray(node_type)
    N = node_type.shape[0]

    is_formation = (node_type == 0)  # NODE_FORMATION
    # Sentinel N is larger than any real index (valid indices are 0..N-1).
    idx_if_formation = np.where(is_formation, np.arange(N), N)
    # Reverse running-min: next_formation_idx[i] = min_{j>=i} idx_if_formation[j]
    next_formation_idx = np.minimum.accumulate(idx_if_formation[::-1])[::-1]

    unresolved = next_formation_idx >= N  # no Formation found before EOF
    safe_idx = np.where(unresolved, N - 1, next_formation_idx)

    if check and length is not None:
        length = np.asarray(length)
        is_epoch = (node_type == 2) | (node_type == 3)  # LightCone or TimeSlice
        idx = np.arange(N)
        within_tree = next_formation_idx <= idx + length - 1
        bad_epoch = is_epoch & (unresolved | ~within_tree)
        n_bad = int(bad_epoch.sum())
        if n_bad:
            bad_rows = np.flatnonzero(bad_epoch)[:10]
            raise AssertionError(
                f"{n_bad} epoch row(s) had no dominant-branch Formation within "
                f"their declared `length` (e.g. rows {bad_rows.tolist()}) -- "
                f"file may be truncated/corrupt, or a non-contiguous slice was passed.")

    dominant_pid = pid[safe_idx]
    return dominant_pid, next_formation_idx


def _read_one_maplog_file(f, load_keys, keys, node_bits, compute_pids, timesteps=None):
    """Read and mask one MapLog file; run in a worker process by
    ``read_maplog_dir``. Module-level (not a closure) so it can be pickled
    and sent to a ``ProcessPoolExecutor`` worker."""
    # Each worker process needs its own copy of this -- it isn't inherited
    # from the parent if the pool ever uses spawn/forkserver instead of fork.
    asdf.get_config().validate_on_read = False

    if not compute_pids:
        # Only the selected modality's rows are needed, so have read_asdf pick them before
        # unpacking rather than unpacking every node and masking afterwards.
        select = {'node_type': _NODE_TYPE_NAMES[node_bits]}
        if timesteps is not None:
            select['timestep'] = np.asarray(timesteps)
        tbl = read_asdf(f, load=keys, select=select)
        result = {key: np.asarray(tbl[key]) for key in keys}
        result['header'] = tbl.meta
        return result

    # The pid walk needs the whole node stream (the Formation nodes after each head), so
    # there is nothing to select up front here.
    tbl = read_asdf(f, load=load_keys)
    control = np.asarray(tbl['control'])
    mask = control % 4 == node_bits
    if timesteps is not None:
        mask &= np.isin(control >> 2, np.asarray(timesteps))
    result = {key: np.asarray(tbl[key])[mask] for key in keys}
    result['header'] = tbl.meta
    head = np.flatnonzero(mask)
    dominant_pid, _ = dominant_formation_pid(
        tbl['node_type'], np.asarray(tbl['pid']), np.asarray(tbl['length']))
    result['pid'] = dominant_pid[head]
    return result


def read_maplog_dir(mdir, node_type='timeslice', compute_pids=False,
                    keys = ['pos', 'vel', 'mult', 'control', 'timestep', 'node_type',
                            'density', 'vel_disp', 'rel_vel', 'rel_vel_healpix'],
                    nworkers=16, redshifts=None):
    """Read a directory's MapLog ASDF files into one dict of head-node
    arrays for one node modality.

    Uses the ``abacusnbody.data.read_asdf`` MapLog interface (``maplogs``
    colname; see ``doc/claude/reference_maplog.md``). Each rank writes its
    own ``maplog_z<z>_<rank_z>_<rank_x>.asdf`` file; the glob below picks up
    every such file under ``mdir`` regardless of the ``z`` baked into its
    name, and concatenates all of them into the returned dict. There's no
    I/O to save by selecting a subset of columns up front -- each file has to be
    decompressed in full to read even one column. The full ASDF header (from the first file) is attached
    unchanged as ``maplog['header']`` -- e.g. ``maplog['header']['BoxSize']``,
    ``maplog['header']['ParticleMassHMsun']`` -- and the per-step
    ``Step_to_Z`` redshift list is also read from there. The ``Map_*`` keys in
    ``_MAP_PARAM_KEYS`` are additionally copied to the top level of the dict
    as a convenience, since those are the ones compared across maplogs most
    often.

    Every file's Log tree holds Formation/Merger/Lightcone/TimeSlice nodes
    (``node_type`` 0/1/2/3) for each MAP's full history -- the whole run's
    accumulated log, written once at the final step (see
    ``doc/claude/reference_maplog.md`` sec. 6.2), so a single file already
    spans every ``P.MapLogRedshifts`` epoch, not just the final one.
    ``node_type`` selects which modality's head nodes to keep:
    ``'timeslice'`` (``node_type == 3``, i.e. ``control % 4 == 3``, the
    default) or ``'lightcone'`` (``node_type == 2``, i.e. ``control % 4 ==
    2``). Only those rows are kept: each file is read and filtered down to
    its selected-modality rows independently (see ``nworkers`` below), and
    the per-key results are concatenated across files at the end via
    ``np.concatenate``, rather than stashing every node type from every
    file in memory at once (as a full ``astropy.table.vstack`` over all
    files would) -- profiling showed that final concatenation step is only
    ~2-3% of total wall time regardless, so there's little to gain from
    changing it; the per-file ``read_asdf`` call is what dominates. Without
    ``compute_pids`` each file is read with ``read_asdf(select={'node_type': ...})``,
    so the other modalities' rows are never unpacked (the file is still decompressed
    in full). The
    returned dict keeps every step/redshift present for the chosen
    ``node_type``; a caller
    wanting one particular epoch filters ``maplog['step']``/
    ``maplog['redshift']``/``maplog['control']`` themselves (see
    ``map_step_for_redshift``).

    A selected head node doesn't carry a pid; ``pid`` is recovered by
    walking forward from the head to the *dominant-branch* Formation node
    (the first ``node_type==0`` encountered -- see ``dominant_formation_pid``
    and ``maplog.cpp:96-108``), which carries the pid of the largest
    progenitor at every merger. This is deliberately not the ``head +
    length - 1`` node: that jump lands on the *final* Formation (the tail of
    the whole tree), which is the recursively-deepest "always the secondary"
    progenitor whenever the MAP has ever been through a merger -- not a
    stable identity for the MAP. The two only coincide for an unbranched
    (never-merged) history. The walk is done per file (each rank's Log tree
    is written whole into that rank's own file, so a file's node indices are
    self-contained) while that file's full node stream (including the
    non-selected rows the walk needs) is still at hand -- before the
    selected-modality rows are concatenated across files.

    compute_pids : bool
        The Formation-node lookup above plus hashing the 3-part pid into an
        int64 (see below) is only needed by callers that key/match on pid
        (e.g. ``acj.match_maps_to_tng``); most callers here only ever use
        ``pos``/``vel``/``mult``. That work is significant compute (a
        gather indexed by ``length`` per file, plus reading the ``pid`` and
        ``length`` columns themselves), so it defaults to off -- the
        ``pid``/``length`` ASDF columns are never even read unless
        ``compute_pids`` is explicitly set True. When False (the default),
        ``maplog`` has no ``'pid'`` key.

    redshifts : iterable of float, optional
        If given, keep only the epoch (timestep) nearest each of these redshifts,
        instead of every epoch present. The unwanted rows are dropped before
        unpacking (``read_asdf`` ``select``), so memory scales with the rows kept.
        The nearest epoch is resolved from a one-file probe read of ``timestep``
        only. ``maplog['step']``/``['redshift']`` then list just the kept
        epochs, so ``map_step_for_redshift`` still resolves to the nearest one.
        Combine with a short ``keys`` (e.g. ``['pos', 'mult', 'control']``) to
        read only what ``get_map_xi`` needs.

    nworkers : int
        Each file's read-and-mask (and, if ``compute_pids``, the pid walk)
        is independent of every other file, so files are read concurrently
        with a ``ProcessPoolExecutor`` of this size (default 16). This is
        processes, not threads: profiling showed (a) the MapLog files'
        blosc2 compression isn't actually multithreaded internally here --
        ``blosc2.set_nthreads()`` measurably made no difference to per-file
        read time -- and (b) the per-file Python-side unpacking (building
        the astropy Table, the boolean-mask copy) holds the GIL long enough
        that a ``ThreadPoolExecutor`` plateaus around 8-16 workers regardless
        of file count, while a process pool keeps scaling (measured ~3.5x
        faster than the threaded plateau at 32 workers on a 104-core node).
        The tradeoff is that each worker's masked per-key arrays are
        pickled back to the main process instead of shared in-memory, and
        up to ``nworkers`` files' full (unfiltered) per-file tables can be
        resident (across worker processes) at once. Set to 1 to read
        serially in-process (e.g. for easier debugging/profiling). Tune
        this to the node you're actually running on -- there's no benefit
        to raising it past however many cores you have available, and
        raising it on a *shared* node (e.g. a login node) risks starving
        other users.
    """

    if node_type not in _NODE_TYPE_BITS:
        raise ValueError(
            "node_type must be one of {}, got {!r}".format(
                sorted(_NODE_TYPE_BITS), node_type))
    node_bits = _NODE_TYPE_BITS[node_type]

    # 'control' (the node-type mask in _read_one_maplog_file) and 'timestep'
    # (results[0]['timestep'] below, to derive maplog['step']/['redshift'])
    # are needed internally regardless of what the caller asked for in
    # `keys` -- add them to what's actually read off disk and passed through
    # _read_one_maplog_file's own `keys`, but keep `all_keys` (what gets
    # concatenated into the returned dict) as just the caller's `keys`, so a
    # caller who didn't ask for 'timestep' doesn't get a full per-particle
    # array back for it.
    worker_keys = list(keys)
    if 'timestep' not in worker_keys:
        worker_keys = worker_keys + ['timestep']
    load_keys = worker_keys + (['control'] if 'control' not in worker_keys else [])
    if compute_pids:
        load_keys = load_keys + ['pid', 'length']

    s = str(Path(mdir) / Path("maplogs/*/maplog_z*_*.asdf"))
    files = sorted(glob(s))
    if not files:
        raise FileNotFoundError("no maplog files matching {}".format(s))

    # TEMPORARY: asdf_output.cpp embeds the build's abbreviated git hash as an
    # unquoted YAML scalar (asdf_library.version); when that hash is all-digit
    # (no a-f), YAML types it as an int, which fails ASDF's core schema (wants
    # a string) on open. Skip validation so those files still read.
    asdf.get_config().validate_on_read = False

    all_keys = keys + ['pid'] if compute_pids else keys

    sel_steps = None
    if redshifts is not None:
        # Probe: which timesteps exist for this modality, and the step -> z table.
        probe = read_asdf(files[0], load=('timestep',),
                          select={'node_type': _NODE_TYPE_NAMES[node_bits]})
        present = np.unique(np.asarray(probe['timestep'])).astype(np.int64)
        z_present = np.asarray(probe.meta['Step_to_Z'])[present - 1]
        sel_steps = np.unique([present[np.argmin(np.abs(z_present - z))]
                               for z in redshifts])
        del probe

    worker = partial(_read_one_maplog_file, load_keys=load_keys, keys=worker_keys,
                      node_bits=node_bits, compute_pids=compute_pids,
                      timesteps=sel_steps)
    # forkserver, not the default fork: the redshift probe above decompresses
    # in this process, which starts blosc2's thread pool, and a fork()ed worker
    # inherits that pool's state without its threads -- its first blosc2
    # decompress then waits forever on a condvar (hung jobs 9040160, 9044870).
    with ProcessPoolExecutor(max_workers=nworkers,
                             mp_context=multiprocessing.get_context("forkserver")) as ex:
        # .map preserves input order regardless of completion order, so
        # results[0] is always files[0] -- matters below, since the header
        # and step -> redshift table are only ever taken from the first file.
        results = list(ex.map(worker, files))

    # step -> redshift: step_to_z[n] is the redshift at FullStepNumber==n. A
    # node's `timestep` is stamped with WriteState.FullStepNumber when the
    # node is created (both TimeSlice and LightCone nodes are packed through
    # the same `pack_base`, see doc/claude/reference_maplog.md sec. 1.3), one
    # step ahead of the ReadState redshift it actually reflects, hence the -1.
    header = results[0]['header']
    step_to_z = np.asarray(header['Step_to_Z'])
    timesteps = np.unique(results[0]['timestep']).astype(np.int64)

    maplog = {key: np.concatenate([r[key] for r in results], axis=0)
              for key in all_keys}
    maplog['step'] = timesteps.astype(np.int32)
    maplog['redshift'] = step_to_z[timesteps - 1].astype(np.float32)

    if compute_pids:
        # hash the 3-part pid into a single int64 so it can be used as a dict
        # key (e.g. acj.match_maps_to_tng keys dicts on maplog['pid'])
        # each pid component is 16 bits (uint16 in maplog.cpp), so pack them at 0/16/32 bits: injective.
        # (An earlier version used shifts of 2**31/2**32, which overlap and collide; any pickle cache
        # written with those hashes is invalid.)
        pid = maplog['pid'].astype(np.int64) & 0xFFFF
        maplog['pid'] = pid[:, 0] + (pid[:, 1] << 16) + (pid[:, 2] << 32)

    maplog['header'] = header
    for key in _MAP_PARAM_KEYS:
        if key in header:
            maplog[key] = header[key]

    return maplog


def read_maplog_dir_cached(mdir, cache_dir, node_type='timeslice', compute_pids=False,
                           keys=('pos', 'vel', 'mult', 'control', 'timestep', 'node_type',
                                 'density', 'vel_disp', 'rel_vel', 'rel_vel_healpix'),
                           nworkers=16, force=False, verbose=True):
    """Cached wrapper around ``read_maplog_dir``: skip directories already read.

    ``read_maplog_dir``'s cost is dominated by per-file ASDF decompression (see
    its docstring), which for AbacusAurora-scale maplogs can exceed a single
    batch job's walltime once a script reads more than a handful of directories
    (e.g. ``acj/plot_cosmo_variations.py`` reading one per cosmology variation).
    This wrapper pickles the *entire* dict ``read_maplog_dir`` returns to
    ``<cache_dir>/maplog_<mdir.name>_<node_type>[_pids].pkl`` the first time a
    directory is successfully read (pickle rather than ``.npz`` since the ASDF
    ``header`` is an arbitrary nested dict); every later call for that same
    directory -- in this process or a resubmission of the same job -- loads the
    cache instead of touching any ASDF files. A batch job killed by walltime
    therefore only has to (re-)read whichever directories aren't cached yet on
    its next run. The write is atomic (write to a ``.tmp`` path, then rename)
    so a job killed mid-write never leaves a corrupt cache file behind.

    The cache key does not encode ``keys``/``nworkers`` -- two calls for the
    same directory/``node_type``/``compute_pids`` with different ``keys`` will
    silently reuse whichever cache was written first (missing any key not
    requested by that first call). Keep ``keys`` consistent across calls
    sharing a ``cache_dir``, or pass ``force=True`` to force a fresh read.

    Parameters
    ----------
    mdir : path-like
        Simulation directory, as passed to ``read_maplog_dir``.
    cache_dir : path-like
        Directory the pickle cache is written to/read from; created if needed.
    force : bool
        If True, re-read from ASDF and overwrite any existing cache.
    verbose : bool
        If True, print one line per call saying whether it was a cache hit or
        a fresh (now-cached) read, so a batch job's log shows resume progress.
    node_type, compute_pids, keys, nworkers :
        Passed through to ``read_maplog_dir``.

    Returns
    -------
    dict
        Same shape as ``read_maplog_dir``.
    """
    cache_dir = Path(cache_dir)
    # '_pids16' (not '_pids'): the pid hash is now the injective 16-bit-per-component packing, so caches
    # written with the old colliding hash (named '_pids') are never matched
    suffix = '_pids16' if compute_pids else ''
    cache_path = cache_dir / 'maplog_{}_{}{}.pkl'.format(Path(mdir).name, node_type, suffix)

    if not force and cache_path.is_file():
        if verbose:
            print(f'  [cache hit] {cache_path.name}')
        with open(cache_path, 'rb') as fh:
            return pickle.load(fh)

    if verbose:
        print(f'  [reading]   {mdir} -> {cache_path.name}')
    maplog = read_maplog_dir(mdir, node_type=node_type, compute_pids=compute_pids,
                             keys=list(keys), nworkers=nworkers)

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(cache_path.name + '.tmp')
    with open(tmp_path, 'wb') as fh:
        pickle.dump(maplog, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.rename(cache_path)

    return maplog


def _count_heads_one_file(f, target_step):
    """Number of timeslice head nodes at ``target_step`` in one MapLog file (worker for
    ``read_maplog_histories``)."""
    asdf.get_config().validate_on_read = False
    tbl = read_asdf(f, load=['mult'], select={'node_type': 'timeslice', 'timestep': int(target_step)})
    return len(tbl)


def _read_histories_one_file(f, sel_local, target_step, n_back):
    """Dominant-branch histories of the target-epoch head nodes of one MapLog file (worker for
    ``read_maplog_histories``). ``sel_local`` indexes into the file's target-epoch heads (None = all)."""
    asdf.get_config().validate_on_read = False
    tbl = read_asdf(f, load=['pos', 'vel', 'mult', 'control', 'timestep', 'node_type'])
    ctrl = np.asarray(tbl['control'])
    N = len(ctrl)
    heads = np.flatnonzero(ctrl == 4 * int(target_step) + 3)
    if sel_local is not None:
        heads = heads[sel_local]
    nt, ts = np.asarray(tbl['node_type']), np.asarray(tbl['timestep'])
    pos, vel, mult = np.asarray(tbl['pos']), np.asarray(tbl['vel']), np.asarray(tbl['mult'])
    step_to_z = np.asarray(tbl.meta['Step_to_Z'], dtype=np.float32)

    H, K = len(heads), n_back + 1
    out = dict(pos=np.full((H, K, 3), np.nan, np.float32), vel=np.full((H, K, 3), np.nan, np.float32),
               mult=np.zeros((H, K), np.uint32), z=np.full((H, K), np.nan, np.float32),
               kind=np.full((H, K), -1, np.int8))
    # The dominant branch is the contiguous run from the head: epoch (timeslice) nodes of the single merged
    # MAP, ending at the first Merger node (the last merger) or Formation node. Record that terminating node
    # too (slot after the last epoch), then stop.
    alive = np.ones(H, bool)
    for k in range(K):
        idx = np.minimum(heads + k, N - 1)
        take = alive & (heads + k < N)
        i = idx[take]
        out['pos'][take, k], out['vel'][take, k], out['mult'][take, k] = pos[i], vel[i], mult[i]
        out['z'][take, k] = step_to_z[ts[i] - 1]
        out['kind'][take, k] = nt[i]
        alive = take & (nt[idx] == 3)
    out['truncated'] = alive        # still on epoch nodes after n_back slots: the run was cut off
    out['header'] = tbl.meta
    return out


def read_maplog_histories(mdir, target_step, select=None, n_back=16, nworkers=12, files=None):
    """Dominant-branch histories, back to the last merger, of the MAPs at one timeslice.

    For each timeslice head node at ``target_step`` (``control == 4*target_step + 3``), follow the
    dominant branch forward through the log -- the contiguous run of earlier TimeSlice nodes of the same
    merged MAP (``maplog.cpp:96-108``) -- until the first Merger node (the most recent merger) or Formation
    node, which is recorded as the final slot. Epochs before that merger belong to a progenitor whose
    identity can swap between runs, so they are not followed. At most ``n_back`` slots after the head are
    kept.

    Heads come back in the same order as ``read_maplog_dir`` returns the rows with
    ``control == 4*target_step + 3`` (file order, then row order), so index ``i`` here is index ``i`` of
    that selection. ``select`` (boolean mask or index array into that ordering) restricts which heads are
    returned; the per-file head counts it needs are obtained with a first, count-only pass.

    Returns a dict of arrays with shape (H, n_back+1[, 3]): ``pos``, ``vel`` (NaN where absent), ``mult``
    (0 where absent), ``z`` (redshift of the slot's node, NaN where absent), ``kind`` (node_type of the
    slot: 3 epoch, 1 Merger, 0 Formation, -1 absent) and ``epoch`` (index into
    ``header['MapLogRedshifts']`` nearest the slot's redshift; -1 unless ``kind == 3``). Slot 0 is the head.
    Also ``truncated`` (H,) -- True if the run was still on epoch nodes after ``n_back`` slots --
    ``header`` and ``select_idx`` (global head indices returned).
    """
    if files is None:
        files = sorted(glob(str(Path(mdir) / Path("maplogs/*/maplog_z*_*.asdf"))))
    if not files:
        raise FileNotFoundError("no maplog files under {}".format(mdir))
    asdf.get_config().validate_on_read = False

    with ProcessPoolExecutor(max_workers=nworkers) as ex:
        if select is None:
            sels, sel_idx = [None] * len(files), None
        else:
            counts = np.array(list(ex.map(partial(_count_heads_one_file, target_step=target_step), files)))
            off = np.r_[0, np.cumsum(counts)]
            select = np.asarray(select)
            sel_idx = np.flatnonzero(select) if select.dtype == bool else np.sort(select)
            if select.dtype == bool and len(select) != off[-1]:
                raise ValueError("select has length {} but there are {} heads at step {}".format(
                    len(select), off[-1], target_step))
            sels = [sel_idx[(sel_idx >= off[i]) & (sel_idx < off[i + 1])] - off[i] for i in range(len(files))]
        results = list(ex.map(partial(_read_histories_one_file, target_step=target_step, n_back=n_back),
                              files, sels))

    hist = {key: np.concatenate([r[key] for r in results], axis=0)
            for key in ('pos', 'vel', 'mult', 'z', 'kind', 'truncated')}
    hist['header'] = results[0]['header']
    hist['select_idx'] = sel_idx

    # epoch index: nearest MapLogRedshifts entry, for epoch (TimeSlice) slots only
    zlist = np.asarray(hist['header']['MapLogRedshifts'], dtype=np.float32)
    order = np.argsort(zlist)
    zs = zlist[order]
    zq = np.nan_to_num(hist['z'], nan=0.)
    j = np.clip(np.searchsorted(zs, zq), 1, len(zs) - 1)
    nearest = np.where(np.abs(zq - zs[j - 1]) <= np.abs(zq - zs[j]), j - 1, j)
    hist['epoch'] = np.where(hist['kind'] == 3, order[nearest], -1).astype(np.int16)
    return hist


def read_subsample_dir(sdir, z_desired=0.2, subsample_type='P8', keys=None):
    """Read a directory's fair particle subsample nearest ``z_desired``.

    Globs ``subsamples/z*/*/particles_sub<subsample_type>*.asdf`` (one file per
    MPI rank), picks the redshift closest to ``z_desired``, and concatenates all
    ranks' particles into one dict alongside ``BoxSize``/``Mp``.

    keys : sequence of str, optional
        Particle fields to load (passed through to ``read_asdf``; valid values
        are ``'pos','vel','pid','density','vel_disp','is_map','mult','rel_vel',
        'rel_vel_healpix'``). Defaults to all of them. Subsample directories can
        hold tens of millions of particles, so callers that only need e.g.
        ``['pos','is_map','mult']`` (as ``get_map_matter_xi`` does) should pass
        that subset to avoid the cost of reading/concatenating fields they don't
        use.

        ``rel_vel`` is the magnitude of a MAP's velocity relative to the mean
        local DM; ``rel_vel_healpix`` is the matching direction as an Nside=4 NEST
        healpix pixel index, which you turn into a unit vector via
        ``abacusnbody.data.output_particle.rel_vel_healpix_lookup``.
    """
    if keys is None:
        keys = ['pos', 'vel', 'pid', 'density', 'vel_disp', 'is_map', 'mult',
                'rel_vel', 'rel_vel_healpix']
    subsamples = []
    subsample = {}

    #get boxsize and particle mass
    params = InputFile(str(Path(sdir)/Path('abacus.par')))
    state_files = sorted(glob(str(Path(sdir) / 'log' / 'last' / 'step*.state')))
    if not state_files:
        raise FileNotFoundError(
            "no state file matching {}".format(Path(sdir) / 'log' / 'last' / 'step*.state'))
    Mp = InputFile(state_files[0])['ParticleMassHMsun']
    subsample['BoxSize'] = params['BoxSize']*u.Mpc
    subsample['Mp'] = Mp*u.Msun

    redshifts = []
    for f in glob(str(Path(sdir)/Path("subsamples/z*"))):
        res = re.search(r'z(\d+.*\d+)', str(Path(f).name))
        if res:
            redshifts.append(float(res.group(1)))
    z = redshifts[np.argmin(np.abs(z_desired-np.array(redshifts)))]
    if z != z_desired:
        print(f"redshift: z={z_desired:1.3f} is not available, using z={z:1.3f} (closest available)")
    
    #actually read and contruct the main subsamples content
    s = str(Path(sdir)/Path(f"subsamples/z{z:1.3f}/*/particles_sub{subsample_type:s}*.asdf"))
    for f in glob(s):
        tbl = read_asdf(f, load=keys) 
        subsamples.append(tbl)

    for key in keys:
        data = np.array(subsamples[0][key])
        for s in subsamples[1:]:
            data = np.concatenate([data, s[key]], axis=0)
        if key == 'pid':
            # same 16-bit-per-component packing as read_maplog_dir, so subsample and MAP pids hash alike
            data   = data.astype(np.int64) & 0xFFFF
            data   = data[:,0] + (data[:,1] << 16) + (data[:,2] << 32)
        
        subsample[key] = data
    
    return subsample

def rel_vel_healpix_to_vector(healpix_index, rel_vel):
    """Reconstruct the MAP relative-velocity vector from its ``rel_vel_healpix``
    direction and ``rel_vel`` magnitude.

    ``healpix_index`` (Nside=4 NEST pixel index, in [0, 192)) and ``rel_vel``
    may each be a scalar or a list/array of matching length; the lookup is a
    simple fancy-index into ``rel_vel_healpix_lookup`` scaled by ``rel_vel``,
    so it vectorizes automatically.

    The vector points from the MAP to the mean dark-matter velocity around it
    (``v_DM - v_MAP``, as Abacus computes it in ``set_env_variables``), so the
    dark-matter velocity estimate is ``vel_map + rel_vel_vector``.

    Returns
    -------
    np.ndarray
        Shape (3,) for scalar input, else (N, 3).
    """
    return rel_vel_healpix_lookup[np.asarray(healpix_index)] * np.asarray(rel_vel)[..., None]

def load_tng_subhalos(basePath, snapno, fields=['SubhaloMassType','SubhaloFlag','SubhaloPos','SubhaloVel']):
    if 'SubhaloFlag' not in fields:
        fields.append('SubhaloFlag')
    tng_subhalos  = il.groupcat.loadSubhalos(basePath, snapno, fields=fields)
    flag = tng_subhalos['SubhaloFlag']
    for k in tng_subhalos.keys():
        if (k!='SubhaloFlag') and (k!='count'):
            if k=='SubhaloPos':
                tng_subhalos[k] = tng_subhalos[k][flag]/1e3
            else:
                tng_subhalos[k] = tng_subhalos[k][flag]
            
    return tng_subhalos

def get_xi(pos, edges, BoxSize, pos2=[], mode='s', max_p=1e7,nthreads=26, args={'los':'z'},
           weights=None, weights2=None):
    """Pair counts / correlation function of ``pos`` (auto) or ``pos`` x ``pos2`` (cross) as a pycorr object. Optional
    ``weights`` / ``weights2``: one weight per point of ``pos`` / ``pos2`` (kept aligned through the ``max_p`` subsampling)."""
    #pos1
    w1 = None if weights is None else np.asarray(weights, dtype=np.float64)
    if w1 is not None:
        assert len(w1) == len(pos), "weights must have one entry per position"
    if len(pos)>max_p:
        idcs  = rng.choice(len(pos), size=int(max_p),replace=False)
        pos_c = np.copy(pos[idcs,:])
        if w1 is not None: w1 = w1[idcs]
    else:
        pos_c = np.copy(pos)
    pos_c = (pos_c + BoxSize.value ) % BoxSize.value
    
    #pos2
    w2 = None if weights2 is None else np.asarray(weights2, dtype=np.float64)
    if len(pos2)>0:
        if w2 is not None:
            assert len(w2) == len(pos2), "weights2 must have one entry per position in pos2"
        if len(pos2)>max_p:
            idcs  = rng.choice(len(pos2), size=int(max_p),replace=False)
            pos2_c = np.copy(pos2[idcs,:])
            if w2 is not None: w2 = w2[idcs]
        else:
            pos2_c = np.copy(pos2)

        pos2_c = (pos2_c + BoxSize.value) % BoxSize.value
    else:
        pos2_c = None
        w2 = None

    #correlation function
    xi = TwoPointCorrelationFunction(mode, edges, data_positions1=pos_c, data_positions2=pos2_c, position_type='pos',\
                                 data_weights1=w1, data_weights2=w2,\
                                 bin_type='custom',estimator='natural',los=args['los'],\
                                 boxsize=BoxSize.value, compute_sepsavg=False, engine='corrfunc',\
                                 nthreads=nthreads)
    return xi


def xi_poisson(twopt, xi, weighted=False):
    """Pair-count shot-noise error on ``xi``. Unweighted: sqrt(DD)/RR. Weighted: (1+xi)/sqrt(N_pairs) from the raw pair
    counts (identical to the unweighted value when all weights are 1; it ignores the extra variance from unequal
    weights, so it underestimates the error for strongly varying weights)."""
    if weighted:
        return (1 + xi) / np.sqrt(twopt.D1D2.ncounts)
    return np.sqrt(twopt.D1D2.wcounts) / twopt.R1R2.wcounts

def test_maplog(logs):
    # Given a table from a log file, do some diagnostic tests.
    print(f"Number of total nodes:    {len(logs):9d}")
    print(f"Number of formation nodes:{np.count_nonzero(logs['node_type']==0):9d}")
    print(f"Number of merger nodes:   {np.count_nonzero(logs['node_type']==1):9d}")
    print(f"Number of lightcone nodes:{np.count_nonzero(logs['node_type']==2):9d}")
    print(f"Number of timeslice nodes:{np.count_nonzero(logs['node_type']==3):9d}")

    print(f"Final timestep is {np.max(logs['timestep']):d}")
    orphans = np.sum(logs['control']==4*np.max(logs['timestep']))
    orphans_lc = np.sum(logs['control']==4*np.max(logs['timestep'])+2)
    orphans_merger = np.sum(logs['control']==4*np.max(logs['timestep'])+1)
    print(f"There are {orphans:d} MAPs formed after the final timeslice")
    print(f"There are {orphans_lc:d} MAPs on LC after the final timeslice")
    print(f"There are {orphans_merger:d} MAPs merged after the final timeslice")
    # Get the indices of the end-time headnodes
    e = np.array(range(len(logs)))[logs['control']==np.max(logs['control'])]
    # Then this index + length-1 should always be a formation node
    last = e + logs['length'][e]-1
    print(f"Number of final MAPs:     {len(e):9d}")

    results = np.unique(logs['node_type'][last], return_counts=True)
    if ((results[0][0]==0)&(results[1][0]==len(e))):
        print("True: All final-time headnodes have lengths that end in a formation node.")
    else:
        print("ERROR: Some final-time headnodes do not have lengths that end in a formation node!")
        print(results)

    # The sum of the final-time lengths should be the entire file
    print(f"Sum of all final time lengths is {np.sum(logs['length'][e].astype(np.int64)):9d}, should be equal to {len(logs)-orphans-orphans_merger:9d}")
    # The log must have the property that the next timestep is <= the current one, unless the current one has node_type==0.

    # The next timestep never greater, unless the current step is a formation node
    print("Testing that all non-formation sequences descend in timestep:", np.all((logs['node_type'][:-1]==0)|(np.diff(logs['timestep'].astype(int))<=0)))

    # Is e in order?
    #print(np.all(np.diff(e) > 0))
    #plt.plot(e[1:], np.cumsum(logs['length'][e])[:-1]-e[1:])
    #plt.show()


def get_colors(x,y,cmap='viridis',lo=0.0,s=1.):
    xyz = np.array([x,y])
    p_est = gaussian_kde(xyz)
    c = p_est(xyz)
    return colormaps[cmap](lo+c/max(c)*s)


def plot_window(maplog, BoxSize, basePath, snapno, str_identifier="",
                 xlim=(-10., 10.), ylim=(-10., 10.), zc=0.0, zwidth=1.0,
                 figsize=(11.5, 11), abundance=5.01e-3, save=False):
    """Plot MAPs and (valid) TNG galaxies in an (x, y) window and a thin z-slice.

    Parameters
    ----------
    map_pos : (N, 3) array
        MAP positions (e.g. ``maplog['pos']`` from analysis.ipynb).
    BoxSize : astropy Quantity
        Simulation box size; used to wrap positions.
    basePath, snapno :
        TNG300 group-catalog base path and snapshot number passed to the
        illustris_python utilities.

    The (x, y) window defaults to +/-10 Mpc/h around the origin and the z-slice
    to 1 Mpc/h centered on ``zc``. The TNG halos whose centers fall in the window
    are located with the illustris_python utilities and *all* of their particles
    (that land in the window) are scattered, colored by their local surface
    density from a 2D kde (see ``get_colors``) using the plasma colormap on a
    black background.

    Positions are wrapped into a box-centered frame ([-BoxSize/2, BoxSize/2]) so
    that "the origin" is well defined and the window never straddles a periodic
    boundary.
    """
    L      = BoxSize.value
    dmtype = il.snapshot.partTypeNum('dm')
    star_type = il.snapshot.partTypeNum('star')
    zlim   = [zc - zwidth/2., zc + zwidth/2.]

    # wrap positions into a box-centered frame so the origin sits at the center
    def wrap(pos):
        return ((np.asarray(pos, dtype=float) + L/2.) % L) - L/2.
    
    # --- MAPs ---
    timestep = np.unique(maplog['timestep'])[-1]
    # control = 4*timestep + node_type; node_type 3 -> timeslice nodes
    timeslice = maplog['control'] == timestep * 4 + 3
    thresh = get_threshold(maplog['mult'][timeslice], BoxSize, guess=1e1,abundance=abundance)
    msk = timeslice & (maplog['mult'] >= thresh)
    map_pos = maplog['pos'][msk, :]
    
    # --- TNG halos in the window: gather all of their dm particles ---
    halos = il.groupcat.loadHalos(basePath, snapno, fields=['GroupPos'])
    hpos  = wrap(halos/1e3)
    hmsk  = get_box_msk(hpos, ylim, zlim, xlim[0], xlim[1])

    halo_p_pos = []
    for hid in np.nonzero(hmsk)[0]:
        p = il.snapshot.loadHalo(basePath, snapno, hid, dmtype, fields=['Coordinates'])/1e3
        halo_p_pos.append(wrap(p))
    if len(halo_p_pos) > 0:
        halo_p_pos = np.concatenate(halo_p_pos, axis=0)
        halo_p_pos = halo_p_pos[get_box_msk(halo_p_pos, ylim, zlim, xlim[0], xlim[1])]
    else:
        halo_p_pos = np.empty((0, 3))

    # --- valid TNG galaxies (subhalos) in the window ---
    subhalos = il.groupcat.loadSubhalos(basePath, snapno, fields=['SubhaloPos', 'SubhaloFlag','SubhaloMassType'])
    subpos   = wrap(subhalos['SubhaloPos']/1e3)
    thresh = get_threshold(subhalos['SubhaloMassType'][:,star_type], BoxSize, guess=1e-3, abundance=abundance)
    submsk   = (subhalos['SubhaloFlag'] == 1) & get_box_msk(subpos, ylim, zlim, xlim[0], xlim[1]) & (subhalos['SubhaloMassType'][:,star_type]>thresh)
    subpos   = subpos[submsk]

    # --- MAPs in the window ---
    map_pos = wrap(map_pos)
    map_pos = map_pos[get_box_msk(map_pos, ylim, zlim, xlim[0], xlim[1]), :]

    # --- plot ---
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.set_facecolor('k')
    if len(halo_p_pos) > 0:
        c = get_colors(halo_p_pos[:, 0], halo_p_pos[:, 1], cmap='plasma', lo=0.1, s=0.9)
        ax.scatter(halo_p_pos[:, 0], halo_p_pos[:, 1], c=c, s=0.5)
    if len(subpos) > 0:
        ax.scatter(subpos[:, 0], subpos[:, 1], marker='*', c='red', s=70, label='TNG galaxies')
    if len(map_pos) > 0:
        ax.scatter(map_pos[:, 0], map_pos[:, 1], marker='o', c='blue', s=40, label='MAPs')
    ax.set(xlim=xlim, ylim=ylim, xlabel='x [Mpc/h]', ylabel='y [Mpc/h]')
    ax.legend()
    plt.show()
    if save:
        fig.savefig('Images/snapshot_x[{:.1f},{:.1f}]_y[{:.1f},{:.1f}]_z{:.1f}+-{:.1f}_'.format(
            xlim[0], xlim[1], ylim[0], ylim[1], zc, zwidth) + str_identifier + '.png',
            bbox_inches='tight')

def get_sci_not(val):
    exp = int(np.floor(np.log10(val)))
    man = val/10**exp
    return man, exp


def get_available_tng_snapshots(basePath):
    """Discover the TNG snapshots whose group catalogs are present under basePath.

    Not every TNG snapshot is downloaded locally; only those with a
    ``groups_<NNN>`` directory in ``basePath`` can be loaded. Returns
    ``(snapnos, redshifts)`` sorted by snapshot number, with redshifts read from
    each group-catalog header.
    """
    snapnos = []
    for d in glob(str(Path(basePath) / 'groups_*')):
        name = Path(d).name
        try:
            snapnos.append(int(name.split('_')[-1]))
        except ValueError:
            continue
    snapnos   = np.array(sorted(snapnos), dtype=int)
    redshifts = np.array([il.groupcat.loadHeader(basePath, int(sn))['Redshift']
                          for sn in snapnos])
    return snapnos, redshifts


def plot_correlation_functions(maplog, redshifts,
                               basePath='/flare/Abacus/ajohnson/TNG300/TNG300-2/output/',
                               rmin=2e-2, rmax=40.0, nbins=30, type_str='gg',
                               abundance=5.01e-3, max_p=np.inf, nthreads=26,
                               map_cmap='Blues_r', tng_cmap='Reds_r', clo=0., chi=0.75,
                               figsize=(8, 6), save=False, str_identifier='', ztol=5e-3):
    """Plot MAP vs TNG-galaxy correlation functions, one figure per redshift.

    Mirrors the "make correlation function plots" section of analysis.ipynb:
    for each requested redshift the matching MAP timeslice nodes are
    abundance-thresholded on multiplicity and fed to ``get_xi``; the matching
    TNG snapshot's subhalos are abundance-thresholded on stellar mass and fed to
    ``get_xi``. Curves are drawn as r^2 xi(r) with a Poisson band.

    Each requested redshift is resolved to the proper MAP step and TNG snapshot:
    the MAP step is the timeslice in ``maplog['step']``/``maplog['redshift']``
    whose redshift is closest to the request; the TNG snapshot is the closest
    available one (only a few TNG snapshots are downloaded locally, see
    ``get_available_tng_snapshots``) and must agree with the request to within
    two decimal places (``ztol``).

    Each requested redshift gets its own figure with the MAP curve (solid,
    ``map_cmap``/Blues) and the TNG curve (dashed, ``tng_cmap``/Reds). Redshift
    is encoded by color via a single shared normalization across all requested
    redshifts, so colors stay consistent between figures; the colormaps are
    sampled on [clo, chi] to skip their near-white low end. The legend on each
    figure labels the MAP and TNG redshifts individually.

    Parameters
    ----------
    maplog : dict
        Output of ``read_maplog_dir``; provides 'control', 'mult', 'pos',
        'header' (with 'BoxSize'), and the parallel 'step'/'redshift' arrays.
    redshifts : float or sequence of float
        Target redshift(s) to plot. Each is mapped to the nearest MAP step and
        the nearest available TNG snapshot.
    basePath : str
        TNG group-catalog base path (default TNG300-2).
    rmin, rmax, nbins : float, float, int
        Log-spaced separation binning (Mpc/h) for the pair counts.
    type_str : str
        Correlation-function subscript in the y-axis label (e.g. 'gg').
    abundance : float
        Target number density (Mpc/h)^-3 used by ``get_threshold`` for both
        MAP multiplicity and TNG stellar mass.
    max_p, nthreads :
        Passed through to ``get_xi``.
    map_cmap, tng_cmap : str
        Colormaps encoding redshift for MAPs and TNG respectively.
    clo, chi : float
        Lower/upper fraction of each colormap to sample (avoids near-white ends).
    figsize : tuple
        Size of each figure.
    save : bool
        If True, save each figure under Images/ with the MAP redshift appended.
    str_identifier : str
        Filename suffix used when ``save`` is True.
    ztol : float
        Maximum allowed redshift difference between a requested redshift and the
        closest available TNG snapshot (default 5e-3, i.e. agreement to two
        decimal places). A request with no TNG snapshot within ``ztol`` raises.

    Returns
    -------
    (figs, axes) : lists, one entry per requested redshift.
    """
    redshifts = np.atleast_1d(redshifts).astype(float)

    BoxSize   = maplog['header']['BoxSize'] * u.Mpc
    star_type = il.snapshot.partTypeNum('star')
    r_edges   = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
    r         = r_edges[:-1] + np.diff(r_edges) / 2.0

    # resolve each requested redshift to a MAP step and a TNG snapshot
    tng_snapnos, tng_zs = get_available_tng_snapshots(basePath)
    timesteps = np.empty(len(redshifts), dtype=int)
    snapnos   = np.empty(len(redshifts), dtype=int)
    map_z     = np.empty(len(redshifts))
    tng_z     = np.empty(len(redshifts))
    for i, z_req in enumerate(redshifts):
        # nearest MAP timeslice
        j            = np.argmin(np.abs(maplog['redshift'] - z_req))
        timesteps[i] = int(maplog['step'][j])
        map_z[i]     = maplog['redshift'][j]
        # nearest available TNG snapshot, required to match to 2 decimal places
        k = np.argmin(np.abs(tng_zs - z_req))
        if abs(tng_zs[k] - z_req) > ztol:
            raise ValueError(
                "requested redshift {:.3f} has no TNG snapshot within {:g} "
                "(2 decimal places); available TNG redshifts are {}".format(
                    z_req, ztol, ', '.join('{:.2f}'.format(z) for z in tng_zs)))
        snapnos[i] = int(tng_snapnos[k])
        tng_z[i]   = tng_zs[k]

    # shared normalization so equal redshift -> equal position on each colormap
    # (keeps colors consistent across the separate figures); sample on [clo, chi]
    # to skip the near-white low end of Blues/Reds.
    zmin, zmax = min(map_z.min(), tng_z.min()), max(map_z.max(), tng_z.max())
    norm = Normalize(vmin=zmin, vmax=zmax)
    map_colormap, tng_colormap = colormaps[map_cmap], colormaps[tng_cmap]
    def shade(cmap, z):
        return cmap(clo + (chi - clo) * (norm(z) if zmax > zmin else 0.5))

    figs, axes = [], []
    for ts, sn, zm, zt in zip(timesteps, snapnos, map_z, tng_z):
        fig, ax = plt.subplots(1, figsize=figsize)

        # --- MAPs: timeslice nodes (control == 4*timestep + 3) ---
        timeslice  = maplog['control'] == int(ts) * 4 + 3
        map_thresh = get_threshold(maplog['mult'][timeslice], BoxSize, guess=1e1, abundance=abundance)
        msk        = timeslice & (maplog['mult'] >= map_thresh)
        twopt      = get_xi(maplog['pos'][msk, :], r_edges, BoxSize, max_p=max_p, nthreads=nthreads)
        xi         = twopt.get_corr(return_sep=False)
        poisson    = np.sqrt(twopt.D1D2.wcounts) / twopt.R1R2.wcounts
        c = 'mediumblue' #shade(map_colormap, zm)
        ax.plot(r, r**2 * xi, '-', lw=2, color=c, label=f'MAP z={zm:.2f}')
        ax.fill_between(r, r**2 * (xi - poisson), r**2 * (xi + poisson), color=c, alpha=0.3)

        # --- TNG galaxies: stellar-mass-thresholded subhalos ---
        tng_subhalos = load_tng_subhalos(basePath, int(sn))
        m_tng        = tng_subhalos['SubhaloMassType'][:, star_type]  # units of 1e10 Msun
        # native TNG box size (header is ckpc/h, SubhaloPos is scaled to Mpc/h)
        tng_boxsize  = il.groupcat.loadHeader(basePath, int(sn))['BoxSize'] / 1e3 * u.Mpc
        tng_thresh   = get_threshold(m_tng, tng_boxsize, guess=1e-3, abundance=abundance)
        msk_tng      = m_tng >= tng_thresh
        twopt        = get_xi(tng_subhalos['SubhaloPos'][msk_tng, :], r_edges, tng_boxsize, max_p=max_p, nthreads=nthreads)
        xi_tng       = twopt.get_corr(return_sep=False)
        poisson_tng  = np.sqrt(twopt.D1D2.wcounts) / twopt.R1R2.wcounts
        c = 'red' #shade(tng_colormap, zt)
        ax.plot(r, r**2 * xi_tng, '--', lw=2, color=c, label=f'TNG z={zt:.2f}')
        ax.fill_between(r, r**2 * (xi_tng - poisson_tng), r**2 * (xi_tng + poisson_tng), color=c, alpha=0.3)

        ax.grid()
        ax.set(xscale='log', xlabel=r'r [Mpc/h]', ylabel=r'$r^2\xi_{%s}(r)$' % type_str)
        ax.legend()
        if save:
            fig.savefig(f'Images/correlation_functions_z{zm:.2f}_n{abundance:.2e}_' + str_identifier + '.png',
                        bbox_inches='tight')
        figs.append(fig)
        axes.append(ax)

    plt.show()
    return figs, axes


def map_step_for_redshift(maplog, z):
    """Resolve a target redshift to the nearest MAP timeslice step.

    Uses the parallel ``maplog['step']``/``maplog['redshift']`` arrays. Returns
    ``(step, redshift)`` for the timeslice whose redshift is closest to ``z``.
    """
    j = np.argmin(np.abs(maplog['redshift'] - z))
    return int(maplog['step'][j]), float(maplog['redshift'][j])


def get_map_xi(maplog, timestep, r_edges, abundance=5.01e-3, mult_thresh=None,
               max_p=np.inf, nthreads=26, weighted=False):
    """MAP correlation function for one maplog at one timeslice timestep.

    Selects the timeslice nodes (control == 4*timestep + 3), abundance-thresholds
    them on multiplicity, and feeds the surviving positions to ``get_xi``.
    Returns ``(xi, poisson)`` on the bin centers implied by ``r_edges``.

    If ``mult_thresh`` is given, it is used directly as the multiplicity cut
    instead of recomputing one from ``abundance`` at this timestep -- lets a
    caller freeze the selection threshold across timesteps (e.g. matching it
    once at a reference redshift) rather than re-deriving an abundance-matched
    cut, which would otherwise drift as the timestep's mult distribution evolves.

    ``weighted``: if True, weight each MAP by its multiplicity (``maplog['mult']``). The abundance threshold is unchanged
    (it counts MAPs, not weight); the correlation function is the weighted one and ``poisson`` is
    ``xi_poisson(..., weighted=True)``.
    """
    BoxSize   = maplog['header']['BoxSize'] * u.Mpc
    timeslice = maplog['control'] == int(timestep) * 4 + 3
    thresh    = mult_thresh if mult_thresh is not None else \
                get_threshold(maplog['mult'][timeslice], BoxSize, guess=1e1, abundance=abundance)
    msk       = timeslice & (maplog['mult'] >= thresh)
    w         = maplog['mult'][msk] if weighted else None
    twopt     = get_xi(maplog['pos'][msk, :], r_edges, BoxSize, max_p=max_p, nthreads=nthreads, weights=w)
    xi        = twopt.get_corr(return_sep=False)
    poisson   = xi_poisson(twopt, xi, weighted=weighted)
    return xi, poisson


def get_map_xi_cached(maplog, cache_key, timestep, r_edges, cache_dir, abundance=5.01e-3,
                      mult_thresh=None, max_p=np.inf, nthreads=26, force=False, verbose=True):
    """Cached wrapper around ``get_map_xi``: skip (directory, timestep) pairs
    whose correlation function was already computed.

    Meant to checkpoint the pair-counting step alongside ``read_maplog_dir_cached``
    checkpointing the read step, so a batch script that computes several maplogs'
    correlation functions right after reading each one (e.g.
    ``acj/plot_cosmo_variations.py``) doesn't lose that work either if the job is
    killed before it finishes every directory. Caches to ``<cache_dir>/xi_<cache_key>
    _t<timestep>_n<abundance>_m<mult_thresh>_p<max_p>_r<hash of r_edges>.npz``
    (write is atomic: written to a ``.tmp`` path, then renamed). ``max_p`` is
    part of the key (not just ``abundance``/``mult_thresh``/``r_edges``)
    because it changes the pair counts whenever the MAP catalog is larger than
    it -- callers must pass the same ``max_p`` here as they'll use for the
    final (uncached) computation, or this cache is silently never reused.

    Parameters
    ----------
    cache_key : str
        Identifies the source simulation directory (e.g. ``Path(mdir).name``).
        Callers are responsible for choosing a key that's stable across runs and
        unique per maplog -- together with ``timestep``/``abundance``/
        ``mult_thresh``/``r_edges`` it's the only thing distinguishing one cache
        entry from another.
    cache_dir : path-like
        Directory the ``.npz`` cache is written to/read from; created if needed.
    force : bool
        If True, recompute and overwrite any existing cache entry.
    verbose : bool
        If True, print one line per call saying whether it was a cache hit or a
        fresh (now-cached) computation.
    maplog, timestep, r_edges, abundance, mult_thresh, max_p, nthreads :
        Passed through to ``get_map_xi``.

    Returns
    -------
    (xi, poisson)
        Same as ``get_map_xi``.
    """
    cache_dir  = Path(cache_dir)
    r_edges    = np.asarray(r_edges, dtype=float)
    edges_hash = hashlib.sha1(r_edges.tobytes()).hexdigest()[:8]
    thresh_str = 'auto' if mult_thresh is None else '{:.6g}'.format(mult_thresh)
    max_p_str  = 'inf' if not np.isfinite(max_p) else '{:.4e}'.format(max_p)
    cache_path = cache_dir / 'xi_{}_t{:d}_n{:.4e}_m{}_p{}_r{}.npz'.format(
        cache_key, int(timestep), abundance, thresh_str, max_p_str, edges_hash)

    if not force and cache_path.is_file():
        if verbose:
            print(f'    [cache hit] {cache_path.name}')
        d = np.load(cache_path)
        return d['xi'], d['poisson']

    if verbose:
        print(f'    [computing] {cache_path.name}')
    xi, poisson = get_map_xi(maplog, timestep, r_edges, abundance=abundance,
                             mult_thresh=mult_thresh, max_p=max_p, nthreads=nthreads)

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(cache_path.name + '.tmp')
    with open(tmp_path, 'wb') as fh:
        np.savez(fh, xi=xi, poisson=poisson)
    tmp_path.rename(cache_path)

    return xi, poisson


def plot_map_xi_zevolution(maplog, redshifts=(5, 4, 2, 0.2), rmin=2e-2, rmax=40.0,
                           nbins=30, abundance=5.01e-3, max_p=np.inf, nthreads=26,
                           cmap='Blues', clo=0.3, chi=1.0, figsize=(7, 5.5),
                           save=True, str_identifier=''):
    """MAP correlation function at several redshifts, overlaid in one figure.

    Unlike ``plot_correlation_functions`` (one figure per redshift, MAP vs.
    TNG), this is a single-run, single-figure view of how the MAP correlation
    function evolves with redshift: each entry of ``redshifts`` is resolved to
    its nearest available MAP timeslice step (``map_step_for_redshift``),
    ``get_map_xi`` computes the abundance-thresholded r^2 xi(r) curve at that
    step, and every requested redshift is drawn on the same axis, colored by
    redshift on a shared normalization (lightest = lowest z).

    Parameters
    ----------
    maplog : dict
        Output of ``read_maplog_dir``.
    redshifts : sequence of float
        Target redshifts; each is resolved to its nearest available MAP step
        (duplicates after resolution, e.g. two requests landing on the same
        step, are collapsed to one curve).
    rmin, rmax, nbins : float, float, int
        Log-spaced separation binning (Mpc/h) for the pair counts.
    abundance : float
        Target number density (Mpc/h)^-3 used by ``get_threshold``.
    max_p, nthreads :
        Passed through to ``get_xi``.
    cmap : str
        Colormap encoding redshift.
    clo, chi : float
        Lower/upper fraction of ``cmap`` to sample (avoids near-white ends).
    save : bool
        If True, save the figure under ``Images/``.
    str_identifier : str
        Identifies the simulation: used in the figure title and (when
        ``save`` is True) the filename, so batch runs over many simulations
        don't collide on the same ``Images/`` path.

    Returns
    -------
    (fig, ax)
    """
    redshifts = np.atleast_1d(redshifts).astype(float)
    r_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
    r = r_edges[:-1] + np.diff(r_edges) / 2.0

    # resolve every request to its nearest MAP step, dropping duplicate steps
    steps, zs = [], []
    seen_steps = set()
    for z_req in sorted(redshifts, reverse=True):
        step, z = map_step_for_redshift(maplog, z_req)
        if step in seen_steps:
            continue
        seen_steps.add(step)
        steps.append(step)
        zs.append(z)

    norm = Normalize(vmin=min(zs), vmax=max(zs))
    cmap_obj = colormaps[cmap]

    fig, ax = plt.subplots(1, figsize=figsize)
    for step, z in zip(steps, zs):
        xi, poisson = get_map_xi(maplog, step, r_edges, abundance=abundance,
                                 max_p=max_p, nthreads=nthreads)
        shade = clo + (chi - clo) * (norm(z) if max(zs) > min(zs) else 0.5)
        c = cmap_obj(shade)
        ax.plot(r, r**2 * xi, '-', lw=2, color=c, label=f'z={z:.2f} (step {step})')
        ax.fill_between(r, r**2 * (xi - poisson), r**2 * (xi + poisson), color=c, alpha=0.3)

    ax.grid()
    ax.set(xscale='log', xlabel=r'r [Mpc/h]', ylabel=r'$r^2\xi_{MAP}(r)$')
    if str_identifier:
        ax.set_title(str_identifier)
    ax.legend()

    if save:
        ids = '_' + str_identifier if str_identifier else ''
        fig.savefig(f'Images/map_xi_zevolution{ids}.png', bbox_inches='tight')
    plt.show()
    return fig, ax


def plot_mult_histogram(maplog, timestep=None, bins=None, figsize=(6, 4.5),
                        save=True, str_identifier=''):
    """Histogram of MAP multiplicity at one timeslice, log-log axes.

    Selects the timeslice nodes (``control == 4*timestep + 3``) at
    ``timestep`` and histograms their ``mult`` on log-spaced bins.

    Parameters
    ----------
    maplog : dict
        Output of ``read_maplog_dir``.
    timestep : int, optional
        Timeslice step to histogram. Default: the last (highest) timestep
        present in ``maplog``, i.e. the final MAP population.
    bins : array-like, optional
        Bin edges. Default: 30 log-spaced bins spanning the selected
        timestep's ``mult`` range.
    save : bool
        If True, save the figure under ``Images/``.
    str_identifier : str
        Identifies the simulation: used in the figure title and (when
        ``save`` is True) the filename, so batch runs over many simulations
        don't collide on the same ``Images/`` path.

    Returns
    -------
    (fig, ax)
    """
    if timestep is None:
        timestep = int(np.unique(maplog['timestep'])[-1])
    timeslice = maplog['control'] == int(timestep) * 4 + 3
    mult = maplog['mult'][timeslice]

    if bins is None:
        bins = np.logspace(np.log10(max(mult.min(), 1)), np.log10(mult.max()), 30)

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.hist(mult, bins=bins, log=True)
    ax.set(xscale='log', xlabel='MAP multiplicity', ylabel='counts')
    if str_identifier:
        ax.set_title(str_identifier)
    ax.grid()

    if save:
        ids = '_' + str_identifier if str_identifier else ''
        fig.savefig(f'Images/mult_histogram{ids}.png', bbox_inches='tight')
    plt.show()
    return fig, ax


def get_map_large_small_cross_xi(maplog, timestep, r_edges, abundance, min_mult=1e1,
                                  mult_thresh=None, max_p=np.inf, nthreads=26):
    """MAP cross-correlation between large and small MAPs on either side of one cut.

    Selects the timeslice nodes (control == 4*timestep + 3) and computes a single
    multiplicity threshold from ``abundance`` via ``get_threshold``. Splits every
    MAP with ``mult >= min_mult`` (i.e. above a noise floor, not spurious/tiny) into
    two disjoint populations: "large" (``mult >= thresh``, the ``abundance`` cut
    itself) and "small" (``min_mult <= mult < thresh``). Cross-correlates the two
    with ``get_xi``. Returns ``(xi, poisson)`` on the bin centers implied by
    ``r_edges``.

    If ``mult_thresh`` is given, it is used directly as the "large" cut instead
    of recomputing one from ``abundance`` at this timestep (see ``get_map_xi``).
    """
    BoxSize   = maplog['header']['BoxSize'] * u.Mpc
    timeslice = maplog['control'] == int(timestep) * 4 + 3
    thresh    = mult_thresh if mult_thresh is not None else \
                get_threshold(maplog['mult'][timeslice], BoxSize, guess=min_mult,
                             abundance=abundance)
    msk_large = timeslice & (maplog['mult'] >= thresh)
    msk_small = timeslice & (maplog['mult'] >= min_mult) & (maplog['mult'] < thresh)
    twopt     = get_xi(maplog['pos'][msk_small, :], r_edges, BoxSize,
                       pos2=maplog['pos'][msk_large, :], max_p=max_p, nthreads=nthreads)
    xi        = twopt.get_corr(return_sep=False)
    poisson   = np.sqrt(twopt.D1D2.wcounts) / twopt.R1R2.wcounts
    return xi, poisson


def get_map_matter_xi(sdir, z_req, r_edges, abundance=5.01e-3, max_p=np.inf,
                       nthreads=26, subsample_type='P8'):
    """MAP x matter cross-correlation for one subsample directory at one redshift.

    Reads the fair particle subsample nearest ``z_req`` via ``read_subsample_dir``,
    abundance-thresholds the MAP-flagged particles on multiplicity (same recipe as
    ``get_map_xi``), and cross-correlates them against the *full* subsample (an
    unbiased fair sample of all particles, used as the matter-field tracer) with
    ``get_xi``. Drawing both MAP and matter positions from the same fair subsample
    keeps their selection functions consistent. Returns ``(xi, poisson)`` on the
    bin centers implied by ``r_edges``.
    """
    subsample = read_subsample_dir(sdir, z_desired=z_req, subsample_type=subsample_type,
                                   keys=['pos', 'is_map', 'mult'])
    BoxSize   = subsample['BoxSize']
    is_map    = subsample['is_map']
    thresh    = get_threshold(subsample['mult'][is_map], BoxSize, guess=1e1, abundance=abundance)
    msk       = is_map & (subsample['mult'] >= thresh)
    twopt     = get_xi(subsample['pos'][msk, :], r_edges, BoxSize, pos2=subsample['pos'],
                       max_p=max_p, nthreads=nthreads)
    xi        = twopt.get_corr(return_sep=False)
    poisson   = np.sqrt(twopt.D1D2.wcounts) / twopt.R1R2.wcounts
    return xi, poisson


TNG300_1_BASEPATH = '/flare/Abacus/ajohnson/TNG300/TNG300-1/output/'
TNG300_2_BASEPATH = '/flare/Abacus/ajohnson/TNG300/TNG300-2/output/'

FLAMINGO_CACHE_DIR = '/flare/Abacus/ajohnson/FLAMINGO/cache'
FLAMINGO_L1_M8_BASEPATH = 'FLAMINGO/L1_m8/L1_m8/SOAP-HBT/'


def load_flamingo_halos(step, root_dir=None, basePath=FLAMINGO_L1_M8_BASEPATH,
                        cache_dir=FLAMINGO_CACHE_DIR):
    """Load (and locally cache) FLAMINGO SOAP halo positions/masses for one snapshot.

    ``step`` is the FLAMINGO snapshot number (e.g. 74 for the z=0 catalog used in
    the FLAMINGO section of ``analysis.ipynb``); the hdfstream path is built as
    ``<basePath>halo_properties_<step:04d>.hdf5``. The first call for a given step
    fetches it over the network via ``hdfstream``/``swiftsimio`` (needs
    ``root_dir``, an open ``hdfstream.open('cosma', '/')`` directory) and writes a
    plain ``.npz`` cache to ``<cache_dir>/halo_properties_<step:04d>.npz``; every
    subsequent call for that step -- in this session or a later one -- reads the
    local cache instead, no network required, and ``root_dir`` can be omitted.

    Positions are converted to comoving Mpc/h (matching TNG's ``SubhaloPos`` and
    the MAP/maplog convention used elsewhere in this module) by scaling the
    swiftsimio comoving-Mpc quantities by ``h``. Mass is left in its native units
    since only its rank order matters for ``get_threshold`` abundance cuts.

    Returns a dict with keys ``pos`` ``(N, 3)`` ndarray, ``mass`` ``(N,)`` ndarray,
    ``boxsize`` (float, comoving Mpc/h), ``h``, and ``redshift``.
    """
    step = int(step)
    cache_path = Path(cache_dir) / 'halo_properties_{:04d}.npz'.format(step)
    if cache_path.is_file():
        d = np.load(cache_path)
        return dict(pos=d['pos'], mass=d['mass'], boxsize=float(d['boxsize']),
                   h=float(d['h']), redshift=float(d['redshift']))

    if root_dir is None:
        raise ValueError(
            f"{cache_path} is not cached yet; pass root_dir=hdfstream.open('cosma', '/') "
            f"to fetch step {step} over the network")

    import swiftsimio as sw
    remote_path = '{}halo_properties_{:04d}.hdf5'.format(basePath, step)
    soap = sw.load(root_dir[remote_path])

    h        = soap.metadata.cosmology.H0.value / 100
    boxsize  = float(soap.metadata.boxsize[0].to('Mpc').value * h)
    pos      = np.asarray(soap.bound_subhalo.centre_of_mass.to('Mpc').value) * h
    mass     = np.asarray(soap.bound_subhalo.stellar_mass.value)
    redshift = float(soap.metadata.redshift)

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, pos=pos, mass=mass, boxsize=boxsize, h=h, redshift=redshift)
    return dict(pos=pos, mass=mass, boxsize=boxsize, h=h, redshift=redshift)


def compare_tng_flamingo_correlation_functions(redshifts, flamingo_snapshots, root_dir=None,
                                               tng_basePath=TNG300_2_BASEPATH, tng_label='TNG300-2',
                                               flamingo_basePath=FLAMINGO_L1_M8_BASEPATH,
                                               ztol=5e-3, rmin=2e-2, rmax=40.0, nbins=30,
                                               type_str='gg', abundance=5.01e-3,
                                               max_p=np.inf, nthreads=26, panel_size=(4, 3),
                                               save=False, str_identifier='',
                                               cache_dir=FLAMINGO_CACHE_DIR):
    """Compare TNG and FLAMINGO halo correlation functions across redshifts.

    One column per entry of ``redshifts``. Each panel overplots the TNG galaxy
    (stellar-mass-thresholded subhalo, red dashed) and FLAMINGO halo (blue solid)
    correlation functions at the same target abundance, each independently
    resolved to the snapshot nearest that column's redshift (within ``ztol``).
    No MAP/maplog curve is drawn -- this is purely a TNG-vs-FLAMINGO check; see
    ``compare_correlation_functions_flamingo`` to also overlay MAP runs (against
    a single fixed FLAMINGO snapshot rather than one resolved per column).

    Parameters
    ----------
    redshifts : float or sequence of float
        Target redshift(s); one column per entry.
    flamingo_snapshots : dict {float: int}
        Maps an available FLAMINGO redshift to its snapshot step number, e.g.
        ``{5: 10, 4: 14, 3: 18, 2: 38, 1: 58, 0.5: 68, 0.2: 74}``. Each snapshot's
        path is built from ``flamingo_basePath`` and the step number (see
        ``load_flamingo_halos``) and loaded (and thus cached) at most once, even
        if reused across columns.
    root_dir :
        Open hdfstream root directory (``hdfstream.open('cosma', '/')``); only
        needed for FLAMINGO snapshots not already cached on disk (see
        ``load_flamingo_halos``).
    tng_basePath, tng_label : str, str
        TNG group-catalog base path and its legend label.
    flamingo_basePath : str
        hdfstream directory holding the FLAMINGO SOAP catalogs; passed through to
        ``load_flamingo_halos``.
    ztol : float
        Max redshift difference between a requested redshift and the closest
        available TNG or FLAMINGO snapshot. A request with no snapshot of either
        catalog within ``ztol`` raises.
    rmin, rmax, nbins : float, float, int
        Log-spaced separation binning (Mpc/h) for the pair counts.
    type_str : str
        Correlation-function subscript in the y-axis label (e.g. 'gg').
    abundance : float
        Target number density (Mpc/h)^-3 used by ``get_threshold``.
    max_p, nthreads :
        Passed through to ``get_xi``.
    panel_size : tuple
        (width, height) in inches of a single subplot; the figure scales with the
        number of columns.
    save : bool
        If True, save the figure under Images/.
    str_identifier : str
        Filename suffix used when ``save`` is True.
    cache_dir :
        Passed through to ``load_flamingo_halos``.

    Returns
    -------
    (fig, axes) : the Figure and the 1D ndarray of Axes (one per redshift).
    """
    redshifts = np.atleast_1d(redshifts).astype(float)
    ncols = len(redshifts)

    r_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
    r       = r_edges[:-1] + np.diff(r_edges) / 2.0

    tng_snapnos, tng_zs = get_available_tng_snapshots(tng_basePath)
    star_type = il.snapshot.partTypeNum('star')
    flamingo_zs = np.array(sorted(flamingo_snapshots.keys()))

    fig, axes = plt.subplots(1, ncols, sharex=True, sharey=True, squeeze=False,
                             figsize=(panel_size[0] * ncols, panel_size[1]))
    fig.subplots_adjust(wspace=0)
    axes = axes[0]

    for col, z_req in enumerate(redshifts):
        ax = axes[col]

        # --- TNG: stellar-mass-thresholded subhalos ---
        k = np.argmin(np.abs(tng_zs - z_req))
        if abs(tng_zs[k] - z_req) > ztol:
            raise ValueError(
                "requested redshift {:.3f} has no TNG snapshot within {:g}; "
                "available TNG redshifts are {}".format(
                    z_req, ztol, ', '.join('{:.2f}'.format(z) for z in tng_zs)))
        sn           = int(tng_snapnos[k])
        tng_subhalos = load_tng_subhalos(tng_basePath, sn)
        m_tng        = tng_subhalos['SubhaloMassType'][:, star_type]
        tng_boxsize  = il.groupcat.loadHeader(tng_basePath, sn)['BoxSize'] / 1e3 * u.Mpc
        tng_thresh   = get_threshold(m_tng, tng_boxsize, guess=1e-3, abundance=abundance)
        msk_tng      = m_tng >= tng_thresh
        twopt_tng    = get_xi(tng_subhalos['SubhaloPos'][msk_tng, :], r_edges, tng_boxsize,
                              max_p=max_p, nthreads=nthreads)
        xi_tng       = twopt_tng.get_corr(return_sep=False)
        poisson_tng  = np.sqrt(twopt_tng.D1D2.wcounts) / twopt_tng.R1R2.wcounts

        # --- FLAMINGO: mass-thresholded halos ---
        j = np.argmin(np.abs(flamingo_zs - z_req))
        if abs(flamingo_zs[j] - z_req) > ztol:
            raise ValueError(
                "requested redshift {:.3f} has no FLAMINGO snapshot within {:g}; "
                "available FLAMINGO redshifts are {}".format(
                    z_req, ztol, ', '.join('{:.2f}'.format(z) for z in flamingo_zs)))
        step             = flamingo_snapshots[float(flamingo_zs[j])]
        flam             = load_flamingo_halos(step, root_dir=root_dir,
                                               basePath=flamingo_basePath, cache_dir=cache_dir)
        flamingo_boxsize = flam['boxsize'] * u.Mpc
        flamingo_thresh  = get_threshold(flam['mass'], flamingo_boxsize, guess=1e-3, abundance=abundance)
        msk_flamingo     = flam['mass'] >= flamingo_thresh
        twopt_flamingo   = get_xi(flam['pos'][msk_flamingo, :], r_edges, flamingo_boxsize,
                                  max_p=max_p, nthreads=nthreads)
        xi_flamingo      = twopt_flamingo.get_corr(return_sep=False)
        poisson_flamingo = np.sqrt(twopt_flamingo.D1D2.wcounts) / twopt_flamingo.R1R2.wcounts

        for xi, poisson, label, style in [
            (xi_tng, poisson_tng, tng_label, dict(color='red', ls='--')),
            (xi_flamingo, poisson_flamingo, 'FLAMINGO', dict(color='blue', ls='-')),
        ]:
            y, yerr = r**2 * xi, r**2 * poisson
            ax.plot(r, y, lw=2, label=label, **style)
            ax.fill_between(r, y - yerr, y + yerr, color=style['color'], alpha=0.3)

        ax.set(xscale='log')
        ax.grid()
        ax.set_title('z = {:.2f}'.format(z_req))
        if col == 0:
            ax.legend(fontsize='small')

    for ax in axes:
        ax.set_xlabel(r'r [$h^{-1}Mpc$]')
    axes[0].set_ylabel(r'$r^2\xi_{%s}(r)$' % type_str)

    if save:
        fig.savefig('../Images/compare_tng_flamingo_n{:.2e}_'.format(abundance)
                    + str_identifier + '.png', bbox_inches='tight', transparent=True)
    plt.show()
    return fig, axes


def _ratio_suffix(plot_ratio):
    """Filename suffix marking a ratio plot, so ratio and non-ratio figures from the
    same call never overwrite each other: ``''`` (off), ``'_ratio'``, or
    ``'_sqrt_ratio'`` for ``plot_ratio == 2`` (the sqrt-of-ratio / bias-ratio plot)."""
    if not plot_ratio:
        return ''
    return '_sqrt_ratio' if plot_ratio == 2 else '_ratio'


def _xi_plot_values(r, xi, poisson, plot_ratio=False, ref=None):
    """Return the ``(y, yerr)`` a correlation-function curve should be drawn with.

    Normally ``y = r**2 * xi`` with Poisson band ``r**2 * poisson``. If
    ``plot_ratio`` is True, ``ref`` must be the ``(xi_ref, poisson_ref)`` of the
    curve to divide by (e.g. the TNG curve for that column): returns
    ``xi / xi_ref`` instead, with the uncertainty propagated from both curves'
    Poisson bands (assuming they're independent).

    ``plot_ratio`` may also be the integer 2 (``True`` is 1): the square root of
    that ratio is returned instead -- the ratio of the two samples' linear
    biases -- with the fractional error halved.
    """
    if plot_ratio:
        if ref is None:
            raise ValueError(
                "plot_ratio=True requires a reference correlation function "
                "(e.g. enable plot_tng2/plot_tng1/plot_tng)")
        xi_ref, poisson_ref = ref
        y = xi / xi_ref
        yerr = np.abs(y) * np.sqrt((poisson / xi)**2 + (poisson_ref / xi_ref)**2)
        if plot_ratio == 2:
            y = np.sqrt(y)
            yerr = yerr / (2 * y)    # d sqrt(x) = dx / (2 sqrt(x))
    else:
        y = r**2 * xi
        yerr = r**2 * poisson
    return y, yerr


def compare_correlation_functions(maplog_tuples, redshifts, labels=None,
                                   row_names=None,
                                   plot_tng2=False, plot_tng1=False,
                                   plot_flamingo=False, flamingo_snapshots=None,
                                   root_dir=None, flamingo_label='FLAMINGO',
                                   flamingo_basePath=FLAMINGO_L1_M8_BASEPATH,
                                   cache_dir=FLAMINGO_CACHE_DIR,
                                   ztol=5e-2,
                                   rmin=2e-2, rmax=40.0, nbins=30, type_str='gg',
                                   abundance=5.01e-3, max_p=1e7, nthreads=26,
                                   cmaps=('Blues_r', 'Greens_r', 'Purples_r', 'Oranges_r'),
                                   clo=0., chi=0.7, panel_size=(4, 3),
                                   save=False, str_identifier='',
                                   map_default=None, default_label='',
                                   maplog_default_npz=None, maplog_default_r_edges=None,
                                   plot_ratio=False, xscale='log',
                                   xi_cache_dir=None, xi_cache_keys=None,
                                   default_cache_key=None,
                                   show_redshift_in_legend=True,
                                   ylim=None, yticks=None,
                                   weighted=False):
    """Compare MAP correlation functions across maplogs on a shared grid.

    Builds an ``nrows x ncols`` grid of subplots where ``nrows`` is the number of
    maplog tuples and ``ncols`` is the number of requested redshifts. Each tuple
    of maplogs is drawn on one row; each redshift is a column. The subplots share
    the x-axis across the whole grid and are packed with no spacing between them;
    the y-axis is shared only within a row, so each row's limits are set from the
    tallest curve in that row rather than the tallest curve anywhere in the grid.

    Within each subplot the curves are the MAP correlation functions of the
    maplogs in that row's tuple, all evaluated at the column's redshift (each
    maplog resolves the redshift to its own nearest timeslice step via
    ``map_step_for_redshift``). All maplogs in the tuple are drawn in shades of a
    single-color colormap chosen per row from ``cmaps`` (cycled if there are more
    rows than colormaps). The near-white end of each colormap is excluded by
    sampling on ``[clo, chi]``; a given maplog keeps the same shade across all
    columns.

    If ``map_default`` is given, its MAP correlation function is drawn as a gray
    dashed reference in every subplot. It represents the single result that all
    the others are compared against, so its correlation function is computed only
    once per column (redshift) and reused across all rows rather than recomputed
    in every subplot.

    Parameters
    ----------
    maplog_tuples : sequence of tuple of dict
        One tuple per row; each tuple is a set of maplogs (outputs of
        ``read_maplog_dir``) to overplot. Every entry is drawn as a colored
        curve.
    redshifts : float or sequence of float
        Target redshift(s); one column per entry.
    labels : sequence of tuple of str, optional
        Same shape as ``maplog_tuples``. ``labels[row][i]`` is prepended (before
        the redshift) to the legend label of the corresponding maplog's curve.
    row_names : sequence of str, optional
        One entry per row, drawn as that row's y-axis label (leftmost column
        only) so rows that would otherwise look identical in their per-curve
        legends -- e.g. two rows both varying ``Omega_M`` but through
        different parameter combinations -- are still distinguishable from the
        figure alone.
    plot_tng2 : bool or int
        If truthy, also overplot the TNG300-2-galaxy correlation function (red
        dashed) in every subplot, using the snapshot matched to the column's
        redshift. Red is reserved for TNG, which is why it is absent from
        ``cmaps``. Like ``map_default``, this depends only on the column
        (redshift), so it is computed once per column and reused across rows
        rather than recomputed in every subplot.
    plot_tng1 : bool or int
        Same as ``plot_tng2`` but for TNG300-1 (dash-dot, orange). The two are
        independent and can both be enabled at once. Both may be integers: with
        ``plot_ratio``, the one with the higher value is the ratio reference
        (True counts as 1; ties go to TNG300-2), e.g. ``plot_tng1=2,
        plot_tng2=1`` draws both with TNG300-1 as the reference.
    plot_flamingo : bool
        If True, also overplot the FLAMINGO (SOAP) correlation function (black
        dotted) in every subplot, using the snapshot matched to the column's
        redshift. Independent of ``plot_tng2``/``plot_tng1`` and can be combined
        with either or both. Like them, it depends only on the column
        (redshift), so it is computed once per column and reused across rows.
        Requires ``flamingo_snapshots``.
    flamingo_snapshots : dict {float: int}, optional
        Maps an available FLAMINGO redshift to its snapshot step number, e.g.
        ``{5: 10, 4: 14, 3: 18, 2: 38, 1: 58, 0.5: 68, 0.2: 74}`` (see
        ``load_flamingo_halos``). Required if ``plot_flamingo``.
    root_dir :
        Open hdfstream root directory (``hdfstream.open('cosma', '/')``); only
        needed for FLAMINGO snapshots not already cached on disk (see
        ``load_flamingo_halos``).
    flamingo_label : str
        Legend label for the FLAMINGO curve.
    flamingo_basePath : str
        hdfstream directory holding the FLAMINGO SOAP catalogs; passed through to
        ``load_flamingo_halos``.
    cache_dir :
        Passed through to ``load_flamingo_halos``.
    ztol : float
        Max redshift difference between a requested redshift and the closest
        available TNG/FLAMINGO snapshot, or the closest MAP timeslice in any
        maplog (including ``map_default``) (default 5e-2, i.e. agreement to
        one decimal place). A request with no match within ``ztol`` prints a
        warning (identifying the mismatched snapshot/timeslice) and proceeds
        using the nearest available one anyway; the actual redshift used is
        drawn into that curve's legend label, so the mismatch is visible and
        the caller can judge whether it's acceptable.
    rmin, rmax, nbins : float, float, int
        Log-spaced separation binning (Mpc/h) for the pair counts.
    type_str : str
        Correlation-function subscript in the y-axis label (e.g. 'gg').
    abundance : float
        Target number density (Mpc/h)^-3 used by ``get_threshold`` on MAP
        multiplicity.
    max_p, nthreads :
        Passed through to ``get_xi``.
    cmaps : sequence of str
        Per-row single-color colormaps for the non-reference curves, cycled by
        row.
    clo, chi : float
        Lower/upper fraction of each colormap to sample (avoids near-white ends).
    panel_size : tuple
        (width, height) in inches of a single subplot; the figure scales with the
        grid shape.
    save : bool
        If True, save the figure under Images/.
    str_identifier : str
        Filename suffix used when ``save`` is True.
    map_default : dict, optional
        A single maplog whose MAP correlation function is drawn as a gray dashed
        reference in every subplot (the result all others are compared against).
        Its curve is computed once per column and reused across rows.
    default_label : str
        Legend label for the ``map_default`` gray dashed curve.
    maplog_default_npz : str or path-like, optional
        Alternative to ``map_default``: path to a target ``.npz`` written by
        ``prod/ParamTuning*/optimize_maps.py::reference_target`` (keys
        ``redshifts``, ``xi_<i>``, ``err_<i>``). Its curves are drawn as the gray
        dashed reference exactly as ``map_default`` would be, but are *loaded*
        rather than recomputed -- so the huge reference maplog never has to be
        read, and the plot shows the very curves the tuning loss used. Each
        column takes the stored redshift nearest to it (``ztol`` warning as
        usual). Mutually exclusive with ``map_default``; cannot be combined with
        ``weighted`` (the stored curves are unweighted).
        The stored curves live on the optimizer's separation bins, so those bins
        are used for *every* curve (``rmin``/``rmax``/``nbins`` are ignored): they
        are read from the npz's ``r_edges`` key (written by current
        ``reference_target``), or else from ``maplog_default_r_edges``.
    maplog_default_r_edges : array-like, optional
        Bin edges (Mpc/h) the npz was computed on, for npz files written before
        ``r_edges`` was stored, e.g. ``np.linspace(R_MIN, R_MAX, N_R_BINS)``
        (or ``np.logspace(...)``). Must match the settings the file was
        built with. Ignored if the npz contains ``r_edges``.
    show_redshift_in_legend : bool
        If True (default), every legend entry carries its ``z=...`` redshift.
        If False the redshift is dropped (it is still in the column titles), so
        only the descriptive labels in the left-most column remain; columns left
        with no labelled curves get no legend.
    plot_ratio : int {0, 1, 2}
        0 (or False): plot ``r^2 xi(r)``. 1 (or True): plot every curve as its
        ratio to a reference correlation function of that column instead of
        ``r^2 xi(r)`` next to it. 2: plot the square root of that ratio, i.e. the
        ratio of the linear biases (the dotted lines then mark +/-10% of the
        plotted quantity, and the y-range is narrowed accordingly). The
        reference is the TNG comparison with the higher ``plot_tng1``/``plot_tng2``
        value (ties: ``plot_tng2``), else ``plot_flamingo``; if none of those are
        enabled, ``map_default`` is used as the reference instead (as in
        ``plot_cosmo_variations.py``, which has no TNG/FLAMINGO overlay). The
        reference curve itself is drawn as a horizontal line at 1 instead of
        its own curve. Requires at least one of
        ``plot_tng2``/``plot_tng1``/``plot_flamingo``/``map_default``.
    xscale : {'log', 'linear'}
        Scale of the x-axis (separation ``r``).
    ylim : tuple (ymin, ymax), optional
        y-axis limits applied to every row (rows share a y-axis only within
        themselves, so each is set). Overrides the automatic per-row limits,
        including the fixed ratio-mode ranges. Either entry may be None to leave
        that end automatic.
    yticks : sequence of float, optional
        y-tick positions (labelled at those values) for every row. Ticks outside a
        row's limits are not shown, and since the panels touch, a tick at a row's
        limit overlaps its neighbour's label, so keep them inside the limits.
        Default: automatic, except with ``plot_ratio`` where three ticks symmetric
        about 1 are chosen from the row's limits (0.8, 1, 1.2 for the default
        (0.7, 1.3) range).
    xi_cache_dir : path-like, optional
        If given (together with ``xi_cache_keys``/``default_cache_key``), every
        correlation function is computed through ``get_map_xi_cached`` instead
        of ``get_map_xi``, so a value already cached to this directory (e.g. by
        ``cache_cosmo_variations_xi.py``) is loaded from its ``.npz`` file
        rather than recomputed -- letting ``maplog_tuples``/``map_default``
        be lightweight dicts (e.g. ``read_maplog_dir(..., keys=[])``) that
        never touch ``pos``/``mult``/``control`` at all, as long as every
        (directory, timestep) asked for is already cached. A genuine cache
        miss still calls ``get_map_xi`` on the passed-in maplog and therefore
        still needs those keys present -- this only changes whether an
        already-cached value is reused, not whether an uncached one can be
        computed from a trimmed maplog. If ``None`` (the default), every
        correlation function is recomputed directly via ``get_map_xi``, as
        before.
    xi_cache_keys : sequence of tuple of str, optional
        Same shape as ``maplog_tuples``; ``xi_cache_keys[row][m]`` is the
        ``cache_key`` passed to ``get_map_xi_cached`` for ``maplog_tuples[row][m]``
        (see that function's ``cache_key`` docstring -- must match whatever
        the cache was originally populated under, e.g. ``Path(mdir).name``).
        Required if ``xi_cache_dir`` is given.
    default_cache_key : str, optional
        ``cache_key`` for ``map_default``, analogous to ``xi_cache_keys``.
        Required if ``xi_cache_dir`` and ``map_default`` are both given.
    weighted : bool
        If True, every curve (including ``map_default``) is the multiplicity-weighted auto-correlation function: each MAP
        counts with a weight equal to its multiplicity. The MAP selection (abundance threshold, timeslice) is unchanged
        (see ``get_map_xi``). The error band is the pair-count shot noise (``xi_poisson``), which ignores the variance
        added by unequal weights. Cannot be combined with ``xi_cache_dir``: cached values are unweighted, so that raises.
        The TNG / FLAMINGO overlays are unaffected.

    Returns
    -------
    (fig, axes) : the Figure and the 2D ndarray of Axes.
    """
    redshifts = np.atleast_1d(redshifts).astype(float)
    nrows = len(maplog_tuples)
    ncols = len(redshifts)

    if weighted and xi_cache_dir is not None:
        raise ValueError("weighted cannot be combined with xi_cache_dir (cached correlation functions are unweighted)")

    if labels is not None:
        assert len(labels) == nrows, "labels must have one tuple per maplog tuple (row)"
        for row, (mls, labs) in enumerate(zip(maplog_tuples, labels)):
            assert len(labs) == len(mls), \
                "labels[{}] must match the number of maplogs in row {}".format(row, row)
    if row_names is not None:
        assert len(row_names) == nrows, "row_names must have one entry per maplog tuple (row)"

    if maplog_default_npz is not None:
        if map_default is not None:
            raise ValueError("pass either map_default or maplog_default_npz, not both")
        if weighted:
            raise ValueError("weighted cannot be combined with maplog_default_npz (stored curves are unweighted)")
    have_default = map_default is not None or maplog_default_npz is not None

    npz_default = None
    if maplog_default_npz is not None:
        npz_default = dict(np.load(maplog_default_npz))
        npz_edges = npz_default.get('r_edges', maplog_default_r_edges)
        if npz_edges is None:
            raise ValueError(
                "{} has no 'r_edges' (older file); pass maplog_default_r_edges, e.g. "
                "np.linspace(R_MIN, R_MAX, N_R_BINS), matching how it was built".format(maplog_default_npz))
        r_edges = np.asarray(npz_edges, dtype=float)
        print("maplog_default_npz: using its {} separation bins ({:g}-{:g} Mpc/h) for all "
              "curves (rmin/rmax/nbins ignored)".format(len(r_edges) - 1, r_edges[0], r_edges[-1]))
    else:
        r_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
    r       = r_edges[:-1] + np.diff(r_edges) / 2.0

    # optional reference-catalog comparisons: each is (key, legend label, curve
    # style). 'key' is a TNG basePath for TNG comparisons, or the sentinel
    # 'flamingo' for the FLAMINGO comparison; it indexes into overlay_xi below.
    overlay_specs = []
    if plot_tng2:
        overlay_specs.append((TNG300_2_BASEPATH, 'TNG300-2', dict(color='red', ls='--')))
    if plot_tng1:
        overlay_specs.append((TNG300_1_BASEPATH, 'TNG300-1', dict(color='maroon', ls='-.')))
    if plot_flamingo:
        if flamingo_snapshots is None:
            raise ValueError("plot_flamingo=True requires flamingo_snapshots")
        overlay_specs.append(('flamingo', flamingo_label, dict(color='black', ls=':')))

    if plot_ratio and not overlay_specs and not have_default:
        raise ValueError("plot_ratio=True requires plot_tng2, plot_tng1, plot_flamingo, or "
                         "map_default to provide the reference correlation function")
    # ratio reference: the TNG comparison with the highest plot_tng* value (ties
    # go to TNG300-2, then TNG300-1); else FLAMINGO if enabled
    tng_prio = [(int(plot_tng2), 1, TNG300_2_BASEPATH), (int(plot_tng1), 0, TNG300_1_BASEPATH)]
    tng_prio = [t for t in tng_prio if t[0] > 0]
    if tng_prio:
        ref_key = max(tng_prio)[2]
    else:
        ref_key = overlay_specs[0][0] if overlay_specs else None
    # No TNG/FLAMINGO overlay enabled: fall back to map_default as the ratio
    # reference (every curve's own ratio to the fiducial), as used by
    # plot_cosmo_variations.py. map_default's own curve then draws as a flat
    # line at 1 (see below) instead of dividing itself by itself.
    ratio_uses_default = plot_ratio and not overlay_specs

    # resolve each column redshift to a TNG snapshot (within ztol of an available one),
    # independently per TNG comparison (their snapshot lists need not match)
    star_type = il.snapshot.partTypeNum('star') if (plot_tng2 or plot_tng1) else None
    tng_col_snap = {}  # basePath -> array of snapnos, one per column
    tng_col_z    = {}  # basePath -> array of the *actual* matched snapshot redshifts
    for basePath, _, _ in overlay_specs:
        if basePath == 'flamingo':
            continue
        tng_snapnos, tng_zs = get_available_tng_snapshots(basePath)
        col_snap = np.empty(ncols, dtype=int)
        col_z    = np.empty(ncols, dtype=float)
        for col, z_req in enumerate(redshifts):
            k = np.argmin(np.abs(tng_zs - z_req))
            if abs(tng_zs[k] - z_req) > ztol:
                print(
                    "WARNING: requested redshift {:.3f} has no TNG snapshot within {:g} "
                    "(2 decimal places); using nearest available z={:.3f} for {} "
                    "(the mismatch is visible in the legend's redshift label)".format(
                        z_req, ztol, tng_zs[k], basePath))
            col_snap[col] = int(tng_snapnos[k])
            col_z[col]    = float(tng_zs[k])
        tng_col_snap[basePath] = col_snap
        tng_col_z[basePath]    = col_z

    # resolve each column redshift to a FLAMINGO snapshot (within ztol of an available one)
    flamingo_col_step = flamingo_col_z = None
    if plot_flamingo:
        flamingo_zs = np.array(sorted(flamingo_snapshots.keys()))
        flamingo_col_step = np.empty(ncols, dtype=int)
        flamingo_col_z    = np.empty(ncols, dtype=float)
        for col, z_req in enumerate(redshifts):
            j = np.argmin(np.abs(flamingo_zs - z_req))
            if abs(flamingo_zs[j] - z_req) > ztol:
                print(
                    "WARNING: requested redshift {:.3f} has no FLAMINGO snapshot within {:g}; "
                    "using nearest available z={:.3f} "
                    "(the mismatch is visible in the legend's redshift label)".format(
                        z_req, ztol, flamingo_zs[j]))
            flamingo_col_step[col] = flamingo_snapshots[float(flamingo_zs[j])]
            flamingo_col_z[col]    = float(flamingo_zs[j])

    # The gray dashed reference (map_default) is the single result everything is
    # compared against. Its correlation function depends only on the column
    # redshift, so compute it once per column here and reuse it across all rows.
    default_xi = default_poisson = default_z = None
    if npz_default is not None:
        default_xi      = [None] * ncols
        default_poisson = [None] * ncols
        default_z       = [None] * ncols
        zs_npz = np.atleast_1d(npz_default['redshifts']).astype(float)
        for col, z_req in enumerate(redshifts):
            j = int(np.argmin(np.abs(zs_npz - z_req)))
            if abs(zs_npz[j] - z_req) > ztol:
                print(
                    "WARNING: requested redshift {:.3f} has no maplog_default_npz entry within "
                    "{:g}; using nearest available z={:.4f}".format(z_req, ztol, zs_npz[j]))
            default_z[col]       = float(zs_npz[j])
            default_xi[col]      = np.asarray(npz_default[f'xi_{j}'])
            default_poisson[col] = np.asarray(npz_default[f'err_{j}'])
    elif map_default is not None:
        default_xi      = [None] * ncols
        default_poisson = [None] * ncols
        default_z       = [None] * ncols
        for col, z_req in enumerate(redshifts):
            ts, zm = map_step_for_redshift(map_default, z_req)
            if abs(zm - z_req) > ztol:
                print(
                    "WARNING: requested redshift {:.3f} has no map_default timeslice within "
                    "{:g} (3 decimal places); using nearest available z={:.4f} "
                    "(the mismatch is visible in the legend's redshift label)".format(
                        z_req, ztol, zm))
            default_z[col] = zm
            if xi_cache_dir is not None and default_cache_key is not None:
                default_xi[col], default_poisson[col] = get_map_xi_cached(
                    map_default, default_cache_key, ts, r_edges, xi_cache_dir,
                    abundance=abundance, max_p=max_p, nthreads=nthreads)
            else:
                default_xi[col], default_poisson[col] = get_map_xi(
                    map_default, ts, r_edges, abundance=abundance,
                    max_p=max_p, nthreads=nthreads, weighted=weighted)

    # Each reference-catalog comparison also depends only on the column
    # redshift (not the row), so compute its correlation function once per
    # column here and reuse it across all rows rather than recomputing it in
    # every subplot.
    overlay_xi = {}  # key -> [(xi, poisson), ...] one per column
    for basePath, _, _ in overlay_specs:
        if basePath == 'flamingo':
            continue
        col_xi = [None] * ncols
        for col in range(ncols):
            sn           = int(tng_col_snap[basePath][col])
            tng_subhalos = load_tng_subhalos(basePath, sn)
            m_tng        = tng_subhalos['SubhaloMassType'][:, star_type]  # 1e10 Msun
            # native TNG box size (header is ckpc/h, SubhaloPos is scaled to Mpc/h)
            tng_boxsize  = il.groupcat.loadHeader(basePath, sn)['BoxSize'] / 1e3 * u.Mpc
            tng_thresh   = get_threshold(m_tng, tng_boxsize, guess=1e-3, abundance=abundance)
            msk_tng      = m_tng >= tng_thresh
            twopt        = get_xi(tng_subhalos['SubhaloPos'][msk_tng, :], r_edges,
                                  tng_boxsize, max_p=max_p, nthreads=nthreads)
            xi_tng       = twopt.get_corr(return_sep=False)
            poisson_tng  = np.sqrt(twopt.D1D2.wcounts) / twopt.R1R2.wcounts
            col_xi[col] = (xi_tng, poisson_tng)
        overlay_xi[basePath] = col_xi

    if plot_flamingo:
        col_xi = [None] * ncols
        for col in range(ncols):
            step             = int(flamingo_col_step[col])
            flam             = load_flamingo_halos(step, root_dir=root_dir,
                                                    basePath=flamingo_basePath, cache_dir=cache_dir)
            flamingo_boxsize = flam['boxsize'] * u.Mpc
            flamingo_thresh  = get_threshold(flam['mass'], flamingo_boxsize, guess=1e-3,
                                             abundance=abundance)
            msk_flamingo     = flam['mass'] >= flamingo_thresh
            twopt            = get_xi(flam['pos'][msk_flamingo, :], r_edges, flamingo_boxsize,
                                      max_p=max_p, nthreads=nthreads)
            xi_flamingo      = twopt.get_corr(return_sep=False)
            poisson_flamingo = np.sqrt(twopt.D1D2.wcounts) / twopt.R1R2.wcounts
            col_xi[col] = (xi_flamingo, poisson_flamingo)
        overlay_xi['flamingo'] = col_xi

    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey='row', squeeze=False,
                             figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
    fig.subplots_adjust(wspace=0, hspace=0)

    def _legend_label(base, z, col):
        """Legend text for one curve: every column gets the redshift (unless
        ``show_redshift_in_legend`` is off), but only the left-most column also
        carries the descriptive ``base`` label -- every other column would
        otherwise just repeat it once per row."""
        if not show_redshift_in_legend:
            return base if (col == 0 and base) else '_nolegend_'
        z_str = f'z={z:.3f}'
        return f'{base} ({z_str})' if (col == 0 and base) else z_str

    # Panels share a y-axis only within their row, so the y-limit for a row has
    # to account for the tallest curve in *that row* (any column), not just
    # whichever panel happens to set it. Track the running max per row here and
    # apply it once after every panel is drawn instead of autoscaling off of
    # column 0 alone.
    data_ymax = np.zeros(nrows)

    for row, maplogs in enumerate(maplog_tuples):
        cmap   = colormaps[cmaps[row % len(cmaps)]]
        n_map  = len(maplogs)  # curves drawn from the colormap
        # one shade per maplog, held constant across the columns
        fracs  = clo + (chi - clo) * np.linspace(0, 1, max(n_map, 1))
        for col, z_req in enumerate(redshifts):
            ax = axes[row][col]
            if plot_ratio and overlay_specs:
                ref = overlay_xi[ref_key][col]
            elif ratio_uses_default:
                ref = (default_xi[col], default_poisson[col])
            else:
                ref = None

            # gray dashed reference, precomputed once per column
            if have_default:
                default_curve_label = _legend_label(default_label, default_z[col], col)
                if ratio_uses_default:
                    # ratio-to-itself is exactly 1 everywhere: draw a flat
                    # reference line (matching the overlay-as-its-own-ref case
                    # below, and plot_map_variations.py's style) instead of
                    # dividing map_default by itself.
                    ax.axhline(1.0, ls='--', lw=2, color='gray', label=default_curve_label)
                    ax.axhline(0.9, ls=':', lw=2.5, color='gray', alpha=0.8)
                    ax.axhline(1.1, ls=':', lw=2.5, color='gray', alpha=0.8)
                else:
                    xi_d, poisson_d = default_xi[col], default_poisson[col]
                    y_d, yerr_d = _xi_plot_values(r, xi_d, poisson_d, plot_ratio, ref)
                    ax.plot(r, y_d, '--', lw=2, color='gray', label=default_curve_label)
                    ax.fill_between(r, y_d - yerr_d, y_d + yerr_d, color='gray', alpha=0.3)
                    if not plot_ratio:
                        data_ymax[row] = max(data_ymax[row], np.nanmax(y_d + yerr_d))

            for m, maplog in enumerate(maplogs):
                ts, zm      = map_step_for_redshift(maplog, z_req)
                if abs(zm - z_req) > ztol:
                    print(
                        "WARNING: requested redshift {:.3f} has no timeslice within {:g} "
                        "(3 decimal places) in maplog_tuples[{}][{}]; using nearest "
                        "available z={:.4f} (the mismatch is visible in the legend's "
                        "redshift label)".format(z_req, ztol, row, m, zm))
                if xi_cache_dir is not None and xi_cache_keys is not None:
                    xi, poisson = get_map_xi_cached(
                        maplog, xi_cache_keys[row][m], ts, r_edges, xi_cache_dir,
                        abundance=abundance, max_p=max_p, nthreads=nthreads)
                else:
                    xi, poisson = get_map_xi(maplog, ts, r_edges, abundance=abundance,
                                             max_p=max_p, nthreads=nthreads, weighted=weighted)
                base_label = str(labels[row][m]) if labels is not None else ''
                label = _legend_label(base_label, zm, col)
                color = cmap(fracs[m])
                y, yerr = _xi_plot_values(r, xi, poisson, plot_ratio, ref)
                ax.plot(r, y, '-', lw=2, color=color, label=label)
                ax.fill_between(r, y - yerr, y + yerr, color=color, alpha=0.3)
                if not plot_ratio:
                    data_ymax[row] = max(data_ymax[row], np.nanmax(y + yerr))

            # --- optional reference catalogs (TNG galaxies / FLAMINGO halos),
            # precomputed once per column above and reused across rows ---
            for key, overlay_label, style in overlay_specs:
                xi_o, poisson_o = overlay_xi[key][col]
                overlay_z = flamingo_col_z[col] if key == 'flamingo' else tng_col_z[key][col]
                overlay_curve_label = _legend_label(overlay_label, overlay_z, col)
                if plot_ratio and key == ref_key:
                    # this is the reference curve itself: ratio to itself is 1
                    ax.axhline(1.0, lw=2, label=overlay_curve_label, **style)
                    continue
                y_o, yerr_o = _xi_plot_values(r, xi_o, poisson_o, plot_ratio, ref)
                ax.plot(r, y_o, lw=2, label=overlay_curve_label, **style)
                ax.fill_between(r, y_o - yerr_o, y_o + yerr_o, color=style['color'], alpha=0.3)
                if not plot_ratio:
                    data_ymax[row] = max(data_ymax[row], np.nanmax(y_o + yerr_o))

            ax.set(xscale=xscale)
            ax.grid()
            if ax.get_legend_handles_labels()[0]:   # none if every label was suppressed
                ax.legend(fontsize='small')
            if row == 0:
                man,exp = get_sci_not(abundance)
                ax.set_title(f'z = {z_req:.2f}') # , $n={:.1f}\times 10^{{{:d}}}\;\;[h^{{3}}Mpc^{{-3}}]$'.format(z_req,man,exp))

    # Each row shares a y-axis only with itself (sharey='row'), so the limit
    # is set per row from the actual max plotted across that row's columns,
    # rather than a single grid-wide limit that could clip a row whose
    # amplitude differs from the others.
    for row in range(nrows):
        if plot_ratio:
            # matches plot_map_variations.py's wider ratio-to-fiducial range
            if plot_ratio == 2:
                axes[row, 0].set_ylim(0.85, 1.15) if ratio_uses_default else axes[row, 0].set_ylim(0.9, 1.1)
            else:
                axes[row, 0].set_ylim(0.7, 1.3) if ratio_uses_default else axes[row, 0].set_ylim(0.8, 1.2)
        else:
            axes[row, 0].set_ylim(0, data_ymax[row] * 1.05)
        if ylim is not None:
            axes[row, 0].set_ylim(ylim)
        if yticks is not None:
            axes[row, 0].set_yticks(yticks)
        elif plot_ratio:
            # panels touch (hspace=0), so a tick at a row's y-limit collides with the
            # neighbouring row's: keep the ticks (1 - d, 1, 1 + d) strictly inside the
            # limits -- 0.8, 1, 1.2 for the default (0.7, 1.3) ratio range
            lo, hi = axes[row, 0].get_ylim()
            hw = (hi - lo) / 2
            d  = 0.2 if hw > 0.2 else 0.1 if hw > 0.1 else 0.05
            axes[row, 0].set_yticks([1 - d, 1.0, 1 + d])

    # shared axis labels on the outer edge only
    for ax in axes[-1, :]:
        ax.set_xlabel(r'r [$h^{-1}Mpc$]')
    if plot_ratio:
        ratio_ylabel = r'$\xi_{%s}/\xi_{%s,\rm ref}$' % (type_str, type_str)
        if plot_ratio == 2:
            ratio_ylabel = r'$\sqrt{\xi_{%s}/\xi_{%s,\rm ref}}$' % (type_str, type_str)
        for row in range(nrows):
            lbl = f'{row_names[row]}\n{ratio_ylabel}' if row_names is not None else ratio_ylabel
            axes[row, 0].set_ylabel(lbl, fontsize='small' if row_names is not None else None)
    elif row_names is not None:
        for row in range(nrows):
            axes[row, 0].set_ylabel(row_names[row], fontsize='small')

    if save:
        fig.savefig('../Images/compare_correlation_functions_n{:.2e}_'.format(abundance)
                    + str_identifier + _ratio_suffix(plot_ratio) + '.png',
                    bbox_inches='tight', transparent=True)
    plt.show()
    return fig, axes


def compare_multiplicity_functions(maplog_tuples, redshifts, labels=None,
                                   row_names=None, ztol=5e-2,
                                   mmin=10, mmax=4e4, nbins=24,
                                   cmaps=('Blues_r', 'Greens_r', 'Purples_r', 'Oranges_r'),
                                   clo=0., chi=0.7, panel_size=(4, 3),
                                   save=False, str_identifier='',
                                   map_default=None, default_label='',
                                   plot_ratio=False,
                                   show_redshift_in_legend=True,
                                   ylim=None, save_columns=False, _show=True):
    """Compare MAP multiplicity functions across maplogs on a shared grid.

    Same grid layout and conventions as ``compare_correlation_functions``: one
    row per tuple of maplogs, one column per redshift, x-axis shared across the
    grid, y-axis shared within a row, no spacing between panels, one
    single-color colormap per row (shades sampled on ``[clo, chi]``), and an
    optional gray dashed ``map_default`` reference drawn in every panel.

    Each curve is the number of MAPs per log-spaced multiplicity bin
    (``np.histogram`` of ``maplog['mult']`` on the timeslice nearest the column's
    redshift) drawn as a line through the bin centers rather than as bars, so
    several curves can share a panel. Empty bins are dropped.

    Parameters
    ----------
    maplog_tuples : sequence of tuple of dict
        One tuple per row; each is a set of maplogs (outputs of
        ``read_maplog_dir``) to overplot.
    redshifts : float or sequence of float
        Target redshift(s); one column per entry. Each maplog resolves it to its
        own nearest timeslice via ``map_step_for_redshift``.
    labels : sequence of tuple of str, optional
        Same shape as ``maplog_tuples``; legend label of each curve.
    row_names : sequence of str, optional
        One entry per row, drawn as that row's y-axis label (leftmost column).
    ztol : float
        Warn if the nearest timeslice is farther than this from the request.
    mmin, mmax, nbins : float, float, int
        ``nbins`` log-spaced bins from ``mmin`` to ``mmax`` in multiplicity.
    cmaps, clo, chi, panel_size, save, str_identifier :
        As in ``compare_correlation_functions`` (the saved file is
        ``Images/compare_multiplicity_functions_<str_identifier>.png``).
    map_default : dict, optional
        A single maplog drawn as a gray dashed reference in every panel (and the
        denominator when ``plot_ratio``).
    default_label : str
        Legend label for the ``map_default`` curve.
    plot_ratio : bool
        If True, plot each curve's counts divided by ``map_default``'s counts in
        the same bin (bins empty in either are dropped), with the default drawn
        as a flat line at 1 and dotted lines at 0.9/1.1. Requires ``map_default``.
    show_redshift_in_legend : bool
        If False, drop the ``z=...`` from legend entries (see
        ``compare_correlation_functions``).

    ylim : tuple (ymin, ymax), optional
        y-axis limits applied to every row (rows share a y-axis only within
        themselves, so each is set). Overrides the automatic limits, including
        the fixed ratio-mode range. Either entry may be None to leave that end
        automatic.
    save_columns : bool
        If True, also draw every column (one redshift, all rows) as its own
        standalone figure -- complete with y-axis labels, x-label, title and
        legend, e.g. for a paper, with the same x- and y-ranges and panel size as the grid's
        panels -- and save each to
        ``Images/compare_multiplicity_functions_<str_identifier>_z<z>[_ratio].png``
        with ``<z>`` the requested redshift to 2 decimals, so the files are told
        apart by redshift. Independent of ``save`` (which saves the whole grid):
        these files are written whether or not ``save`` is set. The returned
        ``(fig, axes)`` is still the combined grid.

    Returns
    -------
    (fig, axes) : the Figure and the 2D ndarray of Axes.
    """
    redshifts = np.atleast_1d(redshifts).astype(float)
    nrows = len(maplog_tuples)
    ncols = len(redshifts)
    if plot_ratio:
        assert map_default is not None, "plot_ratio requires map_default"
    if labels is not None:
        assert len(labels) == nrows, "labels must have one tuple per maplog tuple (row)"
        for row, (mls, labs) in enumerate(zip(maplog_tuples, labels)):
            assert len(labs) == len(mls), \
                "labels[{}] must match the number of maplogs in row {}".format(row, row)
    if row_names is not None:
        assert len(row_names) == nrows, "row_names must have one entry per maplog tuple (row)"

    bin_edges = np.logspace(np.log10(mmin), np.log10(mmax), nbins + 1)
    bins      = bin_edges[:-1] + np.diff(bin_edges) / 2.0

    def _counts(maplog, z_req, where):
        ts, zm = map_step_for_redshift(maplog, z_req)
        if abs(zm - z_req) > ztol:
            print("WARNING: requested redshift {:.3f} has no timeslice within {:g} "
                  "in {}; using nearest available z={:.4f}".format(z_req, ztol, where, zm))
        timeslice = maplog['control'] == ts * 4 + 3
        counts, _ = np.histogram(maplog['mult'][timeslice], bins=bin_edges)
        return counts, zm

    def _legend_label(base, z, col):
        if not show_redshift_in_legend:
            return base if (col == 0 and base) else '_nolegend_'
        z_str = f'z={z:.3f}'
        return f'{base} ({z_str})' if (col == 0 and base) else z_str

    # the default is shared by every row, so histogram it once per column
    default_counts, default_z = [], []
    if map_default is not None:
        for z_req in redshifts:
            c, zm = _counts(map_default, z_req, 'map_default')
            default_counts.append(c)
            default_z.append(zm)

    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey='row', squeeze=False,
                             figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
    fig.subplots_adjust(wspace=0, hspace=0)

    for row, maplogs in enumerate(maplog_tuples):
        cmap  = colormaps[cmaps[row % len(cmaps)]]
        fracs = clo + (chi - clo) * np.linspace(0, 1, max(len(maplogs), 1))
        for col, z_req in enumerate(redshifts):
            ax = axes[row][col]

            if map_default is not None:
                lbl = _legend_label(default_label, default_z[col], col)
                if plot_ratio:
                    ax.axhline(1.0, ls='--', lw=2, color='gray', label=lbl)
                    ax.axhline(0.9, ls=':', lw=2.5, color='gray', alpha=0.8)
                    ax.axhline(1.1, ls=':', lw=2.5, color='gray', alpha=0.8)
                else:
                    c = default_counts[col]
                    ax.plot(bins[c != 0], c[c != 0], '--', lw=2, color='gray', label=lbl)

            for m, maplog in enumerate(maplogs):
                counts, zm = _counts(maplog, z_req, f'maplog_tuples[{row}][{m}]')
                base_label = str(labels[row][m]) if labels is not None else ''
                label = _legend_label(base_label, zm, col)
                color = cmap(fracs[m])
                if plot_ratio:
                    d = default_counts[col]
                    ok = (counts != 0) & (d != 0)
                    ax.plot(bins[ok], counts[ok] / d[ok], '-', lw=2, color=color, label=label)
                else:
                    ax.plot(bins[counts != 0], counts[counts != 0], '-', lw=2, color=color, label=label)

            ax.set(xscale='log', yscale='linear' if plot_ratio else 'log')
            ax.grid()
            if ax.get_legend_handles_labels()[0]:   # none if every label was suppressed
                ax.legend(fontsize='small')
            if row == 0:
                ax.set_title(f'z = {z_req:.2f}')

    if plot_ratio:
        for row in range(nrows):
            axes[row, 0].set_ylim(0.7, 1.3)
    if ylim is not None:
        for row in range(nrows):
            axes[row, 0].set_ylim(ylim)

    for ax in axes[-1, :]:
        ax.set_xlabel(r'multiplicity $[\mathfrak{M}]$')
    ylabel = r'$N_{\rm MAPs}/N_{\rm MAPs,ref}$' if plot_ratio else r'$N_{\rm MAPs}$'
    for row in range(nrows):
        lbl = f'{row_names[row]}\n{ylabel}' if row_names is not None else ylabel
        axes[row, 0].set_ylabel(lbl, fontsize='small' if row_names is not None else None)

    if save:
        fig.savefig('../Images/compare_multiplicity_functions_' + str_identifier
                    + _ratio_suffix(plot_ratio) + '.png',
                    bbox_inches='tight', transparent=True)
    if save_columns:
        # one standalone figure per redshift (all rows, own axis labels/title/legend), built
        # by re-running this function on a single redshift rather than cropping the grid
        # the grid's axis ranges (x shared by all, y by row), reapplied to every figure so each
        # is exactly its grid column apart from labels
        xlim  = axes[0, 0].get_xlim()
        ylims = [axes[row, 0].get_ylim() for row in range(nrows)]
        for z_req in redshifts:
            fig_z, axes_z = compare_multiplicity_functions(
                maplog_tuples, [z_req], labels=labels, row_names=row_names, ztol=ztol,
                mmin=mmin, mmax=mmax, nbins=nbins, cmaps=cmaps, clo=clo, chi=chi,
                panel_size=panel_size, map_default=map_default, default_label=default_label,
                plot_ratio=plot_ratio, show_redshift_in_legend=show_redshift_in_legend,
                ylim=ylim, _show=False)    # not shown yet: the limits below must be set first
            axes_z[0, 0].set_xlim(xlim)   # x is shared down the rows
            for row in range(nrows):
                axes_z[row, 0].set_ylim(ylims[row])
            fig_z.savefig('../Images/compare_multiplicity_functions_' + str_identifier
                          + f'_z{z_req:.2f}' + _ratio_suffix(plot_ratio) + '.png',
                          bbox_inches='tight', transparent=True)
    if _show:
        plt.show()
    return fig, axes


def compare_multiplicity_evolution(maplog_tuples, labels=None, row_names=None,
                                   mult_thresh=None,
                                   cmaps=('Blues_r', 'Greens_r', 'Purples_r', 'Oranges_r'),
                                   clo=0., chi=0.7, panel_size=(5, 2.4),
                                   save=False, str_identifier='',
                                   map_default=None, default_label='',
                                   plot_ratio=False,
                                   show_redshift_in_legend=True):
    """Compare the evolution of the total MAP mass across maplogs.

    Companion to ``compare_multiplicity_functions``: instead of the multiplicity
    distribution at chosen redshifts, it sums the multiplicities of all MAPs
    (their total particle count) at *every* timeslice present in each maplog --
    so no redshift input is needed -- and plots that total against redshift as a
    fraction of ``NP/100`` (``NP`` from the maplog header), shown as a
    percentage. Layout follows ``compare_map_fractions``: a column of subplots,
    one per tuple of maplogs, sharing both axes, each tuple in shades of a
    single-color colormap from ``cmaps`` (sampled on ``[clo, chi]``), with an
    optional gray dashed ``map_default`` reference in every subplot and one
    y-label for the whole figure.

    Parameters
    ----------
    maplog_tuples : sequence of tuple of dict
        One tuple per subplot; each is a set of maplogs (outputs of
        ``read_maplog_dir``; only ``'mult'``, ``'control'``, ``'step'``,
        ``'redshift'`` and the header's ``'NP'`` are used) to overplot.
    labels : sequence of tuple of str, optional
        Same shape as ``maplog_tuples``; legend label of each curve.
    row_names : sequence of str, optional
        One entry per subplot, drawn as a title-style tag in its upper left.
    mult_thresh : float, optional
        Only MAPs with ``mult >= mult_thresh`` at a given redshift are included
        in that redshift's sum. Default None: every MAP counts.
    cmaps, clo, chi, panel_size, save, str_identifier :
        As in ``compare_map_fractions`` (saved to
        ``Images/compare_multiplicity_evolution[_<str_identifier>].png``).
    map_default : dict, optional
        A single maplog drawn as a gray dashed reference in every subplot (and the
        denominator when ``plot_ratio``).
    default_label : str
        Legend label for the ``map_default`` curve.
    plot_ratio : bool
        If True, plot each curve divided by ``map_default``'s curve (the default is
        linearly interpolated in redshift onto each curve's own timeslices, which
        can differ slightly between runs), with the default drawn as a flat line
        at 1 and dotted lines at 0.9/1.1. Requires ``map_default``.
    show_redshift_in_legend : bool
        Accepted for signature parity with the other ``compare_`` functions; the
        redshift is the x-axis here, so it has no effect.

    Returns
    -------
    (fig, axes) : the Figure and the 1D ndarray of Axes.
    """
    nrows = len(maplog_tuples)
    if plot_ratio:
        assert map_default is not None, "plot_ratio requires map_default"
    if labels is not None:
        assert len(labels) == nrows, "labels must have one tuple per maplog tuple (row)"
        for row, (mls, labs) in enumerate(zip(maplog_tuples, labels)):
            assert len(labs) == len(mls), \
                "labels[{}] must match the number of maplogs in row {}".format(row, row)
    if row_names is not None:
        assert len(row_names) == nrows, "row_names must have one entry per maplog tuple (row)"

    def _evolution(maplog):
        """(z, total MAP mass / (NP/100)) over every timeslice, sorted by z."""
        z, tot = [], []
        for step, zs in zip(maplog['step'], maplog['redshift']):
            mult = maplog['mult'][maplog['control'] == int(step) * 4 + 3]
            if mult_thresh is not None:
                mult = mult[mult >= mult_thresh]
            z.append(zs)
            tot.append(np.sum(mult, dtype=np.float64))
        z, tot = np.array(z, dtype=float), np.array(tot)
        order = np.argsort(z)
        return z[order], tot[order] / (maplog['header']['NP'] / 100)

    default = _evolution(map_default) if map_default is not None else None

    fig, axes = plt.subplots(nrows, 1, sharex=True, sharey=True, squeeze=False,
                             figsize=(panel_size[0], panel_size[1] * nrows))
    axes = axes[:, 0]
    fig.subplots_adjust(hspace=0)

    for row, maplogs in enumerate(maplog_tuples):
        ax    = axes[row]
        cmap  = colormaps[cmaps[row % len(cmaps)]]
        fracs = clo + (chi - clo) * np.linspace(0, 1, max(len(maplogs), 1))

        if default is not None:
            if plot_ratio:
                ax.axhline(1.0, ls='--', lw=2, color='gray', label=default_label)
                ax.axhline(0.9, ls=':', lw=2.5, color='gray', alpha=0.8)
                ax.axhline(1.1, ls=':', lw=2.5, color='gray', alpha=0.8)
            else:
                ax.plot(*default, '.--', lw=2, color='gray', label=default_label)

        for m, maplog in enumerate(maplogs):
            z, y = _evolution(maplog)
            if plot_ratio:
                y = y / np.interp(z, *default)
            label = str(labels[row][m]) if labels is not None else ''
            ax.plot(z, y, '.-', lw=2, color=cmap(fracs[m]), label=label)

        ax.grid()
        if row_names is not None:
            ax.text(0.02, 0.95, row_names[row], transform=ax.transAxes, va='top', fontsize='small')
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize='small')

    axes[-1].set(xscale='log', xlabel=r'redshift $[z]$')
    axes[-1].invert_xaxis()                      # high-z left, low-z right
    if plot_ratio:
        for ax in axes:
            ax.set_ylim(0.7, 1.3)
        ylabel = r'$\Sigma \mathcal{M}_{MAP}/\Sigma \mathcal{M}_{MAP,\rm ref}$'
    else:
        for ax in axes:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
        ylabel = r'$\Sigma \mathcal{M}_{MAP}\,/\,N_p$'

    # one label for the whole figure, placed just left of the widest tick labels
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    x_left = min(ax.yaxis.get_tightbbox(renderer).x0 for ax in axes)
    x_left = fig.transFigure.inverted().transform((x_left, 0))[0]
    fig.supylabel(ylabel, x=x_left - 0.01, ha='right')

    if save:
        ids = "_" + str_identifier if len(str_identifier) > 0 else ''
        fig.savefig('../Images/compare_multiplicity_evolution' + ids + _ratio_suffix(plot_ratio) + '.png',
                    bbox_inches='tight', transparent=True)
    plt.show()
    return fig, axes


try:
    from numba import njit as _njit
except ImportError:                       # pure-Python fallback (slow on real maplogs)
    _njit = lambda f: f


@_njit
def _next_smaller(C):
    """``out[i]`` = the first ``k > i`` with ``C[k] < C[i]`` (-1 if none), O(N) via a stack."""
    n = C.shape[0]
    out = np.full(n, -1, np.int64)
    stack = np.empty(n, np.int64)
    top = 0
    for k in range(n):
        while top > 0 and C[k] < C[stack[top - 1]]:
            out[stack[top - 1]] = k
            top -= 1
        stack[top] = k
        top += 1
    return out


def _merge_events_one_file(f, mult_thresh):
    """Every Merger node in the trees of one maplog file's above-threshold MAPs.

    Runs in a worker process (see ``compare_merge_vrel``). A file's node stream
    is a concatenation of whole trees, so a file is a self-contained unit. The
    "head" nodes are the final-timeslice TimeSlice nodes of MAPs with
    ``mult >= mult_thresh``; each heads a contiguous run of ``length`` nodes
    holding every merger that built that MAP, so the merger rows to keep are
    just the ones falling inside one of those runs (no walk needed for those).

    For each kept merger the two progenitor branches follow it in the stream --
    primary's tree first, then secondary's -- and the first node of each branch
    is the last time that progenitor was logged. A tree ends where the running
    count (+1 Merger, -1 Formation) first drops below its value at the tree's
    root, which ``_next_smaller`` finds for every node at once; that locates the
    first node of the secondary branch.
    """
    asdf.get_config().validate_on_read = False
    tbl = read_asdf(f, load=['vel', 'mult', 'control', 'node_type', 'timestep',
                             'mult_sec', 'density', 'vel_disp', 'length'])
    nt     = np.asarray(tbl['node_type'])
    ts     = np.asarray(tbl['timestep']).astype(np.int64)
    mult   = np.asarray(tbl['mult']).astype(np.int64)
    length = np.asarray(tbl['length']).astype(np.int64)
    vel    = np.asarray(tbl['vel'])
    step_to_z = np.asarray(tbl.meta['Step_to_Z'])

    final = ts[nt == 3].max()                       # the last timeslice
    heads = np.flatnonzero((nt == 3) & (ts == final) & (mult >= mult_thresh))
    mi    = np.flatnonzero(nt == 1)
    h     = np.searchsorted(heads, mi, side='right') - 1
    keep  = h >= 0
    keep[keep] = mi[keep] < heads[h[keep]] + length[heads[h[keep]]]
    mi, h = mi[keep], h[keep]
    head_nmerge = np.bincount(h, minlength=len(heads)).astype(np.int32)   # mergers in each MAP's tree

    # Every TimeSlice node in those trees (the heads included): a non-Formation node's own
    # previous state is the next node in the stream, which gives one accretion interval
    # (previous node -> this timeslice) per node, with or without a merger in between.
    ei = np.flatnonzero(nt == 3)
    hh = np.searchsorted(heads, ei, side='right') - 1
    ke = hh >= 0
    ke[ke] = ei[ke] < heads[hh[ke]] + length[heads[hh[ke]]]
    ei, hh = ei[ke], hh[ke]
    eprev = ei + 1

    C   = np.cumsum((nt == 1).astype(np.int32) - (nt == 0).astype(np.int32))
    nse = _next_smaller(C)
    prim = mi + 1                                   # first node of the primary's branch
    end1 = nse[mi]                                  # last node of the primary's branch
    sec  = end1 + 1                                 # first node of the secondary's branch
    assert np.all(end1 >= 0), "unterminated tree in {}".format(f)

    # The Formation node on a branch's own dominant (primary-most) line is the first Formation
    # at or after the branch's head (``dominant_formation_pid``'s reverse running-min), which
    # dates when that progenitor MAP itself formed.
    N = nt.shape[0]
    next_form = np.minimum.accumulate(np.where(nt == 0, np.arange(N), N)[::-1])[::-1]
    assert np.all(next_form[prim] < N) and np.all(next_form[sec] < N), "branch without a Formation in {}".format(f)

    # sanity check: where the primary branch starts at an epoch node, its stored
    # length must agree with the walked branch
    ep = nt[prim] == 3
    n_bad = int(np.sum(end1[ep] != prim[ep] + length[prim[ep]] - 1))
    if n_bad:
        print("WARNING: {} merger(s) in {} whose walked primary branch disagrees with "
              "the epoch node's stored length".format(n_bad, f))

    return {
        'mult_sec':      np.asarray(tbl['mult_sec'])[mi].astype(np.int32),
        'mult_primary':  (mult[mi] - np.asarray(tbl['mult_sec'])[mi]).astype(np.int32),
        'mult_merged':   mult[mi].astype(np.int32),
        'head_mult':     mult[heads[h]].astype(np.int32),
        'vrel':          np.linalg.norm(vel[prim] - vel[sec], axis=1).astype(np.float32),
        'vel_disp':      np.asarray(tbl['vel_disp'])[mi].astype(np.float32),
        'density':       np.asarray(tbl['density'])[mi].astype(np.float32),
        'redshift':      step_to_z[ts[mi] - 1].astype(np.float32),
        'step':          ts[mi].astype(np.int32),
        'v_merged':      vel[mi].astype(np.float32),
        'v_primary':     vel[prim].astype(np.float32),
        'v_secondary':   vel[sec].astype(np.float32),
        'z_form_primary':   step_to_z[ts[next_form[prim]] - 1].astype(np.float32),
        'z_form_secondary': step_to_z[ts[next_form[sec]] - 1].astype(np.float32),
        'age_primary':   (ts[mi] - ts[prim]).astype(np.int32),
        'age_secondary': (ts[mi] - ts[sec]).astype(np.int32),
        # multiplicity / redshift of each progenitor's last logged node, for accretion rates
        'mult_prev_primary':   mult[prim].astype(np.int32),
        'mult_prev_secondary': mult[sec].astype(np.int32),
        'z_prev_primary':      step_to_z[ts[prim] - 1].astype(np.float32),
        'z_prev_secondary':    step_to_z[ts[sec] - 1].astype(np.float32),
        'head_nmerge':         head_nmerge[h],
        # one entry per TimeSlice node in the trees (``ep_`` = epoch): multiplicity / redshift
        # of the node and of the same MAP's previous logged node, and the MAP's final
        # multiplicity and total number of mergers
        'ep_mult':        mult[ei].astype(np.int32),
        'ep_mult_prev':   mult[eprev].astype(np.int32),
        'ep_z':           step_to_z[ts[ei] - 1].astype(np.float32),
        'ep_z_prev':      step_to_z[ts[eprev] - 1].astype(np.float32),
        'ep_prev_type':   nt[eprev].astype(np.uint8),
        'ep_vel_disp':    np.asarray(tbl['vel_disp'])[ei].astype(np.float32),
        'ep_density':     np.asarray(tbl['density'])[ei].astype(np.float32),
        'ep_head_mult':   mult[heads[hh]].astype(np.int32),
        'ep_head_nmerge': head_nmerge[hh],
        'cosmology': {k: float(tbl.meta[k]) for k in
                      ('H0', 'Omega_M', 'Omega_K', 'Omega_DE', 'w0', 'wa')},
    }


def read_merge_events(rdir, mult_thresh=100, nworkers=16, executor=None):
    """Merger events in the histories of a run's above-threshold MAPs.

    Parses every ``maplogs/*/maplog_z*.asdf`` file of ``rdir`` once (one task per
    file, in parallel) and returns a dict of per-event arrays, one entry per
    Merger node found in the log tree of any final-timeslice MAP with
    ``mult >= mult_thresh``:

    ``mult_sec`` / ``mult_primary`` / ``mult_merged``
        multiplicity of the secondary, of the primary, and of the merged MAP
    ``head_mult``
        final multiplicity of the MAP whose history contains the event
    ``redshift``, ``step``
        when the merger happened
    ``z_form_primary`` / ``z_form_secondary``
        redshift at which each progenitor MAP formed: the Formation node on its own dominant
        (primary-most) branch, i.e. the first Formation at or after its branch head
    ``vel_disp``, ``density``
        environment values stored on the Merger node (``density`` is the matter density in
        units of the cosmic mean, rho/rho_bar, smoothed on ``Map_DensityKernelRadius_cMpc``). These are the *primary's*
        (``tracers.cpp`` copies them from the larger MAP), a good proxy for the
        secondary's since the two collide in the same group
    ``vrel``
        |v_primary - v_secondary| (km/s) from the last logged node of each
        branch (``v_primary``, ``v_secondary``). **An approximation**: the
        merger node does not record the collision's relative velocity, and
        those nodes can predate the merger by ``age_primary`` /
        ``age_secondary`` steps (up to a timeslice interval)
    ``v_merged``
        the merger node's mass-weighted COM velocity (km/s)
    ``mult_prev_primary`` / ``mult_prev_secondary``, ``z_prev_primary`` / ``z_prev_secondary``
        multiplicity and redshift of each progenitor's last logged node (see
        ``compare_accretion_rate``)
    ``head_nmerge``
        total number of mergers in the history of the MAP the event belongs to (0 for
        a MAP that never merged)
    ``ep_mult``, ``ep_mult_prev``, ``ep_z``, ``ep_z_prev``, ``ep_prev_type``, ``ep_vel_disp``, ``ep_density``, ``ep_head_mult``, ``ep_head_nmerge``, ``ep_file``
        a second, longer list with one entry per TimeSlice node in those trees (the
        final-timeslice heads and every earlier timeslice of every progenitor): its
        multiplicity and redshift, the same MAP's previous logged node's multiplicity
        and redshift (``ep_prev_type``: 0 Formation, 1 Merger, 3 TimeSlice), the
        environment velocity dispersion and density at the node (``ep_vel_disp`` / ``ep_density``, as ``vel_disp`` / ``density`` for mergers), and the
        host MAP's final multiplicity and merger count. Merger events and epoch
        events together cover every interval between consecutive logged states
    ``cosmology``
        dict (``H0``, ``Omega_M``, ``Omega_K``, ``Omega_DE``, ``w0``, ``wa``) from the maplog header
    ``file``
        index of the source file, in sorted order
    """
    files = sorted(glob(str(Path(rdir) / "maplogs/*/maplog_z*_*.asdf")))
    if not files:
        raise FileNotFoundError("no maplog files under {}".format(rdir))
    worker = partial(_merge_events_one_file, mult_thresh=mult_thresh)
    if executor is None:
        with ProcessPoolExecutor(max_workers=nworkers) as ex:
            results = list(ex.map(worker, files))
    else:
        results = list(executor.map(worker, files))
    cosmology = results[0]['cosmology']
    out = {k: np.concatenate([r[k] for r in results]) for k in results[0] if k != 'cosmology'}
    out['cosmology'] = cosmology
    out['file'] = np.concatenate([np.full(len(r['redshift']), i, np.int32)
                                  for i, r in enumerate(results)])
    out['ep_file'] = np.concatenate([np.full(len(r['ep_z']), i, np.int32)
                                     for i, r in enumerate(results)])
    return out


def _merge_event_scatter(dir_tuples, quantities, labels, mult_thresh, nworkers, data,
                         cmap_range, s, alpha, max_points, seed, panel_size, save,
                         fname, str_identifier, xscale, yscale, xlabel, ylabel, clabel):
    """Shared engine of ``compare_merge_vrel`` / ``compare_accretion_rate``.

    Parses each run's merger events (``read_merge_events``) unless ``data`` is
    given, then draws one scatter per run: two columns (lo | hi, from tuples
    ordered ``(hi, lo)``), one row per tuple, coloured by the upper half of
    ``hsv``. ``quantities(events)`` returns ``(x, y, c)`` arrays for the points
    to plot, and may add derived keys to the event dict (kept in the returned data).
    """
    from matplotlib.colors import LinearSegmentedColormap
    nrows = len(dir_tuples)
    if labels is not None:
        assert len(labels) == nrows, "labels must have one tuple per dir tuple (row)"
        for row, (dirs, labs) in enumerate(zip(dir_tuples, labels)):
            assert len(labs) == len(dirs) == 2, \
                "each row needs exactly (hi, lo) directories and labels (row {})".format(row)

    if data is None:
        with ProcessPoolExecutor(max_workers=nworkers) as ex:   # one pool for every run
            data = [tuple(read_merge_events(d, mult_thresh, executor=ex) for d in dirs)
                    for dirs in dir_tuples]

    xyc = [[quantities(ev) for ev in row] for row in data]
    cmap = LinearSegmentedColormap.from_list('hsv_upper', colormaps['hsv'](np.linspace(*cmap_range, 256)))
    call = np.concatenate([c for row in xyc for (_, _, c) in row])
    norm = Normalize(call.min(), call.max())
    rng  = np.random.default_rng(seed)

    fig, axes = plt.subplots(nrows, 2, sharex=True, sharey=True, squeeze=False,
                             figsize=(panel_size[0] * 2, panel_size[1] * nrows))
    fig.subplots_adjust(wspace=0, hspace=0)
    for row in range(nrows):
        for col, i in enumerate((1, 0)):             # left = lo (index 1), right = hi (index 0)
            x, y, c = xyc[row][i]
            ax = axes[row, col]
            idx = rng.choice(len(x), max_points, replace=False) if len(x) > max_points else slice(None)
            ax.scatter(x[idx], y[idx], c=c[idx], cmap=cmap, norm=norm, s=s, alpha=alpha,
                       linewidths=0, rasterized=True)
            ax.set(xscale=xscale, yscale=yscale)
            ax.grid()
            if labels is not None:
                ax.text(0.04, 0.05, str(labels[row][i]), transform=ax.transAxes,
                        va='bottom', fontsize='small',
                        bbox=dict(boxstyle='round', fc='white', ec='none', alpha=0.8))
    for ax in axes[-1, :]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, pad=0.02)
    cb.set_label(clabel)

    if save:
        ids = "_" + str_identifier if len(str_identifier) > 0 else ''
        fig.savefig('../Images/' + fname + ids + '.png', bbox_inches='tight', dpi=200)
    plt.show()
    return fig, axes, data


def compare_merge_vrel(dir_tuples, labels=None, mult_thresh=100, nworkers=16,
                       data=None, cmap_range=(0.5, 1.0), s=2, alpha=0.5,
                       max_points=200000, seed=0, panel_size=(5, 4),
                       save=False, str_identifier='', xscale='log', color_key='merge_redshift'):
    """Merger mass ratio vs environment velocity dispersion, coloured by redshift.

    For every run, walks the merger history of each final-timeslice MAP with
    ``mult >= mult_thresh`` (see ``read_merge_events``) and scatters every merger
    event: x = the DM velocity dispersion of the environment recorded on the
    Merger node (``vel_disp``; the primary's value, which is also the
    secondary's to good approximation since the two collide in the same group),
    y = the ratio of the secondary's to the primary's multiplicity (<= 1), colour = the redshift chosen by ``color_key`` (by default that of the merger event), with the upper half
    of matplotlib's ``hsv`` (``cmap_range=(0.5, 1.0)``: cyan -> blue -> violet ->
    magenta -> red) so that higher redshift is redder. One row per tuple; the
    left column is the lower value of the varied parameter and the right column
    the higher one. There is no default/reference run. The returned data also
    holds the other per-event quantities (including the approximate
    reconstructed relative velocity ``vrel``) for further plots.

    Parameters
    ----------
    dir_tuples : sequence of tuple of (str or Path)
        One tuple per row, ordered ``(hi, lo)`` like the other ``compare_``
        functions (e.g. ``mapvar_dirs``); drawn lo on the left, hi on the right.
    labels : sequence of tuple of str, optional
        Same shape/order as ``dir_tuples``; text drawn in each panel's corner.
    mult_thresh : float
        Only MAPs with at least this final multiplicity have their histories walked.
    nworkers : int
        Worker processes for parsing (one task per maplog file).
    data : list of tuple of dict, optional
        Previously returned ``data``; skips the (slow) parse and just re-plots.
    cmap_range : (float, float)
        Portion of ``hsv`` used; low redshift maps to the first value, high to the second.
    s, alpha :
        Scatter marker size / opacity.
    max_points : int
        Plot at most this many randomly chosen events per panel (all are kept in
        the returned data).
    seed : int
        RNG seed for that subsampling.
    panel_size, save, str_identifier :
        As in the other ``compare_`` functions (saved to
        ``Images/compare_merge_vrel_<color_key>[_<str_identifier>].png``, so figures
        coloured by different quantities don't overwrite each other).
    color_key : {'merge_redshift', 'z_form_secondary', 'z_form_primary'}
        What the colour shows: the redshift of the merger itself (default), or the redshift at
        which the secondary / primary progenitor MAP formed (the Formation node on that MAP's own
        dominant branch; see ``read_merge_events``).
    xscale : {'log', 'linear'}
        Scale of the velocity-dispersion axis. Log by default: the values span
        a decade or more with a long high tail, and the stored 8-bit float has
        fixed fractional resolution.

    Returns
    -------
    (fig, axes, data) : the Figure, the 2D ndarray of Axes (rows x [lo, hi]) and
        ``data[row] = (events_hi, events_lo)`` aligned with ``dir_tuples``, each an
        event dict from ``read_merge_events``.
    """
    color_fields = {'merge_redshift':   ('redshift',         'redshift'),
                    'z_form_secondary': ('z_form_secondary', 'formation redshift of the secondary'),
                    'z_form_primary':   ('z_form_primary',   'formation redshift of the primary')}
    if color_key not in color_fields:
        raise ValueError("color_key must be one of {}, got {!r}".format(sorted(color_fields), color_key))
    cfield, clabel = color_fields[color_key]
    if data is not None and not all(cfield in ev for row in data for ev in row):
        print("compare_merge_vrel: `data` predates the {} field; re-reading the maplogs".format(cfield))
        data = None
    quantities = lambda ev: (ev['vel_disp'], ev['mult_sec'] / ev['mult_primary'], ev[cfield])
    return _merge_event_scatter(
        dir_tuples, quantities, labels, mult_thresh, nworkers, data, cmap_range, s, alpha,
        max_points, seed, panel_size, save, f'compare_merge_vrel_{color_key}', str_identifier, xscale, 'log',
        r'environment $\sigma_v$ [km/s]', r'$\mathfrak{M}_{sec}\,/\,\mathfrak{M}_{prim}$', clabel)


def _cosmic_time_curve(cosmology, ngrid=4000):
    """Cosmic time ``t(z)`` [Gyr] from ``dt = dz / ((1+z) H(z))``.

    Integrates on a uniform grid in ``ln(1+z)`` (where ``dt = du / H``) from z = 99
    down to z = 0, with ``H(z)^2 = H0^2 [Om (1+z)^3 + Ok (1+z)^2 + Ode (1+z)^{3(1+w0+wa)}
    exp(-3 wa z/(1+z))]`` from the maplog header cosmology (``Omega_M`` already
    includes the smooth component). ``t`` is measured from z = 99, which cancels in
    the differences used here. Returns ``(z_grid, t_grid)``, both ascending in z.
    """
    c = cosmology
    u_grid = np.linspace(0.0, np.log(100.0), ngrid)
    one_z = np.exp(u_grid)
    E2 = (c['Omega_M'] * one_z**3 + c['Omega_K'] * one_z**2
          + c['Omega_DE'] * one_z**(3 * (1 + c['w0'] + c['wa'])) * np.exp(-3 * c['wa'] * (one_z - 1) / one_z))
    inv_H = 1.0 / (c['H0'] * np.sqrt(E2))                       # Mpc s / km
    seg = 0.5 * (inv_H[1:] + inv_H[:-1]) * np.diff(u_grid)       # trapezoid, per grid cell
    t = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])      # t(z) = int_z^99 du / H
    return one_z - 1.0, t * (1.0 * u.Mpc / (u.km / u.s)).to(u.Gyr).value


def compare_accretion_rate(dir_tuples, labels=None, mult_thresh=100, nworkers=16,
                           data=None, cmap_range=(0.5, 1.0), s=2, alpha=0.5,
                           max_points=200000, seed=0, panel_size=(5, 4),
                           save=False, str_identifier='', xscale='log', yscale='log',
                           midpoint='time', include_secondary=True, include_epochs=True,
                           yaxis_key='vel_disp'):
    """Rough MAP accretion rate vs an environment property, coloured by redshift.

    Same layout and machinery as ``compare_merge_vrel`` (final-timeslice MAPs with
    ``mult >= mult_thresh``, every Merger node in their histories, two columns
    lo | hi, upper half of ``hsv`` for the colour). Every Merger node in a tree
    is its own event, wherever it sits in the tree, and each joins two MAPs
    (the larger "primary" and the smaller "secondary"), each of which has its own
    branch following the node. The first node of a branch (a TimeSlice, an earlier
    Merger, or the Formation) is that progenitor's last logged state: multiplicity
    ``M0`` at redshift ``z0``. Just before the merger the primary has multiplicity
    ``Mf = mult_merged - mult_sec`` and the secondary ``Mf = mult_sec``. Mergers are
    logged but accretion is not, so ``Mf - M0`` is what that progenitor accreted in
    between, and each progenitor gives one observation:

        rate = (Mf - M0) / (t(zf) - t(z0)),      t(z) from dt = dz / ((1+z) H(z))

    with ``zf`` the merger's redshift (``t(z)`` and ``H(z)`` use the cosmology in the
    maplog header). x = that rate [multiplicity / Gyr], y = the maplog field chosen by
    ``yaxis_key``, taken at the later node (the Merger node's value, which is the primary's
    and stands in for the secondary's, or the TimeSlice node's), colour = the redshift at
    the midpoint between the two observations. Events with no accretion (rate <= 0) or zero elapsed time are
    not plotted (log axis); they stay in the returned data.

    Parameters
    ----------
    yaxis_key : {'vel_disp', 'density', 'mult'}
        Quantity on the y-axis: ``'vel_disp'`` the DM velocity dispersion of the environment
        [km/s]; ``'density'`` the environment matter density in units of
        the cosmic mean, rho/rho_bar (8-bit float: integer resolution below 16, then ~3%);
        ``'mult'`` the MAP's multiplicity at the later time.
    include_epochs : bool
        If True (default), also include the accretion between a MAP's previous
        logged node and each of its TimeSlice nodes (every timeslice of every
        progenitor in the trees), which exist for MAPs that never merge as well.
        Together with the merger intervals this covers every interval between
        consecutive logged states of every MAP. ``head_nmerge`` /
        ``ep_head_nmerge`` in the returned data give the host MAP's merger count,
        so merged and never-merged populations can be separated.
    include_secondary : bool
        If True (default), each merger contributes both progenitors' accretion
        observations (primary and secondary branch); if False, only the primary's.
    midpoint : {'time', 'redshift'}
        How to place the event between the two observations: at the midpoint in cosmic
        time (default; the redshift at ``(t0 + tf)/2``) or at ``(z0 + zf)/2``.
    yscale : {'log', 'linear'}
        Scale of the y-axis.
    Everything else is as in ``compare_merge_vrel`` (``xscale`` here scales the rate axis).

    Returns
    -------
    (fig, axes, data) : as in ``compare_merge_vrel``, with the event dicts also gaining
        the primary's ``mult_later`` (``Mf``), ``dmult_acc`` (``Mf - M0``), ``dt_acc`` [Gyr],
        ``acc_rate`` [1/Gyr] and ``z_mid``, and the same for the secondary with a
        ``_sec`` suffix, and for the timeslice intervals with an ``_ep`` suffix.
    """
    assert midpoint in ('time', 'redshift')
    ylabels = {'vel_disp': r'environment $\sigma_v$ [km/s]',
               'density':  r'environment $\rho/\bar{\rho}$',
               'mult':     r'$\mathfrak{M}$ (later time)'}
    if yaxis_key not in ylabels:
        raise ValueError("yaxis_key must be one of {}, got {!r}".format(sorted(ylabels), yaxis_key))
    needed = ('mult_prev_primary', 'z_prev_primary', 'mult_prev_secondary', 'z_prev_secondary',
              'cosmology', 'ep_mult', 'head_nmerge') + (() if yaxis_key == 'mult' else ('ep_' + yaxis_key,))
    if data is not None and not all(k in ev for row in data for ev in row for k in needed):
        print("compare_accretion_rate: `data` predates the accretion fields; re-reading the maplogs")
        data = None

    def quantities(ev):
        zg, tg = _cosmic_time_curve(ev['cosmology'])

        def interval(mf, mult_prev, z_later, z_prev):
            """multiplicity at the later time, its change, elapsed time, rate and midpoint redshift for each interval"""
            t0 = np.interp(z_prev, zg, tg)
            tf = np.interp(z_later, zg, tg)
            dt = tf - t0                                              # t(zf) - t(z0) > 0
            mf = mf.astype(float)
            dm = mf - mult_prev
            with np.errstate(divide='ignore', invalid='ignore'):
                rate = np.where(dt > 0, dm / dt, np.nan)
            if midpoint == 'time':
                z_mid = np.interp(0.5 * (t0 + tf), tg[::-1], zg[::-1])  # invert t(z)
            else:
                z_mid = 0.5 * (z_prev + z_later)
            return mf, dm, dt, rate, z_mid

        yfield = None if yaxis_key == 'mult' else yaxis_key          # None: use the later multiplicity
        sets = [('',     ev['mult_primary'], ev['mult_prev_primary'],   ev['redshift'], ev['z_prev_primary'],
                 ev.get(yfield))]
        if include_secondary:
            sets.append(('_sec', ev['mult_sec'], ev['mult_prev_secondary'], ev['redshift'], ev['z_prev_secondary'],
                         ev.get(yfield)))
        if include_epochs:
            sets.append(('_ep', ev['ep_mult'], ev['ep_mult_prev'], ev['ep_z'], ev['ep_z_prev'],
                         ev.get('ep_' + yfield) if yfield else None))
        out = []
        for suffix, *args, yvals in sets:
            mf, dm, dt, rate, z_mid = interval(*args)
            yvals = mf if yvals is None else yvals
            ev.update({'mult_later' + suffix: mf, 'dmult_acc' + suffix: dm, 'dt_acc' + suffix: dt,
                       'acc_rate' + suffix: rate, 'z_mid' + suffix: z_mid})
            ok = rate > 0                                             # also drops NaN
            out.append((rate[ok], yvals[ok], z_mid[ok]))
        return tuple(np.concatenate(a) for a in zip(*out))

    return _merge_event_scatter(
        dir_tuples, quantities, labels, mult_thresh, nworkers, data, cmap_range, s, alpha,
        max_points, seed, panel_size, save, 'compare_accretion_rate', str_identifier, xscale, yscale,
        r'$\Delta\mathfrak{M}/\Delta t$ [Gyr$^{-1}$]', ylabels[yaxis_key],
        'redshift (midpoint)')


@dataclass
class MapHistoryProperty:
    """One scalar merger-history property for every final-timeslice MAP of one run.

    ``values[i]`` belongs to the i-th final-timeslice MAP, in the same order as
    the rows of ``maplog`` (from ``read_maplog_dir``) with ``control == 4*step + 3``.
    That alignment is checked (``mult``) when the property is used, so a stale
    property or one built from a different run is caught rather than silently
    mis-assigning MAPs to subsamples.

    Built by ``read_map_history_property`` -- which parses the full merger trees,
    something the head-node-only ``read_maplog_dir`` dict cannot do -- and meant
    to be built once, pickled (``cache_dir``) and re-used for any number of
    subsample splits, since the tree walk is the expensive part and the split
    itself is just a comparison on ``values``.

    Attributes
    ----------
    name : str       what ``values`` is (axis/legend text).
    values : (N,) float array, or a dict ``{field: (N,) float array}`` for a
        property with several components (e.g. a ratio *and* the redshift of the
        event). NaN where undefined for a MAP (e.g. a min-merger-ratio for a MAP
        that never merged). Whatever it is, it is what a subsample selector
        receives.
    step, redshift : the final timeslice the MAPs belong to.
    mult : (N,) int32 multiplicity of each MAP at ``step`` (alignment check).
    pos : (N,3) float32 position of each MAP at ``step`` (alignment check).
    source : the simulation directory it was built from.
    """
    name: str
    values: np.ndarray
    step: int
    redshift: float
    mult: np.ndarray
    pos: np.ndarray
    source: str = ''

    def __len__(self):
        v = self.values
        return len(next(iter(v.values()))) if isinstance(v, dict) else len(v)


def min_merger_ratio_primary(h):
    """Smallest secondary/primary multiplicity ratio over a MAP's primary-branch mergers.

    A history extractor for ``read_map_history_property``. The "primary branch"
    of a final MAP is its dominant lineage: the run of nodes from its head to the
    first Formation node (see ``dominant_formation_pid``), which contains every
    Merger node where the lineage absorbed a smaller MAP (mergers *within* those
    absorbed MAPs' own histories are not counted). At a Merger node ``mult`` is
    the merged multiplicity and ``mult_sec`` the secondary's, so the ratio is
    ``mult_sec / (mult - mult_sec)`` (<= 1). Returns the minimum over the run, or
    NaN for a MAP that never merged.
    """
    nt, mult, msec = h['nt'], h['mult'], h['mult_sec']
    ratio = np.full(nt.shape[0] + 1, np.inf)   # +1 pad so a run ending at the last node is valid
    mi = nt == 1
    ratio[:-1][mi] = msec[mi] / np.maximum(mult[mi] - msec[mi], 1)
    out = h['reduce_dominant_run'](np.minimum, ratio)
    return np.where(np.isfinite(out), out, np.nan)


def max_merger_fraction_primary(h):
    """Largest primary-branch merger, as a fraction of the MAP's final multiplicity.

    A history extractor for ``read_map_history_property``. Over the Merger nodes
    of a final MAP's primary branch (see ``min_merger_ratio_primary``) it returns
    the maximum of ``mult_sec / mult_final``: the secondary's multiplicity at the
    merger divided by the MAP's multiplicity at the final timeslice. 0 for a MAP
    that never merged (no NaN), so such MAPs fall in the lowest bin.
    """
    nt, msec = h['nt'], h['mult_sec']
    frac = np.zeros(nt.shape[0] + 1)                       # +1 pad so a run ending at the last node is valid
    k  = np.searchsorted(h['heads'], np.arange(nt.shape[0]), side='right') - 1
    mi = nt == 1
    frac[:-1][mi] = msec[mi] / h['mult'][h['heads'][np.maximum(k[mi], 0)]]   # the head owning each merger
    return h['reduce_dominant_run'](np.maximum, frac)


def min_merger_ratio_and_redshift_primary(h):
    """``min_merger_ratio_primary`` plus when that smallest-ratio merger, and its secondary, happened.

    Returns ``{'min_ratio', 'z_min_ratio', 'z_form_secondary'}``, all NaN for a MAP
    with no primary-branch merger: the smallest ratio, the redshift of that
    merger, and the redshift at which the merger's secondary progenitor formed
    (the Formation node on the secondary's own dominant line, as in
    ``_merge_events_one_file``). If several mergers tie for the smallest ratio,
    the most recent one (the one nearest the head) is used.

    Locating the secondary's branch needs the tree-end search of
    ``_next_smaller`` over the whole node stream (a Python loop), so this is
    slower per file than ``min_merger_ratio_primary``.
    """
    nt, mult, msec, ts = h['nt'], h['mult'], h['mult_sec'], h['ts']
    heads, ends = h['heads'], h['ends']
    mi = np.flatnonzero(nt == 1)
    k  = np.searchsorted(heads, mi, side='right') - 1     # candidate head of each merger
    ok = k >= 0
    ok[ok] = mi[ok] < ends[k[ok]]                          # inside that head's dominant run
    mi, k = mi[ok], k[ok]
    ratio = msec[mi] / np.maximum(mult[mi] - msec[mi], 1)
    order = np.lexsort((mi, ratio, k))                     # by head, then ratio, then recency
    k_s = k[order]
    first = order[np.unique(k_s, return_index=True)[1]]    # smallest-ratio merger per head
    min_ratio = np.full(len(heads), np.nan)
    z_min     = np.full(len(heads), np.nan)
    min_ratio[k[first]] = ratio[first]
    z_min[k[first]]     = h['step_to_z'][ts[mi[first]] - 1]

    # secondary's branch starts right after the primary's ends, where the running
    # count (+1 Merger, -1 Formation) first drops below its value at the merger
    C   = np.cumsum((nt == 1).astype(np.int32) - (nt == 0).astype(np.int32))
    end = _next_smaller(C)[mi[first]]
    assert np.all(end >= 0), "unterminated tree"
    z_form_sec = np.full(len(heads), np.nan)
    z_form_sec[k[first]] = h['step_to_z'][ts[h['next_form'][end + 1]] - 1]
    return {'min_ratio': min_ratio, 'z_min_ratio': z_min, 'z_form_secondary': z_form_sec}


def formation_redshift_primary(h):
    """Redshift at which each final MAP formed: its main-branch Formation node.

    A history extractor for ``read_map_history_property``. The main (dominant)
    branch of a final MAP runs from its head to the first Formation node (see
    ``min_merger_ratio_primary``); the timestep of that node, converted to
    redshift, is the MAP's formation time.
    """
    return h['step_to_z'][h['ts'][h['next_form'][h['heads']]] - 1]


def half_mass_redshift_primary(h):
    """Redshift at which each final MAP's main branch reached half its final multiplicity.

    A history extractor for ``read_map_history_property``. Walks the main (dominant)
    branch from the final head back in time (see ``min_merger_ratio_primary``) to
    the first node whose multiplicity is below half the final one; the
    half-mass epoch lies between that node and the next newer one. The branch is
    only sampled at its logged nodes (Formation, Merger and time-slice/lightcone
    epochs), so for gradual growth the redshift is interpolated linearly in scale
    factor between the two nodes; if the crossing is at a Merger node it is a
    jump, and the Merger's own redshift is used. A MAP already at half its final
    multiplicity when it formed gets its formation redshift.
    """
    nt, mult, ts = h['nt'], h['mult'], h['ts']
    heads, ends = h['heads'], h['ends']
    a_node = 1.0 / (1.0 + h['step_to_z'])                          # scale factor of each step
    node = np.arange(nt.shape[0])
    k    = np.searchsorted(heads, node, side='right') - 1          # head whose run could hold each node
    inrun = k >= 0
    inrun[inrun] = node[inrun] < ends[k[inrun]]
    half  = mult[heads] / 2.0
    below = inrun & (mult < half[np.maximum(k, 0)])
    jk, first = np.unique(k[below], return_index=True)            # heads that dip below half, and
    j   = np.flatnonzero(below)[first]                             # their first (newest) such node
    out = a_node[ts[ends - 1] - 1]                                 # default: formation epoch
    hi, lo = j - 1, j                                              # newest node >= half, next older < half
    a_hi, a_lo = a_node[ts[hi] - 1], a_node[ts[lo] - 1]
    frac = (half[jk] - mult[lo]) / (mult[hi] - mult[lo])
    out[jk] = np.where(nt[hi] == 1, a_hi, a_lo + frac * (a_hi - a_lo))
    return 1.0 / out - 1.0


def merger_rate_primary(h):
    """Merger rate [1/Gyr] of each final MAP: the Poisson estimate over its main-branch mergers.

    A history extractor for ``read_map_history_property``: with N Merger nodes on the main branch at cosmic
    times t_1 <= ... <= t_N, ``rate = (N - 1) / (t_N - t_1)`` (the same estimate as ``read_merger_rates``).
    NaN for N < 2 or when all mergers fall in one timestep, so selectors must decide where such MAPs go.
    """
    nt, ts = h['nt'], h['ts']
    is_m = nt == 1
    t_node = h['t_step'][ts - 1]                      # a node's timestep is one ahead of its redshift
    tmin = np.full(nt.shape[0] + 1, np.inf)           # +1 pad so a run ending at the last node is valid
    tmax = np.full(nt.shape[0] + 1, -np.inf)
    cnt = np.zeros(nt.shape[0] + 1)
    tmin[:-1][is_m] = t_node[is_m]
    tmax[:-1][is_m] = t_node[is_m]
    cnt[:-1][is_m] = 1.0
    n = h['reduce_dominant_run'](np.add, cnt)
    span = h['reduce_dominant_run'](np.maximum, tmax) - h['reduce_dominant_run'](np.minimum, tmin)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where((n >= 2) & (span > 0), (n - 1) / span, np.nan)


def _history_one_file(f, extractor):
    """Read one maplog file and evaluate ``extractor`` on its final-timeslice MAPs.

    Runs in a worker process (module-level so it pickles). ``extractor(h)``
    receives a dict ``h`` of whole-file node-stream arrays -- ``nt`` (node
    type), ``ts`` (timestep), ``mult``, ``mult_sec``, ``pos`` -- plus
    ``heads`` (indices of the final-timeslice head nodes, in stream order),
    ``ends`` (one past each head's dominant run), ``step_to_z``,
    ``next_form`` (per node, index of the first Formation at or after it), ``t_step``
    (cosmic time [Gyr] of each step, from the header cosmology) and
    ``reduce_dominant_run(ufunc, arr)``, which reduces any per-node array
    (length N or N+1) with a ufunc over each head's dominant-branch run
    ``[head, next_form[head]]`` inclusive. It returns one value per head (or a
    dict of such arrays, for a multi-component property).
    """
    asdf.get_config().validate_on_read = False
    tbl = read_asdf(f, load=['pos', 'mult', 'control', 'node_type', 'timestep', 'mult_sec'])
    meta = tbl.meta
    zg, tg = _cosmic_time_curve({k: float(meta[k]) for k in ('H0', 'Omega_M', 'Omega_K', 'Omega_DE', 'w0', 'wa')})
    return _history_from_stream(
        np.asarray(tbl['node_type']), np.asarray(tbl['timestep']).astype(np.int64),
        np.asarray(tbl['mult']).astype(np.int64), np.asarray(tbl['mult_sec']).astype(np.int64),
        np.asarray(tbl['pos']), np.asarray(meta['Step_to_Z']), extractor,
        t_step=np.interp(np.asarray(meta['Step_to_Z']), zg, tg))


def _history_from_stream(nt, ts, mult, msec, pos, step_to_z, extractor, t_step=None):
    """Array-only core of ``_history_one_file`` (separate so it's testable without ASDF)."""
    N = nt.shape[0]
    final = int(ts[nt == 3].max())
    heads = np.flatnonzero((nt == 3) & (ts == final))
    next_form = np.minimum.accumulate(np.where(nt == 0, np.arange(N), N)[::-1])[::-1]
    assert np.all(next_form[heads] < N), "head without a Formation (truncated file?)"
    ends = next_form[heads] + 1

    def reduce_dominant_run(ufunc, arr):
        # reduceat over [head, end) pairs: the even outputs are the runs we want
        # (odd ones span the gap to the next head and are discarded).
        idx = np.stack([heads, ends], axis=1).ravel()
        return ufunc.reduceat(arr, idx)[::2]

    h = dict(nt=nt, ts=ts, mult=mult, mult_sec=msec, pos=pos, heads=heads, ends=ends,
             next_form=next_form, step_to_z=np.asarray(step_to_z), t_step=t_step,
             reduce_dominant_run=reduce_dominant_run)
    values = extractor(h)
    values = {k: np.asarray(v, dtype=float) for k, v in values.items()} \
             if isinstance(values, dict) else np.asarray(values, dtype=float)
    return {'values': values,
            'mult': mult[heads].astype(np.int32),
            'pos': np.asarray(pos)[heads].astype(np.float32),
            'step': final, 'redshift': float(step_to_z[final - 1])}


def read_map_history_property(mdir, extractor=min_merger_ratio_primary, name=None,
                              nworkers=16, cache_dir=None, force=False, verbose=True):
    """Evaluate a merger-history property for every final-timeslice MAP of a run.

    Parses each ``maplogs/*/maplog_z*_*.asdf`` file (in parallel, one task per
    file, like ``read_maplog_dir``) and calls ``extractor`` on its node stream
    (see ``_history_one_file`` for the ``h`` dict it receives; ``min_merger_ratio_primary``
    is a worked example). Per-file results are concatenated in sorted-file
    order, which is the same order ``read_maplog_dir`` uses, so the returned
    ``values`` line up with that dict's final-timeslice rows.

    Writing a new property means writing one ``extractor(h) -> (len(h['heads']),)``
    function; everything downstream (``subsample_correlation_functions``) is
    unchanged. With ``nworkers > 1`` the extractor is pickled to worker
    processes, so it must be defined at module level (not a lambda or a
    notebook-local closure) -- or pass ``nworkers=1`` to run in-process.

    If ``cache_dir`` is given the result is pickled to
    ``<cache_dir>/history_<name>_<mdir.name>.pkl`` and reloaded on later calls
    (``force=True`` to rebuild). The cache key is the ``name``, not the extractor's
    code, so give a changed extractor a new ``name`` or pass ``force=True``.
    """
    name = name or getattr(extractor, '__name__', 'property')
    cache_path = None
    if cache_dir is not None:
        safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', name)
        cache_path = Path(cache_dir) / 'history_{}_{}.pkl'.format(safe, Path(mdir).name)
        if not force and cache_path.is_file():
            if verbose:
                print(f'  [cache hit] {cache_path.name}')
            with open(cache_path, 'rb') as fh:
                return pickle.load(fh)

    files = sorted(glob(str(Path(mdir) / Path("maplogs/*/maplog_z*_*.asdf"))))
    if not files:
        raise FileNotFoundError("no maplog files under {}".format(mdir))
    if verbose:
        print(f'  [parsing]   {len(files)} maplog files of {mdir}')
    asdf.get_config().validate_on_read = False
    if nworkers == 1:
        results = [_history_one_file(f, extractor) for f in files]
    else:
        with ProcessPoolExecutor(max_workers=nworkers) as ex:
            results = list(ex.map(partial(_history_one_file, extractor=extractor), files))
    steps = {r['step'] for r in results}
    assert len(steps) == 1, "maplog files disagree on the final timeslice: {}".format(steps)

    prop = MapHistoryProperty(
        name=name,
        values=({k: np.concatenate([r['values'][k] for r in results]) for k in results[0]['values']}
                if isinstance(results[0]['values'], dict)
                else np.concatenate([r['values'] for r in results])),
        step=results[0]['step'], redshift=results[0]['redshift'],
        mult=np.concatenate([r['mult'] for r in results]),
        pos=np.concatenate([r['pos'] for r in results], axis=0),
        source=str(mdir))

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(cache_path.name + '.tmp')
        with open(tmp_path, 'wb') as fh:
            pickle.dump(prop, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.rename(cache_path)
    return prop


def _in_zrange(z, zrange):
    """Boolean mask of ``z`` inside ``zrange=(zlo, zhi)`` (inclusive); all True for ``zrange=None``."""
    return np.ones(np.shape(z), dtype=bool) if zrange is None else (z >= zrange[0]) & (z <= zrange[1])


def _merger_rate_one_file(f, mult_min, with_vel_disp=False, zrange=None):
    """Read one maplog file; return per-MAP ``(mult, n_merger, rate)`` for final MAPs with ``mult >= mult_min``.

    Runs in a worker process. For each final-timeslice MAP the Merger nodes on its main branch (head
    to first Formation, see ``min_merger_ratio_primary``) are Poisson events: with N mergers at cosmic
    times t_1 <= ... <= t_N there are N-1 separations whose sum is t_N - t_1, so
    ``rate = (N - 1) / (t_N - t_1)`` [1/Gyr] (cosmic time from the header cosmology). NaN for N < 2 or
    when every merger falls in one timestep. With ``with_vel_disp`` the final timeslice node's ``vel_disp``
    (the environment's dark-matter velocity dispersion around the MAP, km/s) is returned too. With
    ``zrange=(zlo, zhi)`` only the mergers whose node redshift lies in ``[zlo, zhi]`` are the events, so the rate is
    that of the window (MAPs with fewer than 2 mergers in it have none).
    """
    asdf.get_config().validate_on_read = False
    tbl = read_asdf(f, load=['mult', 'control', 'node_type', 'timestep'] + (['vel_disp'] if with_vel_disp else []))
    nt = np.asarray(tbl['node_type'])
    ts = np.asarray(tbl['timestep']).astype(np.int64)
    mult = np.asarray(tbl['mult']).astype(np.int64)
    meta = tbl.meta
    cosmo = {k: float(meta[k]) for k in ('H0', 'Omega_M', 'Omega_K', 'Omega_DE', 'w0', 'wa')}
    step_to_z = np.asarray(meta['Step_to_Z'])
    zg, tg = _cosmic_time_curve(cosmo)
    t_step = np.interp(step_to_z, zg, tg)            # cosmic time [Gyr] of each step

    def extractor(h):
        nt_, ts_ = h['nt'], h['ts']
        is_m = (nt_ == 1) & _in_zrange(step_to_z[ts_ - 1], zrange)
        t_node = t_step[ts_ - 1]                      # a node's timestep is one ahead of its redshift
        tmin = np.full(nt_.shape[0] + 1, np.inf)      # +1 pad so a run ending at the last node is valid
        tmax = np.full(nt_.shape[0] + 1, -np.inf)
        cnt = np.zeros(nt_.shape[0] + 1)
        tmin[:-1][is_m] = t_node[is_m]
        tmax[:-1][is_m] = t_node[is_m]
        cnt[:-1][is_m] = 1.0
        n = h['reduce_dominant_run'](np.add, cnt)
        span = h['reduce_dominant_run'](np.maximum, tmax) - h['reduce_dominant_run'](np.minimum, tmin)
        with np.errstate(invalid='ignore', divide='ignore'):
            rate = np.where((n >= 2) & (span > 0), (n - 1) / span, np.nan)
        return {'n_merger': n, 'rate': rate}

    N = nt.shape[0]                                   # mult_sec is unused; `pos` is only indexed at the heads, so it
    # carries vel_disp through _history_from_stream (a zero-memory stand-in when it is not wanted)
    pos = (np.asarray(tbl['vel_disp']).astype(np.float32)[:, None] if with_vel_disp
           else np.broadcast_to(np.zeros(3, dtype=np.float32), (N, 3)))
    r = _history_from_stream(nt, ts, mult, np.zeros(N, dtype=np.int64), pos, step_to_z, extractor)
    keep = r['mult'] >= mult_min
    out = {'mult': r['mult'][keep], 'n_merger': r['values']['n_merger'][keep].astype(np.int32),
           'rate': r['values']['rate'][keep], 'step': r['step'], 'redshift': r['redshift']}
    if with_vel_disp:
        out['vel_disp'] = r['pos'][keep, 0]
    return out


def _accretion_rate_one_file(f, mult_min, with_vel_disp=False, zrange=None):
    """Read one maplog file; per-MAP accretion rate of the final MAPs with ``mult >= mult_min``.

    Runs in a worker process. Along a final MAP's main branch (head to first Formation) the
    multiplicity changes by merging in secondaries and by accretion; mergers are logged, accretion
    is not, so it is inferred as what is left. Between consecutive logged nodes (older state
    ``M0``; newer state ``Mf``, which for a Merger node is ``mult - mult_sec``, the primary just
    before merging) the accretion is ``Mf - M0``, and these telescope over the whole branch to

        accreted  = mult_final - sum(mult_sec over the branch's Merger nodes) - mult_at_formation,
        duration  = t(final) - t(formation),            rate = accreted / duration   [mult / Gyr]

    (cosmic time from the header cosmology). Because the sum telescopes, the result does not depend
    on which intermediate nodes (TimeSlice, LightCone) the run happened to log. ``rate`` is NaN unless
    ``accreted > 0`` and ``duration > 0``; ``accreted`` and ``duration`` are kept for every MAP. With
    ``with_vel_disp`` the final timeslice node's ``vel_disp`` (the environment's velocity dispersion, km/s)
    is returned too.

    With ``zrange=(zlo, zhi)`` only the intervals between consecutive logged nodes whose newer node has redshift
    in ``[zlo, zhi]`` are summed (accretion and duration alike), so the rate is that of the window; the
    interval touching a window edge is taken whole, so the window is resolved to the logged-node spacing. MAPs
    with no interval in the window have no rate.
    """
    asdf.get_config().validate_on_read = False
    tbl = read_asdf(f, load=['mult', 'mult_sec', 'control', 'node_type', 'timestep']
                    + (['vel_disp'] if with_vel_disp else []))
    nt = np.asarray(tbl['node_type'])
    ts = np.asarray(tbl['timestep']).astype(np.int64)
    mult = np.asarray(tbl['mult']).astype(np.int64)
    msec = np.asarray(tbl['mult_sec']).astype(np.int64)
    meta = tbl.meta
    cosmo = {k: float(meta[k]) for k in ('H0', 'Omega_M', 'Omega_K', 'Omega_DE', 'w0', 'wa')}
    step_to_z = np.asarray(meta['Step_to_Z'])
    zg, tg = _cosmic_time_curve(cosmo)
    t_step = np.interp(step_to_z, zg, tg)            # cosmic time [Gyr] of each step

    def extractor(h):
        nt_, mult_, msec_ = h['nt'], h['mult'], h['mult_sec']
        heads, form = h['heads'], h['next_form'][h['heads']]
        is_m = nt_ == 1
        s_sec = np.zeros(nt_.shape[0] + 1)            # +1 pad so a run ending at the last node is valid
        s_sec[:-1][is_m] = msec_[is_m]
        t_node = t_step[h['ts'] - 1]                  # a node's timestep is one ahead of its redshift
        if zrange is None:
            accreted = mult_[heads] - h['reduce_dominant_run'](np.add, s_sec) - mult_[form]
            duration = t_node[heads] - t_node[form]
        else:                                         # per-interval sums over the intervals ending in the window
            inw = (nt_ != 0) & _in_zrange(step_to_z[h['ts'] - 1], zrange)
            acc_n = np.zeros(nt_.shape[0] + 1)        # interval from node j+1 (older) to node j (newer)
            dur_n = np.zeros(nt_.shape[0] + 1)
            acc_n[:-2] = mult_[:-1] - s_sec[:-2] - mult_[1:]
            dur_n[:-2] = t_node[:-1] - t_node[1:]
            accreted = h['reduce_dominant_run'](np.add, np.where(np.append(inw, False), acc_n, 0.0))
            duration = h['reduce_dominant_run'](np.add, np.where(np.append(inw, False), dur_n, 0.0))
        with np.errstate(invalid='ignore', divide='ignore'):
            rate = np.where((accreted > 0) & (duration > 0), accreted / duration, np.nan)
        return {'accreted': accreted, 'duration': duration, 'rate': rate}

    N = nt.shape[0]                                   # `pos` is only indexed at the heads: it carries vel_disp
    pos = (np.asarray(tbl['vel_disp']).astype(np.float32)[:, None] if with_vel_disp
           else np.broadcast_to(np.zeros(3, dtype=np.float32), (N, 3)))
    r = _history_from_stream(nt, ts, mult, msec, pos, step_to_z, extractor)
    keep = r['mult'] >= mult_min
    out = {'mult': r['mult'][keep], 'accreted': r['values']['accreted'][keep].astype(np.float32),
           'duration': r['values']['duration'][keep].astype(np.float32), 'rate': r['values']['rate'][keep],
           'step': r['step'], 'redshift': r['redshift']}
    if with_vel_disp:
        out['vel_disp'] = r['pos'][keep, 0]
    return out


def _read_rate_runs(mdir, worker, prefix, mult_min, nfiles, seed, nworkers, cache_dir, force, verbose, fraction,
                    zrange=None):
    """Shared reader of ``read_merger_rates`` / ``read_accretion_rates``: run ``worker(file, mult_min=)``
    over the chosen maplog files in a process pool and concatenate its per-MAP arrays."""
    mdir = Path(mdir)
    files = sorted(glob(str(mdir / 'maplogs/*/maplog_z*_*.asdf')))
    if not files:
        raise FileNotFoundError("no maplog files under {}".format(mdir))
    if fraction is not None:
        nfiles = max(1, int(round(fraction * len(files))))
    n = len(files) if nfiles is None else min(int(nfiles), len(files))
    chosen = list(range(len(files))) if n == len(files) else sorted(random.Random(seed).sample(range(len(files)), n))
    label = mdir.name or mdir.parent.name
    if zrange is not None:
        zrange = (float(zrange[0]), float(zrange[1]))
        prefix = '{}_z{:g}-{:g}'.format(prefix, *zrange)          # a windowed rate is cached separately
        worker = partial(worker, zrange=zrange)
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / '{}_{}_m{}_n{}_s{}.pkl'.format(prefix, label, mult_min, n, seed)
        if not force and cache_path.is_file():
            if verbose:
                print(f'  [cache hit] {cache_path.name}')
            with open(cache_path, 'rb') as fh:
                return pickle.load(fh)
    if verbose:
        print(f'  [reading]   {n} of {len(files)} maplog files of {label} (mult >= {mult_min}, {nworkers} workers)')
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max(1, min(nworkers, n))) as ex:
        results = list(ex.map(partial(worker, mult_min=mult_min), [files[i] for i in chosen]))
    steps = {r['step'] for r in results}
    assert len(steps) == 1, "maplog files disagree on the final timeslice: {}".format(steps)
    run = {k: np.concatenate([r[k] for r in results]) for k in results[0] if k not in ('step', 'redshift')}
    run.update({'file': np.concatenate([np.full(len(r['mult']), i, dtype=np.int32) for i, r in zip(chosen, results)]),
                'redshift': results[0]['redshift'], 'nfiles': n, 'label': label, 'seconds': time.time() - t0,
                'zrange': zrange})
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(cache_path.name + '.tmp')
        with open(tmp_path, 'wb') as fh:
            pickle.dump(run, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.rename(cache_path)
    return run


def read_merger_rates(mdir, mult_min=100, nfiles=None, seed=0, nworkers=16, cache_dir=None, force=False,
                      verbose=True, fraction=None, with_vel_disp=False, zrange=None):
    """Merger rate of every final-timeslice MAP with ``mult >= mult_min`` in a run (see ``_merger_rate_one_file``).

    ``nfiles`` (default: all) reads that many maplog files chosen at random (``seed``);
    ``fraction`` (0-1) instead reads that fraction of the run's files -- use it to compare runs
    with different file counts (different decompositions), since the same number of files then
    means different volumes. Files are independent and read in parallel by ``nworkers``
    processes, each holding one decompressed file (several GB). If ``cache_dir`` is given the
    result is pickled there, keyed on run name, ``mult_min``, file count and seed (``force=True``
    to rebuild). Returns a dict: ``mult``, ``n_merger``, ``rate`` (per MAP), ``file`` (file index of
    each MAP), ``redshift``, ``nfiles``, ``label`` (the run's directory name), ``seconds``.
    """
    if with_vel_disp:                                 # also returns ``vel_disp`` per MAP; cached separately
        return _read_rate_runs(mdir, partial(_merger_rate_one_file, with_vel_disp=True), 'merger_rates_vd', mult_min,
                               nfiles, seed, nworkers, cache_dir, force, verbose, fraction, zrange)
    return _read_rate_runs(mdir, _merger_rate_one_file, 'merger_rates', mult_min, nfiles, seed, nworkers,
                           cache_dir, force, verbose, fraction, zrange)


def read_accretion_rates(mdir, mult_min=100, nfiles=None, seed=0, nworkers=16, cache_dir=None, force=False,
                         verbose=True, fraction=None, with_vel_disp=False, zrange=None):
    """Accretion rate of every final-timeslice MAP with ``mult >= mult_min`` in a run (see ``_accretion_rate_one_file``).

    Same file selection, parallelism and caching as ``read_merger_rates``. Returns a dict: ``mult``,
    ``accreted`` (multiplicity accreted along the main branch), ``duration`` [Gyr, formation to final],
    ``rate`` (= accreted / duration [mult/Gyr]; NaN unless both are positive), ``file``, ``redshift``,
    ``nfiles``, ``label``, ``seconds``. ``with_vel_disp=True`` adds ``vel_disp`` per MAP (cached separately).
    """
    if with_vel_disp:
        return _read_rate_runs(mdir, partial(_accretion_rate_one_file, with_vel_disp=True), 'accretion_rates_vd',
                               mult_min, nfiles, seed, nworkers, cache_dir, force, verbose, fraction, zrange)
    return _read_rate_runs(mdir, _accretion_rate_one_file, 'accretion_rates', mult_min, nfiles, seed, nworkers,
                           cache_dir, force, verbose, fraction, zrange)


def _binned_percentiles(x, y, edges, nmin=10):
    """16/50/84th percentiles of the finite ``y`` in the bins ``edges`` of ``x`` (NaN where fewer than ``nmin``)."""
    out = np.full((3, len(edges) - 1), np.nan)
    idx = np.digitize(x, edges) - 1
    for i in range(len(edges) - 1):
        v = y[(idx == i) & np.isfinite(y)]
        if len(v) >= nmin:
            out[:, i] = np.percentile(v, [16, 50, 84])
    return out


def _binned_median_ci(x, y, edges, nmin=10):
    """Sampling uncertainty of the median of the finite ``y`` in the bins ``edges`` of ``x``: ``(2, nbins)`` array of
    the lower/upper bounds of a ~68% confidence interval (NaN where fewer than ``nmin``).

    Distribution-free, from order statistics: the number K of samples below the true median is Binomial(N, 1/2),
    and X_(j) <= median <= X_(N-j+1) exactly when j <= K <= N-j, so with j chosen so the exact binomial coverage is closest to 68%
    (j ~ N/2 - sqrt(N)/2) the sorted samples X_(j), X_(N-j+1) bracket the median at ~68% (1 sigma). This is the rank-space
    form of the standard error 1 / (2 sqrt(N) f(median)): it needs no density estimate and no finite variance.
    Counts the sampling noise only -- not the intrinsic scatter (``_binned_percentiles``) nor clustering of the MAPs.
    """
    from scipy.stats import binom
    out = np.full((2, len(edges) - 1), np.nan)
    idx = np.digitize(x, edges) - 1
    for i in range(len(edges) - 1):
        v = y[(idx == i) & np.isfinite(y)]
        n = len(v)
        if n >= nmin:
            v = np.sort(v)
            j0 = int(binom.ppf(0.1587, n, 0.5))             # 1-based rank of the lower bound: of the neighbouring
            cand = np.arange(max(j0 - 1, 1), j0 + 2)       # integers, the one whose exact coverage 1 - 2 P(K <= j-1)
            j = int(cand[np.argmin(np.abs(1 - 2 * binom.cdf(cand - 1, n, 0.5) - 0.6827))])   # is closest to 68%
            out[:, i] = v[j - 1], v[n - j]
    return out


def _median_diff_sigma(med, ci, med0, ci0):
    """Difference of two median curves in units of its own sampling error: ``(med - med0) / sqrt(sigma^2 + sigma0^2)``,
    where sigma is the half-width of the ~68% CI of the median (``_binned_median_ci``), i.e. the 1-sigma sampling error
    of that curve. The two errors are added in quadrature, which assumes the runs' samples are independent (runs sharing
    their MAPs / initial phases are correlated, which makes this conservative). |value| > 1: the medians differ by
    more than their combined sample variance."""
    with np.errstate(invalid='ignore', divide='ignore'):
        return (med - med0) / np.hypot(0.5 * (ci[1] - ci[0]), 0.5 * (ci0[1] - ci0[0]))


def _select_abundance(runs, dirs, abundance):
    """Abundance-match each per-run dict (as returned by ``read_merger_rates`` / ``read_accretion_rates``).

    The multiplicity threshold of a run is ``get_threshold``'s: the multiplicity of the
    ``abundance * V``-th most massive MAP, where ``abundance`` is a number density [(h/Mpc)^3] and V is the
    volume the MAPs came from -- the box volume times the fraction of the run's maplog files that were read
    (the files are taken to cover equal volumes). MAPs with ``mult >= threshold`` are kept (ties kept, as in
    ``get_map_xi``) in every per-MAP array; the dict gains ``mult_threshold``, ``abundance`` and ``volume``.
    The MAPs read must reach down to the abundance (``mult_min`` low enough), or a ``ValueError`` says so.
    Idempotent: applying it to already selected runs changes nothing.
    """
    out = []
    for run, mdir in zip(runs, dirs):
        files = sorted(glob(str(Path(mdir) / 'maplogs/*/maplog_z*_*.asdf')))
        asdf.get_config().validate_on_read = False
        with asdf.open(files[0], lazy_load=True, validate_checksums=False) as af:
            L = float(af.tree['header']['BoxSize'])
        L_eff = L * (run['nfiles'] / len(files)) ** (1.0 / 3.0)        # side of a cube with the volume read
        ntarget = int(abundance * L_eff**3)
        if ntarget > len(run['mult']):
            raise ValueError("{}: abundance {:g} (h/Mpc)^3 over the {:.3g} (Mpc/h)^3 read needs {:,} MAPs but only {:,} "
                             "were read (mult >= {}): lower mult_min".format(run['label'], abundance, L_eff**3, ntarget,
                                                                           len(run['mult']), int(run['mult'].min())))
        thr = int(get_threshold(run['mult'], L_eff * u.Mpc, guess=0, abundance=abundance)[0])
        keep = run['mult'] >= thr
        n = len(run['mult'])
        sel = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n else v) for k, v in run.items()}
        sel.update(mult_threshold=thr, abundance=float(abundance), volume=float(L_eff**3))
        out.append(sel)
    return out


def _save_tag(runs):
    """File-name tag of a rate figure: the redshift window and abundance (or multiplicity cut) it used."""
    zr, ab = runs[0].get('zrange'), runs[0].get('abundance')
    return ('' if zr is None else '_z{:g}-{:g}'.format(*zr)) + ('' if ab is None else '_ab{:.1e}'.format(ab))


def _window_title(runs):
    """Figure title suffix naming the redshift window the rates were computed in (empty for the whole history)."""
    zr = runs[0].get('zrange')
    return '' if zr is None else '   rates in z = [{:.3g}, {:.3g}]'.format(*zr)


def _abundance_title(runs, mult_min):
    """Figure title text: the abundance (if the runs were abundance-matched) or the multiplicity cut."""
    if 'abundance' in runs[0]:
        return 'abundance {:.1e} (h/Mpc)$^3$'.format(runs[0]['abundance'])
    return 'mult >= {}'.format(mult_min)


def _compare_rate_runs(data, labels, mult_min, nbins, panel_size, colors, min_count, ratio_ylim, rate_label,
                       rate_word, fname, save, str_identifier, mult_nbins=None):
    """Shared engine of ``compare_merger_rate`` / ``compare_accretion_rates``: draw the per-run dicts in ``data``
    (each with ``mult`` and ``rate``, NaN = no rate) as histograms of multiplicity and rate, rate vs multiplicity,
    and the ratios to the first run. Returns ``(fig, axes)``."""
    runs = [dict(r, label=(labels[i] if labels is not None else r['label'])) for i, r in enumerate(data)]
    colors = colors or ['C0', 'C3', 'C2', 'C1', 'C4', 'C5']

    allm = np.concatenate([r['mult'] for r in runs])
    allr = np.concatenate([r['rate'][np.isfinite(r['rate'])] for r in runs])
    lo = min(r.get('mult_threshold', mult_min) for r in runs)
    mb = np.logspace(np.log10(max(lo, 1)), np.log10(allm.max()), (mult_nbins or nbins - 1) + 1)   # mult_nbins bins; default as before
    rb = np.logspace(np.log10(allr.min()), np.log10(allr.max()), nbins)
    mmid, rmid = np.sqrt(mb[:-1] * mb[1:]), np.sqrt(rb[:-1] * rb[1:])
    nrow = 2 if len(runs) > 1 else 1
    fig, axes = plt.subplots(nrow, 3, sharex='col', squeeze=False, constrained_layout=True,
                             figsize=(panel_size[0] * 3, panel_size[1] * (1 + 0.45 * (nrow - 1))),
                             gridspec_kw={'height_ratios': [1, 0.45][:nrow]})

    # histograms (normalised to unit area) and their raw counts, for the Poisson errors of the ratios
    hist = []
    for run in runs:
        ok = np.isfinite(run['rate'])
        hist.append([(np.histogram(x, bins=b, density=True)[0], np.histogram(x, bins=b)[0])
                     for x, b in ((run['mult'], mb), (run['rate'][ok], rb))])
    for run, c, hh in zip(runs, colors, hist):
        ok = np.isfinite(run['rate'])
        axes[0, 0].stairs(hh[0][0], mb, color=c, lw=1.6,
                          label=f'{run["label"]}\n{len(run["mult"]):,} MAPs, '
                                + (f'mult >= {run["mult_threshold"]}, ' if 'mult_threshold' in run else '')
                                + f'{100 * ok.mean():.0f}% {rate_word}')
        axes[0, 1].stairs(hh[1][0], rb, color=c, lw=1.6)
    axes[0, 0].set(xscale='log', yscale='log', ylabel='normalised N', xlim=(mb[0], mb[-1]))
    axes[0, 0].legend(fontsize=6.5, loc='upper right')
    axes[0, 1].set(xscale='log', yscale='log', ylabel='normalised N', xlim=(rb[0], rb[-1]))

    first = runs[0]
    ok = np.isfinite(first['rate'])
    h = axes[0, 2].hist2d(first['mult'][ok], first['rate'][ok], bins=[mb, rb], norm=LogNorm(), cmap='Greys')
    fig.colorbar(h[3], ax=axes[:, 2], label=f'N MAPs ({runs[0]["label"]}, reference)')
    quant = [_binned_percentiles(r['mult'], r['rate'], mb) for r in runs]
    mci = [_binned_median_ci(r['mult'], r['rate'], mb) for r in runs]
    for run, c, q, ci in zip(runs, colors, quant, mci):
        axes[0, 2].plot(mmid, q[1], color=c, lw=2, label=run['label'] + (' (reference)' if run is runs[0] else ''))
        axes[0, 2].fill_between(mmid, q[0], q[2], color=c, alpha=0.2)
        for b in ci:                                  # sampling uncertainty of the median (dotted)
            axes[0, 2].plot(mmid, b, color=c, lw=1, ls=':')
    axes[0, 2].set(xscale='log', yscale='log', ylabel=rate_label)
    axes[0, 2].legend(fontsize=7, title='median, 16-84% (dotted: 68% CI of median)', title_fontsize=7)

    if nrow > 1:                                      # ratios to the first run
        for k, (xm, col) in enumerate(((mmid, 0), (rmid, 1))):
            for run, c, hh in list(zip(runs, colors, hist))[1:]:
                d0, n0 = hist[0][col]
                d1, n1 = hh[col]
                good = (n0 >= min_count) & (n1 >= min_count)
                ratio = np.where(good, d1 / np.where(d0 > 0, d0, np.nan), np.nan)
                err = ratio * np.sqrt(1.0 / np.maximum(n0, 1) + 1.0 / np.maximum(n1, 1))
                axes[1, k].errorbar(xm, ratio, yerr=err, fmt='o-', ms=2.5, lw=1, color=c)
        for run, c, q, ci in list(zip(runs, colors, quant, mci))[1:]:
            axes[1, 2].plot(mmid, _median_diff_sigma(q[1], ci, quant[0][1], mci[0]), color=c, lw=1.5)
        ref_lab = runs[0]['label']
        ratio_labels = (f'norm. N / ({ref_lab})', f'norm. N / ({ref_lab})', r'$\Delta$median / $\sigma_{\rm comb}$')
        ratio_notes = ('ratio of the normalised histograms', 'ratio of the normalised histograms',
                       r'(median $-$ median$_{\rm reference}$) / $\sigma$')
        for ax, lab, note in zip(axes[1], ratio_labels, ratio_notes):
            ref = 0.0 if ax is axes[1, 2] else 1.0         # the difference panel is centred on 0, the ratios on 1
            ax.axhline(ref, color='k', lw=0.8)
            ax.set(xscale='log')
            ax.set_ylabel(lab, fontsize=8)
            ax.text(0.02, 0.04, note, transform=ax.transAxes, fontsize=5.5, va='bottom', color='0.3')
            if ratio_ylim is not None and ref == 0.0:     # the histogram ratios are always autoscaled
                ax.set_ylim(ratio_ylim)
            else:
                v = np.concatenate([np.asarray(l.get_ydata(), dtype=float).ravel() for l in ax.get_lines()[1:]]
                                   + [np.asarray(pa.vertices[:, 1], dtype=float) for c in ax.collections
                                      for pa in c.get_paths()])      # every error bar / band, not just the first
                v = v[np.isfinite(v)]
                if len(v):
                    lo, hi = np.percentile(v, [2, 98])
                    lo, hi = min(lo, ref), max(hi, ref)
                    if ref == 0.0:
                        lo, hi = min(lo, -1.0), max(hi, 1.0)
                    pad = 0.15 * (hi - lo) + 0.01
                    ax.set_ylim(lo - pad, hi + pad)
            if ref == 0.0:
                for g in (-1, 1):
                    ax.axhline(g, color='0.5', lw=0.6, ls=':')
    axes[-1, 0].set_xlabel('multiplicity')
    axes[-1, 1].set_xlabel(rate_label)
    axes[-1, 2].set_xlabel('multiplicity')
    fig.suptitle(f'z={first["redshift"]:.3g}   {_abundance_title(runs, mult_min)}{_window_title(runs)}', y=1.0, va='bottom')

    if save:
        ids = "_" + str_identifier if len(str_identifier) > 0 else ''
        fig.savefig('../Images/' + fname + _save_tag(runs) + ids + '.png', bbox_inches='tight', dpi=200)
    plt.show()
    return fig, axes


def compare_merger_rate(dirs, labels=None, mult_min=100, nfiles=None, fraction=None, seed=0, nworkers=16,
                        data=None, cache_dir=None, nbins=50, panel_size=(5, 3.8), colors=None, min_count=20,
                        ratio_ylim=(-3, 3), save=False, str_identifier='', abundance=None, zrange=None,
                        mult_nbins=None):
    """Compare the merger-rate distributions of several runs (same MAP selection in each).

    For every final-timeslice MAP with ``mult >= mult_min`` the merger rate is the Poisson estimate
    ``(N - 1) / (t_N - t_1)`` over its main-branch mergers (see ``_merger_rate_one_file``); MAPs with
    fewer than 2 mergers have no rate (they stay in the multiplicity histogram, not in the rate ones).
    A multiplicity cut selects like-for-like MAPs only between runs with the same particle mass.
    Three panels: the multiplicity histogram, the rate histogram (both normalised to unit area, so
    runs of different size / file subsets compare), and rate vs multiplicity -- the first run as a
    2D histogram, every run's median and 16-84% range in multiplicity bins as lines. With more than one
    run a second row shows each curve of the top row divided by the first run's, bin by bin: the ratio
    of the normalised histograms (with Poisson errors; bins with fewer than ``min_count`` MAPs in either
    run are left out) and, in the third column, 
    the difference of the median rates over the two medians' 1-sigma sampling errors (half the 68% CI from
    the sqrt(N)/2 order statistics) added in quadrature.

    Parameters
    ----------
    dirs : sequence of run directories (each with ``maplogs/col*/maplog_z*.asdf``).
    labels : legend labels (default: the directory names).
    mult_min, nfiles, fraction, seed, nworkers, cache_dir : see ``read_merger_rates`` (by default every file
        is read; ``fraction`` reads the same fraction of each run's files).
    data : list of per-run dicts from an earlier call, to redraw without re-reading.
    colors : line colours (default C0, C3, C2, C1, ...).
    min_count : minimum MAPs per histogram bin, in both runs, for a point of the ratio row.
    mult_nbins : number of log-spaced multiplicity bins, used for the multiplicity histogram and for the medians / CIs
        in rate vs multiplicity (default ``nbins - 1``, as before). Fewer, wider bins put more MAPs in each, shrinking the CI of the
        median, but average the rate over a wider range of multiplicity.
    ratio_ylim : y-limits of the median-difference panel (Delta median / sigma_comb; default (-3, 3); None autoscales from the
        central 96% of the curves, always including +/-1). The histogram-ratio panels are always autoscaled.

    abundance : number density [(h/Mpc)^3] to match instead of a fixed multiplicity: each run's threshold is the
        multiplicity of its ``abundance * V``-th most massive MAP (``get_threshold``; V = box volume x fraction of
        files read), applied to every panel; ``mult_min`` is then only the floor of the MAPs read and must be low
        enough to reach the abundance. Thresholds are printed in the legend; the returned ``data`` is the selected one.

    zrange : ``(zlo, zhi)`` -- compute the rates only from the history inside this redshift window instead of over each
        MAP's whole main branch (``None``): the merger rate uses only the mergers in the window (the same Poisson estimate over them), MAPs with no two mergers in the window have no rate. The MAPs are still the
        final-timeslice ones, selected (abundance / multiplicity) at the final redshift; the window is in the title.

    Returns ``(fig, axes, data)`` with ``axes`` of shape (2, 3) -- (1, 3) for a single run; ``data`` is the list of per-run dicts of ``read_merger_rates``.
    """
    dirs = list(dirs)
    if labels is not None:
        assert len(labels) == len(dirs), "labels must have one entry per directory"
    if data is None:
        data = [read_merger_rates(d, mult_min, nfiles=nfiles, seed=seed, nworkers=nworkers, cache_dir=cache_dir,
                                  fraction=fraction, zrange=zrange) for d in dirs]
    if abundance is not None:
        data = _select_abundance(data, dirs, abundance)
    fig, axes = _compare_rate_runs(data, labels, mult_min, nbins, panel_size, colors, min_count, ratio_ylim,
                                   r'merger rate [Gyr$^{-1}$]', 'with a rate', 'compare_merger_rate', save,
                                   str_identifier, mult_nbins)
    return fig, axes, data


def compare_accretion_rates(dirs, labels=None, mult_min=100, nfiles=None, fraction=None, seed=0, nworkers=16,
                            data=None, cache_dir=None, nbins=50, panel_size=(5, 3.8), colors=None, min_count=20,
                            ratio_ylim=(-3, 3), save=False, str_identifier='', abundance=None, zrange=None,
                            mult_nbins=None):
    """Compare the MAP accretion-rate distributions of several runs; same layout as ``compare_merger_rate``.

    For every final-timeslice MAP with ``mult >= mult_min`` the accretion rate is the multiplicity it
    accreted (as opposed to merged in) along its main branch, divided by the cosmic time over which it did so
    (formation to the final timeslice), in multiplicity / Gyr -- the amount of accretion over the time spent
    accreting. Mergers are logged but accretion is not, so the amount is inferred from the multiplicity
    changes between logged nodes, excluding what mergers added (see ``_accretion_rate_one_file``):

        accreted = mult_final - sum(mult_secondary of the main-branch mergers) - mult_at_formation.

    The sum telescopes over the intervals between consecutive nodes, so it is the same whichever
    intermediate nodes are logged: TimeSlice and LightCone nodes alike (a sim with a lightcone and one
    without compare like for like) -- they would only matter for a per-interval rate. MAPs with no net
    accretion (``accreted <= 0``) have no rate: they stay in the multiplicity histogram, not in the rate ones.

    Parameters, panels, ratio row and return value are as in ``compare_merger_rate``; the data dicts are those of
    ``read_accretion_rates`` (cached separately from the merger rates).

    abundance : number density [(h/Mpc)^3] to match instead of a fixed multiplicity: each run's threshold is the
        multiplicity of its ``abundance * V``-th most massive MAP (``get_threshold``; V = box volume x fraction of
        files read), applied to every panel; ``mult_min`` is then only the floor of the MAPs read and must be low
        enough to reach the abundance. Thresholds are printed in the legend; the returned ``data`` is the selected one.

    zrange : ``(zlo, zhi)`` -- compute the rates only from the history inside this redshift window instead of over each
        MAP's whole main branch (``None``): the accretion rate sums the accreted multiplicity and the time of the logged-node intervals ending in the window, MAPs with no interval in the window have no rate. The MAPs are still the
        final-timeslice ones, selected (abundance / multiplicity) at the final redshift; the window is in the title.
    """
    dirs = list(dirs)
    if labels is not None:
        assert len(labels) == len(dirs), "labels must have one entry per directory"
    if data is None:
        data = [read_accretion_rates(d, mult_min, nfiles=nfiles, seed=seed, nworkers=nworkers, cache_dir=cache_dir,
                                     fraction=fraction, zrange=zrange) for d in dirs]
    if abundance is not None:
        data = _select_abundance(data, dirs, abundance)
    fig, axes = _compare_rate_runs(data, labels, mult_min, nbins, panel_size, colors, min_count, ratio_ylim,
                                   r'accretion rate [mult. Gyr$^{-1}$]', 'with accretion > 0',
                                   'compare_accretion_rates', save, str_identifier, mult_nbins)
    return fig, axes, data


def _compare_rate_vs_veldisp(data, labels, mult_min, nbins, panel_size, colors, min_count, ratio_ylim, rate_label,
                             fname, save, str_identifier):
    """Shared engine of ``compare_merger_rate_vs_veldisp`` / ``compare_accretion_rate_vs_veldisp``: one panel per
    per-run dict in ``data`` (with ``vel_disp``, ``rate``; NaN rate = none), then the overlay and ratio axes.
    Returns ``(fig, axes)``."""
    runs = [dict(r, label=(labels[i] if labels is not None else r['label'])) for i, r in enumerate(data)]
    colors = colors or ['C0', 'C3', 'C2', 'C1', 'C4', 'C5']
    n = len(runs)
    sel = [np.isfinite(r['rate']) & np.isfinite(r['vel_disp']) & (r['vel_disp'] > 0) for r in runs]
    allx = np.concatenate([r['vel_disp'][m] for r, m in zip(runs, sel)])
    ally = np.concatenate([r['rate'][m] for r, m in zip(runs, sel)])
    xb = np.logspace(np.log10(np.percentile(allx, 0.05)), np.log10(allx.max()), nbins)   # trim a few very low-sigma_v outliers
    yb = np.logspace(np.log10(ally.min()), np.log10(ally.max()), nbins)
    xmid = np.sqrt(xb[:-1] * xb[1:])
    ratio = n > 1

    fig = plt.figure(figsize=(panel_size[0] * (n + 1), panel_size[1]), constrained_layout=True)
    gs = fig.add_gridspec(2, n + 1, height_ratios=[1, 0.45])
    run_axes = []
    for k in range(n):
        run_axes.append(fig.add_subplot(gs[:, k], sharex=run_axes[0] if k else None, sharey=run_axes[0] if k else None))
    ax_top = fig.add_subplot(gs[0, n]) if ratio else fig.add_subplot(gs[:, n])
    ax_rat = fig.add_subplot(gs[1, n], sharex=ax_top) if ratio else None

    counts = [np.histogram2d(r['vel_disp'][m], r['rate'][m], bins=[xb, yb])[0] for r, m in zip(runs, sel)]
    norm = LogNorm(vmin=1, vmax=max(c.max() for c in counts))
    quant = [_binned_percentiles(r['vel_disp'][m], r['rate'][m], xb, nmin=min_count) for r, m in zip(runs, sel)]
    mci = [_binned_median_ci(r['vel_disp'][m], r['rate'][m], xb, nmin=min_count) for r, m in zip(runs, sel)]
    for k, (ax, run, c, q, cnt) in enumerate(zip(run_axes, runs, colors, quant, counts)):
        im = ax.pcolormesh(xb, yb, cnt.T, norm=norm, cmap='Greys', rasterized=True)
        ax.plot(xmid, q[1], color=c, lw=2)
        ax.fill_between(xmid, q[0], q[2], color=c, alpha=0.25)
        for b in mci[k]:
            ax.plot(xmid, b, color=c, lw=1, ls=':')
        ax.set(xscale='log', yscale='log', xlabel=r'environment $\sigma_v$ [km/s]', xlim=(xb[0], xb[-1]))
        ax.text(0.04, 0.95, f'{run["label"]}{" (reference)" if k == 0 else ""}\n{sel[k].sum():,} MAPs'
                + (f', mult >= {run["mult_threshold"]}' if 'mult_threshold' in run else ''),
                transform=ax.transAxes, va='top', fontsize=7,
                bbox=dict(boxstyle='round', fc='white', ec='none', alpha=0.8))
        if k == 0:
            ax.set_ylabel(rate_label)
    fig.colorbar(im, ax=run_axes, label='N MAPs', pad=0.01)
    for run, c, q, ci in zip(runs, colors, quant, mci):
        ax_top.plot(xmid, q[1], color=c, lw=2, label=run['label'] + (' (reference)' if run is runs[0] else ''))
        ax_top.fill_between(xmid, q[0], q[2], color=c, alpha=0.2)
        for b in ci:
            ax_top.plot(xmid, b, color=c, lw=1, ls=':')
    ax_top.set(xscale='log', yscale='log', ylabel=rate_label, xlim=(xb[0], xb[-1]))
    ax_top.legend(fontsize=7, title='median, 16-84% (dotted: 68% CI of median)', title_fontsize=7)
    if ratio:
        for run, c, q, ci in list(zip(runs, colors, quant, mci))[1:]:
            ax_rat.plot(xmid, _median_diff_sigma(q[1], ci, quant[0][1], mci[0]), color=c, lw=1.5)
        ax_rat.axhline(0, color='k', lw=0.8)
        for g in (-1, 1):
            ax_rat.axhline(g, color='0.5', lw=0.6, ls=':')
        ax_rat.set(xscale='log', xlabel=r'environment $\sigma_v$ [km/s]')
        ax_rat.set_ylabel(r'$\Delta$median / $\sigma_{\rm comb}$', fontsize=8)
        ax_rat.text(0.02, 0.04, r'(median $-$ median$_{\rm reference}$) / $\sigma$',
                    transform=ax_rat.transAxes, fontsize=5.5, va='bottom', color='0.3')
        if ratio_ylim is not None:
            ax_rat.set_ylim(ratio_ylim)
        plt.setp(ax_top.get_xticklabels(), visible=False)
    else:
        ax_top.set_xlabel(r'environment $\sigma_v$ [km/s]')
    fig.suptitle(f'z={runs[0]["redshift"]:.3g}   {_abundance_title(runs, mult_min)}{_window_title(runs)}', y=1.0, va='bottom')

    if save:
        ids = "_" + str_identifier if len(str_identifier) > 0 else ''
        fig.savefig('../Images/' + fname + _save_tag(runs) + ids + '.png', bbox_inches='tight', dpi=200)
    plt.show()
    return fig, run_axes + [ax_top] + ([ax_rat] if ratio else [])


def compare_merger_rate_vs_veldisp(dirs, labels=None, mult_min=100, nfiles=None, fraction=None, seed=0,
                                   nworkers=16, data=None, cache_dir=None, nbins=40, panel_size=(5, 4.2),
                                   colors=None, min_count=20, ratio_ylim=(-3, 3), save=False,
                                   str_identifier='', abundance=None, zrange=None):
    """Merger rate of the MAPs vs the velocity dispersion of the environment around them, for several runs.

    For every final-timeslice MAP with ``mult >= mult_min`` and a merger rate (see ``compare_merger_rate``)
    the rate is plotted against the MAP's ``vel_disp``: the dark-matter velocity dispersion of its
    environment, as logged on its final timeslice node [km/s]. One panel per run: the 2D histogram (shared
    axes and colour scale) with the run's median and 16-84% range of the rate in ``vel_disp`` bins.
    The last column overlays those medians (line) and ranges (band) for all runs and, below it for more than
    one run, their difference to the first run's over the two medians'
    1-sigma sampling errors added in quadrature. ``min_count`` is the minimum MAPs per ``vel_disp`` bin for a percentile.

    Needs the per-MAP ``vel_disp``, which the plain merger-rate caches lack: ``read_merger_rates`` is called
    with ``with_vel_disp=True`` (a separate cache, ``merger_rates_vd_*``), which re-reads the maplogs once.
    Other parameters as in ``compare_merger_rate``. Returns ``(fig, axes, data)``; ``axes`` is the list of
    per-run axes followed by the overlay axes (and the ratio axes, for more than one run).

    abundance : number density [(h/Mpc)^3] to match instead of a fixed multiplicity: each run's threshold is the
        multiplicity of its ``abundance * V``-th most massive MAP (``get_threshold``; V = box volume x fraction of
        files read), applied to every panel; ``mult_min`` is then only the floor of the MAPs read and must be low
        enough to reach the abundance. Thresholds are printed in the legend; the returned ``data`` is the selected one.

    zrange : ``(zlo, zhi)`` -- compute the rates only from the history inside this redshift window instead of over each
        MAP's whole main branch (``None``): the merger rate uses only the mergers in the window (the same Poisson estimate over them), MAPs with no two mergers in the window have no rate. The MAPs are still the
        final-timeslice ones, selected (abundance / multiplicity) at the final redshift; the window is in the title.
    """
    dirs = list(dirs)
    if labels is not None:
        assert len(labels) == len(dirs), "labels must have one entry per directory"
    if data is None:
        data = [read_merger_rates(d, mult_min, nfiles=nfiles, seed=seed, nworkers=nworkers, cache_dir=cache_dir,
                                  fraction=fraction, with_vel_disp=True, zrange=zrange) for d in dirs]
    if abundance is not None:
        data = _select_abundance(data, dirs, abundance)
    fig, axes = _compare_rate_vs_veldisp(data, labels, mult_min, nbins, panel_size, colors, min_count, ratio_ylim,
                                         r'merger rate [Gyr$^{-1}$]', 'compare_merger_rate_vs_veldisp', save,
                                         str_identifier)
    return fig, axes, data


def compare_accretion_rate_vs_veldisp(dirs, labels=None, mult_min=100, nfiles=None, fraction=None, seed=0,
                                      nworkers=16, data=None, cache_dir=None, nbins=40, panel_size=(5, 4.2),
                                      colors=None, min_count=20, ratio_ylim=(-3, 3), save=False,
                                      str_identifier='', abundance=None, zrange=None):
    """Accretion rate of the MAPs vs the velocity dispersion of the environment around them, for several runs.

    Same plot as ``compare_merger_rate_vs_veldisp`` with the y-axis the per-MAP accretion rate of
    ``compare_accretion_rates`` (multiplicity accreted along the main branch / time from formation to the final
    timeslice, mult/Gyr; MAPs with no net accretion have no rate and are left out) against the MAP's ``vel_disp``
    on its final timeslice node [km/s]: one 2D histogram per run with its median and 16-84% range, then the overlay
    and the difference of the medians to the first run's over the two medians' 1-sigma sampling errors added in quadrature.
    ``vel_disp`` is not in the plain accretion-rate caches, so ``read_accretion_rates`` is called with
    ``with_vel_disp=True`` (cached separately, ``accretion_rates_vd_*``), which re-reads the maplogs once.
    Parameters as in ``compare_merger_rate``. Returns ``(fig, axes, data)``.

    abundance : number density [(h/Mpc)^3] to match instead of a fixed multiplicity: each run's threshold is the
        multiplicity of its ``abundance * V``-th most massive MAP (``get_threshold``; V = box volume x fraction of
        files read), applied to every panel; ``mult_min`` is then only the floor of the MAPs read and must be low
        enough to reach the abundance. Thresholds are printed in the legend; the returned ``data`` is the selected one.

    zrange : ``(zlo, zhi)`` -- compute the rates only from the history inside this redshift window instead of over each
        MAP's whole main branch (``None``): the accretion rate sums the accreted multiplicity and the time of the logged-node intervals ending in the window, MAPs with no interval in the window have no rate. The MAPs are still the
        final-timeslice ones, selected (abundance / multiplicity) at the final redshift; the window is in the title.
    """
    dirs = list(dirs)
    if labels is not None:
        assert len(labels) == len(dirs), "labels must have one entry per directory"
    if data is None:
        data = [read_accretion_rates(d, mult_min, nfiles=nfiles, seed=seed, nworkers=nworkers, cache_dir=cache_dir,
                                     fraction=fraction, with_vel_disp=True, zrange=zrange) for d in dirs]
    if abundance is not None:
        data = _select_abundance(data, dirs, abundance)
    fig, axes = _compare_rate_vs_veldisp(data, labels, mult_min, nbins, panel_size, colors, min_count, ratio_ylim,
                                         r'accretion rate [mult. Gyr$^{-1}$]', 'compare_accretion_rate_vs_veldisp',
                                         save, str_identifier)
    return fig, axes, data


def bin_subsamples(edges, fmt='.3g', quantity='x'):
    """Subsample spec splitting a property at ``edges``, for ``subsample_correlation_functions``.

    Bins are ``(lo, hi]`` with ``-inf``/``+inf`` on the ends, so a single edge
    ``[0.01]`` gives "x <= 0.01" and "x > 0.01". NaN values (property undefined)
    fall in no bin. Returns a list of ``(label, selector)`` where ``selector``
    maps the value array to a boolean mask; write such a list by hand for any
    selection that isn't a simple binning.
    """
    edges = [-np.inf] + [float(e) for e in np.atleast_1d(edges)] + [np.inf]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if np.isinf(lo):
            lab = f'{quantity} $\\leq$ {hi:{fmt}}'
        elif np.isinf(hi):
            lab = f'{quantity} > {lo:{fmt}}'
        else:
            lab = f'{lo:{fmt}} < {quantity} $\\leq$ {hi:{fmt}}'
        out.append((lab, partial(_in_bin, lo=lo, hi=hi)))
    return out


def _in_bin(v, lo, hi):
    return (v > lo) & (v <= hi)


def subsample_correlation_functions(maplog_tuples, property_tuples, subsamples, labels=None,
                                    row_names=None, abundance=5.01e-3, mult_thresh=None,
                                    rmin=2e-2, rmax=40.0, nbins=30, type_str='gg',
                                    max_p=1e7, nthreads=26, plot_full=True, plot_ratio=False,
                                    cmaps=('Blues_r', 'Greens_r', 'Purples_r', 'Oranges_r'),
                                    clo=0., chi=0.7,
                                    xscale='log', ylim=None, sharey=False, panel_size=(4.5, 3.5),
                                    save=False, str_identifier='',
                                    cross=None, matter_pos_tuples=None):
    """Correlation functions of MAP subsamples split on a merger-history property.

    For every maplog, the final-timeslice MAPs are abundance-matched as in
    ``compare_correlation_functions`` (multiplicity cut giving ``abundance``, or
    ``mult_thresh`` directly), *then* that sample is split by each entry of
    ``subsamples`` using the maplog's ``MapHistoryProperty``, and the
    auto-correlation function of each subsample is computed. So the subsamples
    are subsets of one number-density-matched catalog (their own densities are
    lower), which is what lets their clustering be compared to each other and
    to the whole catalog (``plot_full``).

    The property describes each *final* MAP's history, and a MAP at an earlier
    timeslice cannot be linked to it, so this only works at the property's own
    final timeslice (``prop.step``/``prop.redshift``).

    Layout: one row per tuple of ``maplog_tuples`` and one column per subsample
    (titled by its label), so each panel compares the maplogs of a tuple on the
    same population. Panels share x; y is independent per panel unless ``sharey``. Within a row, each
    maplog keeps one shade of that row's colormap (``cmaps``/``clo``/``chi``, as in
    ``compare_correlation_functions``); its full catalog is the thinner dashed
    curve of the same shade (a flat gray line at 1 under ``plot_ratio``). Legend
    entries give each maplog's label and its subsample size out of the full
    abundance-matched catalog; one gray dashed entry marks the full-catalog curves.

    Parameters
    ----------
    maplog_tuples : sequence of tuple of dict
        One tuple per row, as for ``compare_correlation_functions``. Each maplog
        must hold ``control`` and ``mult`` (and ``pos`` for the xi) -- i.e. the
        default ``read_maplog_dir`` keys.
    property_tuples : sequence of tuple of MapHistoryProperty
        Same shape as ``maplog_tuples``; ``property_tuples[row][m]`` is the
        property of ``maplog_tuples[row][m]`` (``read_map_history_property`` on
        the same directory). Reusable across calls with different ``subsamples``.
    subsamples : sequence of (label, selector)
        Each ``selector(values) -> bool mask`` picks a subsample of the
        abundance-matched MAPs from ``prop.values`` (an array, or a dict of
        arrays for a multi-component property; see ``bin_subsamples`` for the
        common single-array case, or pass any callables). Subsamples need not
        partition the catalog. NaN (undefined) values fail ``>``/``<``
        comparisons, so such a MAP is in a bin only if the selector says so --
        e.g. a ``~mask`` complement picks it up.
    labels, row_names : as for ``compare_correlation_functions`` (``labels[row][m]``
        names a maplog in the legend; ``row_names[row]`` is the row's y-label).
    abundance, mult_thresh, rmin, rmax, nbins, type_str, max_p, nthreads :
        As for ``compare_correlation_functions``.
    plot_full : bool
        Also draw the full abundance-matched catalog (gray).
    plot_ratio : bool
        Plot each subsample's xi divided by the full catalog's (so the full
        catalog is the line at 1) instead of ``r^2 xi``. Requires ``plot_full``.
        The two are not independent samples, so the drawn error band, which treats
        them as independent, overstates the uncertainty on the ratio.
    sharey : bool
        If True, panels in a row share one y-axis (limits from the row's tallest
        curve) and are packed with no gap; default False gives each its own range.
    cmaps : sequence of str
        Per-row single-color colormaps, cycled by row; ``clo``/``chi`` are the
        fractions of each colormap sampled.

    cross : {None, 'full', 'matter'}
        None (default) computes each subsample's auto-correlation. ``'full'``
        cross-correlates each subsample with the whole abundance-matched catalog
        (the dashed "full" curve then stays the catalog's own auto-correlation).
        ``'matter'`` cross-correlates the subsamples *and* the full catalog with
        the matter positions of ``matter_pos_tuples``. Set ``type_str`` (y-label)
        to match, e.g. ``'g,all'`` or ``'gm'``.
    matter_pos_tuples : sequence of tuple of (N, 3) array, optional
        Same shape as ``maplog_tuples``; matter tracer positions [Mpc/h] for each
        maplog (e.g. ``read_subsample_dir(...)['pos']``). Required for ``cross='matter'``.

    Returns
    -------
    (fig, axes, data) : ``data[row][m]`` is a dict with ``'r'`` and, for each
        subsample label plus ``'full'``, ``(xi, poisson, N)`` -- ``N`` the
        number of MAPs in it -- so results can be re-plotted or saved.
    """
    nrows = len(maplog_tuples)
    assert len(property_tuples) == nrows, "property_tuples must match maplog_tuples (rows)"
    for row, (mls, pps) in enumerate(zip(maplog_tuples, property_tuples)):
        assert len(pps) == len(mls), f"property_tuples[{row}] must match maplog_tuples[{row}]"
    if labels is not None:
        assert len(labels) == nrows, "labels must have one tuple per maplog tuple (row)"
    if row_names is not None:
        assert len(row_names) == nrows, "row_names must have one entry per maplog tuple (row)"
    if plot_ratio and not plot_full:
        raise ValueError("plot_ratio=True needs plot_full=True (the reference is the full catalog)")

    r_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
    r       = r_edges[:-1] + np.diff(r_edges) / 2.0

    if cross not in (None, 'full', 'matter'):
        raise ValueError("cross must be None, 'full' or 'matter'")
    if cross == 'matter':
        assert matter_pos_tuples is not None and len(matter_pos_tuples) == nrows, \
            "cross='matter' needs matter_pos_tuples with one tuple per maplog tuple (row)"

    def _xi(pos, boxsize, pos2=()):
        twopt = get_xi(pos, r_edges, boxsize, pos2=pos2, max_p=max_p, nthreads=nthreads)
        return twopt.get_corr(return_sep=False), np.sqrt(twopt.D1D2.wcounts) / twopt.R1R2.wcounts

    data = []
    for row, (maplogs, props) in enumerate(zip(maplog_tuples, property_tuples)):
        row_data = []
        for m, (maplog, prop) in enumerate(zip(maplogs, props)):
            BoxSize = maplog['header']['BoxSize'] * u.Mpc
            final   = maplog['control'] == int(prop.step) * 4 + 3
            if final.sum() != len(prop):
                raise ValueError(
                    "maplog_tuples[{0}][{1}] has {2} MAPs at step {3} but its property '{4}' "
                    "has {5}: it was built from a different run/read".format(
                        row, m, final.sum(), prop.step, prop.name, len(prop)))
            mult = maplog['mult'][final]
            if not np.array_equal(mult, prop.mult):
                raise ValueError(
                    "maplog_tuples[{0}][{1}]: property '{2}' (from {3}) is not aligned with "
                    "this maplog's final-timeslice MAPs (multiplicities differ)".format(
                        row, m, prop.name, prop.source))
            pos    = maplog['pos'][final, :]
            thresh = mult_thresh if mult_thresh is not None else \
                     get_threshold(mult, BoxSize, guess=1e1, abundance=abundance)
            above  = mult >= thresh
            d = {'r': r}
            matter = matter_pos_tuples[row][m] if cross == 'matter' else ()
            d['full'] = (*_xi(pos[above], BoxSize, matter), int(above.sum()))
            vals  = list(prop.values.values()) if isinstance(prop.values, dict) else [prop.values]
            n_nan = int(np.sum(above & ~np.all([np.isfinite(v) for v in vals], axis=0)))
            print(f"maplog_tuples[{row}][{m}]: {above.sum()} MAPs above threshold "
                  f"(mult >= {np.squeeze(thresh):.4g}); {n_nan} have an undefined '{prop.name}' "
                  f"(NaN: fails any comparison in a selector)")
            for lab, selector in subsamples:
                sel = above & np.asarray(selector(prop.values), dtype=bool)
                print(f"    {lab}: {sel.sum()} MAPs")
                pos2 = pos[above] if cross == 'full' else matter
                d[lab] = (*_xi(pos[sel], BoxSize, pos2), int(sel.sum()))
            row_data.append(d)
        data.append(row_data)

    ncols = len(subsamples)
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey='row' if sharey else False,
                             squeeze=False,
                             figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
    fig.subplots_adjust(hspace=0, wspace=0 if sharey else 0.3)
    ymax = np.zeros((nrows, ncols))
    for row, row_data in enumerate(data):
        cmap  = colormaps[cmaps[row % len(cmaps)]]
        fracs = clo + (chi - clo) * np.linspace(0, 1, max(len(row_data), 1))
        for col, (lab, _) in enumerate(subsamples):
            ax = axes[row, col]
            if plot_full and plot_ratio:
                ax.axhline(1.0, ls='--', lw=2, color='gray', label='all MAPs')
            if plot_full and not plot_ratio:
                # one proxy entry stands for every dashed full-catalog curve in the panel
                ax.plot([], [], '--', lw=1.5, color='gray', label='all MAPs')
            for m, d in enumerate(row_data):
                color = cmap(fracs[m])
                xi, poisson, n = d[lab]
                n_full = d['full'][2]
                # sim label, then this subsample's size out of the whole abundance-matched catalog
                legend = f'{labels[row][m]} ' if labels is not None else ''
                legend += f'(N={n} of {n_full})'
                if plot_full and not plot_ratio:
                    xi_f, poisson_f, _ = d['full']
                    y_f, _ = _xi_plot_values(r, xi_f, poisson_f)
                    ax.plot(r, y_f, '--', lw=1.5, color=color, alpha=0.6)
                    ymax[row, col] = max(ymax[row, col], np.nanmax(y_f))
                y, yerr = _xi_plot_values(r, xi, poisson, plot_ratio,
                                          d['full'][:2] if plot_ratio else None)
                ax.plot(r, y, '-', lw=2, color=color, label=legend)
                ax.fill_between(r, y - yerr, y + yerr, color=color, alpha=0.3)
                if not plot_ratio:
                    ymax[row, col] = max(ymax[row, col], np.nanmax(y + yerr))
            ax.set(xscale=xscale)
            ax.grid()
            ax.legend(fontsize='small')
            if row == 0:
                ax.set_title(lab)
    for row in range(nrows):
        for col in range(ncols):
            if ylim is not None:
                axes[row, col].set_ylim(ylim)
            elif not plot_ratio:
                # shared: setting one panel sets the row, so use the row's max
                top = ymax[row].max() if sharey else ymax[row, col]
                axes[row, col].set_ylim(0, top * 1.05)
        ylab = (r'$\xi_{%s}/\xi_{%s,\rm all}$' % (type_str, type_str)) if plot_ratio \
               else r'$r^2\xi_{%s}$ [$h^{-2}Mpc^2$]' % type_str
        if row_names is not None:
            ylab = f'{row_names[row]}\n{ylab}'
        axes[row, 0].set_ylabel(ylab, fontsize='small' if row_names is not None else None)
    fig.suptitle(f'{property_tuples[0][0].name}, z = {property_tuples[0][0].redshift:.2f}', y=1.0, va='bottom')   # above the column titles
    for ax in axes[-1, :]:
        ax.set_xlabel(r'r [$h^{-1}Mpc$]')

    if save:
        fig.savefig('../Images/subsample_correlation_functions_n{:.2e}_'.format(abundance)
                    + str_identifier + _ratio_suffix(plot_ratio) + '.png',
                    bbox_inches='tight', transparent=True)
    plt.show()
    return fig, axes, data


def examine_correlation_functions(maplogs, redshifts, abundances, labels=None,
                                  reference_maplog=None, reference_label='',
                                  ztol=5e-2,
                                  rmin=2e-2, rmax=40.0, nbins=30, type_str='gg',
                                  max_p=1e7, nthreads=26, cross_min_mult=1e1,
                                  cmap='viridis', clo=0., chi=0.7, panel_size=(4, 3),
                                  save=False, str_identifier='',
                                  plot_ratio=False, xscale='log'):
    """Examine MAP auto- and cross-correlation functions across abundance/redshift.

    Builds an ``(len(abundances) + 1) x len(redshifts)`` grid of subplots where
    rows correspond to abundance and columns to redshift, so every subplot in a
    row probes the same number density (comparable clustering scales) and every
    subplot in a column probes the same epoch. Each of the first ``len(abundances)``
    rows plots the MAP auto-correlation function of every maplog series in
    ``maplogs`` at that row's abundance and column's redshift (each maplog
    resolves the redshift to its own nearest timeslice step via
    ``map_step_for_redshift``, see ``maplogs`` below). The abundance-to-multiplicity
    mapping is resolved once per series at the *lowest* of the requested
    ``redshifts`` and that multiplicity cutoff is then held fixed across every
    column, so a row tracks the same population of MAPs across time rather than
    re-matching the abundance (and picking a different, drifting population) at
    each redshift. A final row is appended showing, for each series, the
    cross-correlation between the "large" and "small" MAPs on either side of the
    single most exclusive cut -- the multiplicity threshold for
    ``min(abundances)`` (the rarest, largest-MAP population requested), likewise
    fixed at the lowest requested redshift. MAPs with ``mult >= threshold`` are
    "large"; MAPs with ``cross_min_mult <= mult < threshold`` are "small"
    (``cross_min_mult`` excludes spurious/tiny MAPs from the small population).
    One column per redshift, as in the other rows.

    Every maplog is drawn in a shade of a single colormap (``cmap``), sampled on
    ``[clo, chi]`` to avoid its near-white end; a given maplog keeps the same
    shade in every panel (both auto-correlation rows and the cross-correlation
    row), so curves are identifiable across the whole grid. The subplots share
    the x-axis across the whole grid and are packed with no spacing between them;
    the y-axis is shared only within a row, so each row's limits are set from the
    tallest curve in that row rather than the tallest curve anywhere in the grid.

    If ``reference_maplog`` is given, its correlation function (auto in the
    abundance rows, cross in the final row) is drawn as a gray dashed reference
    in every subplot -- the single result every other maplog is compared against.

    Each row's right-hand label shows the multiplicity cutoff (``'m >= <cutoff>'``)
    that row's abundance resolved to at the lowest requested redshift, rather than
    the abundance itself. The value shown is ``reference_maplog``'s cutoff if a
    reference is given, else ``maplogs[0]``'s.

    Parameters
    ----------
    maplogs : sequence of dict
        One entry per curve to overplot. Each entry is a maplog (output of
        ``read_maplog_dir``) whose Log tree already spans every requested
        redshift -- a single file holds every ``P.MapLogRedshifts`` epoch (see
        ``doc/claude/reference_maplog.md`` sec. 6.2), so unlike the old
        one-directory-per-redshift layout there is no need to read (or pass)
        a separate maplog per column. Each column looks up its nearest
        available timeslice within that one maplog via
        ``map_step_for_redshift``.
    redshifts : float or sequence of float
        Target redshift(s); one column per entry. Each is resolved to its
        nearest available MAP timeslice step independently in every maplog
        (including ``reference_maplog``); a mismatch beyond ``ztol`` prints a
        warning but still plots the nearest available timeslice.
    abundances : sequence of float
        Target number densities (Mpc/h)^-3; one row per entry, plus a final
        large/small MAP cross-correlation row that cuts at ``min(abundances)``
        (see ``cross_min_mult``).
    labels : sequence of str, optional
        One label per entry of ``maplogs``, used as the legend label of that
        series' curve (same in every row).
    reference_maplog : dict, optional
        A single maplog (same shape as one entry of ``maplogs``) whose
        correlation function is drawn as a gray dashed reference in every
        subplot -- the result all others are compared against.
    reference_label : str
        Legend label for the ``reference_maplog`` gray dashed curve.
    ztol : float
        Max redshift difference between a requested redshift and the closest
        available MAP timeslice in any maplog (including ``reference_maplog``)
        (default 5e-2, i.e. agreement to one decimal place). A request with no
        match within ``ztol`` prints a warning (identifying which maplog) and
        proceeds using the nearest available timeslice anyway, since two
        maplogs being compared should ideally land on the same redshift, but a
        mismatch is not necessarily fatal to the comparison.
    rmin, rmax, nbins : float, float, int
        Log-spaced separation binning (Mpc/h) for the pair counts.
    type_str : str
        Correlation-function subscript in the y-axis label (e.g. 'gg').
    max_p, nthreads :
        Passed through to ``get_xi``.
    cross_min_mult : float
        Minimum MAP multiplicity included in the final row's "small" population
        (excludes spurious/tiny MAPs); passed as ``min_mult`` to
        ``get_map_large_small_cross_xi``.
    cmap : str or Colormap
        Colormap used to color the maplog curves; each maplog gets a distinct,
        fixed shade (see ``clo``/``chi``).
    clo, chi : float
        Lower/upper fraction of ``cmap`` to sample (avoids near-white ends).
    panel_size : tuple
        (width, height) in inches of a single subplot; the figure scales with the
        grid shape.
    save : bool
        If True, save the figure under Images/.
    str_identifier : str
        Filename suffix used when ``save`` is True.
    plot_ratio : bool
        If True, plot every curve as its ratio to ``reference_maplog`` of that
        panel instead of ``r^2 xi(r)`` next to it. The reference itself is then
        drawn as a horizontal line at 1. Requires ``reference_maplog``.
    xscale : {'log', 'linear'}
        Scale of the x-axis (separation ``r``).

    Returns
    -------
    (fig, axes) : the Figure and the 2D ndarray of Axes.
    """
    redshifts  = np.atleast_1d(redshifts).astype(float)
    abundances = np.atleast_1d(abundances).astype(float)
    ncols = len(redshifts)
    nrows = len(abundances) + 1  # + 1 for the large/small MAP cross-correlation row

    if labels is not None:
        assert len(labels) == len(maplogs), \
            "labels must have one entry per maplog"

    if plot_ratio and reference_maplog is None:
        raise ValueError("plot_ratio=True requires reference_maplog to provide the "
                         "reference correlation function")

    r_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
    r       = r_edges[:-1] + np.diff(r_edges) / 2.0

    # the single most exclusive cut requested (rarest, largest-MAP population);
    # the final row's large/small split is made against this one threshold
    cross_abundance = float(np.min(abundances))

    # Each row's multiplicity cut is fixed once, per series, at the lowest
    # requested redshift (where it matches that row's target abundance) and then
    # reused unchanged at every other redshift -- rather than re-deriving an
    # abundance-matched threshold at each column, which would let the selected
    # population drift with a MAP's own growth/accretion history instead of
    # tracking the same objects across time.
    z_min = float(np.min(redshifts))

    def _resolve_step(maplog, z_req, context):
        """``map_step_for_redshift`` plus the ``ztol`` mismatch check."""
        ts, zm = map_step_for_redshift(maplog, z_req)
        if abs(zm - z_req) > ztol:
            print(
                "WARNING: requested redshift {:.3f} has no timeslice within {:g} "
                "(one decimal place) in {}; using nearest available z={:.4f}".format(
                    z_req, ztol, context, zm))
        return ts

    def _low_z_mult_thresh(maplog, abundance, guess=1e1, context=''):
        ts_low        = _resolve_step(maplog, z_min, context)
        timeslice_low = maplog['control'] == int(ts_low) * 4 + 3
        return float(get_threshold(maplog['mult'][timeslice_low], maplog['header']['BoxSize'] * u.Mpc,
                                   guess=guess, abundance=abundance)[0])

    def mult_cutoff_label(thresh):
        return f'm $\\geq$ {thresh:.3g}'

    # one fixed shade per maplog, held constant across every row and column
    cmap_obj = colormaps[cmap] if isinstance(cmap, str) else cmap
    n_map    = len(maplogs)
    fracs    = clo + (chi - clo) * np.linspace(0, 1, max(n_map, 1))
    colors   = [cmap_obj(f) for f in fracs]

    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey='row', squeeze=False,
                             figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
    fig.subplots_adjust(wspace=0, hspace=0)

    # Panels share a y-axis only within their row, so the y-limit for a row has
    # to account for the tallest curve in *that row* (any column), not just
    # whichever panel happens to set it. Track the running max per row here and
    # apply it once after every panel is drawn instead of autoscaling off of
    # column 0 alone.
    data_ymax = np.zeros(nrows)

    for row, abundance in enumerate(abundances):
        ref_thresh  = _low_z_mult_thresh(reference_maplog, abundance, context='reference_maplog') \
                      if reference_maplog is not None else None
        map_threshs = [_low_z_mult_thresh(maplog, abundance, context='maplogs[{}]'.format(m))
                       for m, maplog in enumerate(maplogs)]

        for col, z_req in enumerate(redshifts):
            ax  = axes[row][col]
            ref = None

            if reference_maplog is not None:
                ts_ref = _resolve_step(reference_maplog, z_req, 'reference_maplog')
                xi_ref, poisson_ref = get_map_xi(reference_maplog, ts_ref, r_edges,
                                                 mult_thresh=ref_thresh, max_p=max_p,
                                                 nthreads=nthreads)
                ref = (xi_ref, poisson_ref)
                if plot_ratio:
                    ax.axhline(1.0, lw=2, ls='--', color='gray', label=reference_label)
                else:
                    y_d, yerr_d = _xi_plot_values(r, xi_ref, poisson_ref, plot_ratio, ref)
                    ax.plot(r, y_d, '--', lw=2, color='gray', label=reference_label)
                    ax.fill_between(r, y_d - yerr_d, y_d + yerr_d, color='gray', alpha=0.3)
                    data_ymax[row] = max(data_ymax[row], np.nanmax(y_d + yerr_d))

            for m, maplog in enumerate(maplogs):
                ts          = _resolve_step(maplog, z_req, 'maplogs[{}]'.format(m))
                xi, poisson = get_map_xi(maplog, ts, r_edges, mult_thresh=map_threshs[m],
                                         max_p=max_p, nthreads=nthreads)
                label = str(labels[m]) if labels is not None else ''
                y, yerr = _xi_plot_values(r, xi, poisson, plot_ratio, ref)
                ax.plot(r, y, '-', lw=2, color=colors[m], label=label)
                ax.fill_between(r, y - yerr, y + yerr, color=colors[m], alpha=0.3)
                if not plot_ratio:
                    data_ymax[row] = max(data_ymax[row], np.nanmax(y + yerr))

            ax.set(xscale=xscale)
            ax.grid()
            if col == 0:
                ax.legend(fontsize='small')
            if row == 0:
                ax.set_title(f'z = {z_req:.2f}')
            if col == ncols - 1:
                axr = ax.twinx()
                axr.set_yticks([])
                label_thresh = ref_thresh if ref_thresh is not None else map_threshs[0]
                axr.set_ylabel(mult_cutoff_label(label_thresh), rotation=270, labelpad=15)

    # --- final row: cross-correlation of the "large" MAPs (above the
    # cross_abundance cut) against the "small" MAPs (below it, but still above
    # cross_min_mult), one curve per maplog ---
    cross_row = nrows - 1
    ref_cross_thresh  = _low_z_mult_thresh(reference_maplog, cross_abundance, guess=cross_min_mult,
                                           context='reference_maplog') \
                        if reference_maplog is not None else None
    map_cross_threshs = [_low_z_mult_thresh(maplog, cross_abundance, guess=cross_min_mult,
                                            context='maplogs[{}]'.format(m))
                          for m, maplog in enumerate(maplogs)]

    for col, z_req in enumerate(redshifts):
        ax  = axes[cross_row][col]
        ref = None

        if reference_maplog is not None:
            ts_ref = _resolve_step(reference_maplog, z_req, 'reference_maplog')
            xi_ref, poisson_ref = get_map_large_small_cross_xi(
                reference_maplog, ts_ref, r_edges, cross_abundance, min_mult=cross_min_mult,
                mult_thresh=ref_cross_thresh, max_p=max_p, nthreads=nthreads)
            ref = (xi_ref, poisson_ref)
            if plot_ratio:
                ax.axhline(1.0, lw=2, ls='--', color='gray', label=reference_label)
            else:
                y_d, yerr_d = _xi_plot_values(r, xi_ref, poisson_ref, plot_ratio, ref)
                ax.plot(r, y_d, '--', lw=2, color='gray', label=reference_label)
                ax.fill_between(r, y_d - yerr_d, y_d + yerr_d, color='gray', alpha=0.3)
                data_ymax[cross_row] = max(data_ymax[cross_row], np.nanmax(y_d + yerr_d))

        for m, maplog in enumerate(maplogs):
            ts          = _resolve_step(maplog, z_req, 'maplogs[{}]'.format(m))
            xi, poisson = get_map_large_small_cross_xi(maplog, ts, r_edges, cross_abundance,
                                                       min_mult=cross_min_mult,
                                                       mult_thresh=map_cross_threshs[m],
                                                       max_p=max_p, nthreads=nthreads)
            label = str(labels[m]) if labels is not None else ''
            y, yerr = _xi_plot_values(r, xi, poisson, plot_ratio, ref)
            ax.plot(r, y, '-', lw=2, color=colors[m], label=label)
            ax.fill_between(r, y - yerr, y + yerr, color=colors[m], alpha=0.3)
            if not plot_ratio:
                data_ymax[cross_row] = max(data_ymax[cross_row], np.nanmax(y + yerr))

        ax.set(xscale=xscale)
        ax.grid()
        if col == 0:
            ax.legend(fontsize='small')
        if col == ncols - 1:
            axr = ax.twinx()
            axr.set_yticks([])
            label_thresh = ref_cross_thresh if ref_cross_thresh is not None else map_cross_threshs[0]
            axr.set_ylabel(mult_cutoff_label(label_thresh), rotation=270, labelpad=15)

    # Each row shares a y-axis only with itself (sharey='row'), so the limit
    # is set per row from the actual max plotted across that row's columns,
    # rather than a single grid-wide limit that could clip a row whose
    # amplitude differs from the others.
    for row in range(nrows):
        if not plot_ratio:
            axes[row, 0].set_ylim(0, data_ymax[row] * 1.05)

    # shared axis labels on the outer edge only. The last row is the
    # large x small MAP cross-correlation, not the auto-correlation every
    # other row plots, so it gets its own distinct y-axis label.
    for ax in axes[-1, :]:
        ax.set_xlabel(r'r [$h^{-1}Mpc$]')
    if plot_ratio:
        for ax in axes[:-1, 0]:
            ax.set_ylabel(r'$\xi_{%s}/\xi_{%s,\rm ref}$' % (type_str, type_str))
        axes[cross_row, 0].set_ylabel(r'$\xi_{L\times S}/\xi_{L\times S,\rm ref}$')
    else:
        for ax in axes[:-1, 0]:
            ax.set_ylabel(r'$r^2\xi_{%s}(r)$' % type_str)
        axes[cross_row, 0].set_ylabel(r'$r^2\xi_{L\times S}(r)$')

    if save:
        fig.savefig('../Images/examine_correlation_functions_' + str_identifier
                    + _ratio_suffix(plot_ratio) + '.png',
                    bbox_inches='tight', transparent=True)
    plt.show()
    return fig, axes


def compare_correlation_functions_flamingo(maplog_tuples, redshifts, flamingo_snapshots,
                                            root_dir=None, labels=None,
                                            plot_flamingo=True, flamingo_label='FLAMINGO',
                                            flamingo_basePath=FLAMINGO_L1_M8_BASEPATH,
                                            cache_dir=FLAMINGO_CACHE_DIR, ztol=5e-3,
                                            rmin=2e-2, rmax=40.0, nbins=30, type_str='gg',
                                            abundance=5.01e-3, max_p=np.inf, nthreads=26,
                                            cmaps=('Blues_r', 'Greens_r', 'Purples_r', 'Oranges_r'),
                                            clo=0., chi=0.7, panel_size=(4, 3),
                                            save=False, str_identifier='',
                                            map_default=None, default_label='',
                                            plot_ratio=False, xscale='log'):
    """Compare MAP correlation functions across maplogs on a shared grid, against FLAMINGO.

    Identical in structure and behavior to ``compare_correlation_functions`` (same
    ``nrows x ncols`` grid, per-row colormap shading, shared axes, ``map_default``
    gray-dashed reference, ``save``/legend/title conventions) except the reference
    catalog is FLAMINGO (SOAP) halos instead of TNG.

    Like the TNG overlay in ``compare_correlation_functions``, FLAMINGO is
    time-dependent: each column resolves and loads its own snapshot, nearest that
    column's redshift (within ``ztol``), so the FLAMINGO curve evolves across
    columns rather than repeating one fixed snapshot everywhere. Snapshots are
    loaded via ``load_flamingo_halos`` (see ``compare_tng_flamingo_correlation_functions``
    for the same pattern applied to a TNG-vs-FLAMINGO-only grid), which caches each
    one to disk after its first network fetch, and each snapshot is loaded at most
    once even if reused across rows/columns.

    Parameters
    ----------
    maplog_tuples, redshifts, labels :
        Same as ``compare_correlation_functions``.
    flamingo_snapshots : dict {float: int}
        Maps an available FLAMINGO redshift to its snapshot step number, e.g.
        ``{5: 10, 4: 14, 3: 18, 2: 38, 1: 58, 0.5: 68, 0.2: 74}`` (see
        ``load_flamingo_halos``). Only used if ``plot_flamingo``.
    root_dir :
        Open hdfstream root directory (``hdfstream.open('cosma', '/')``); only
        needed for FLAMINGO snapshots not already cached on disk (see
        ``load_flamingo_halos``).
    plot_flamingo : bool
        If True (default), overplot the FLAMINGO correlation function (red dashed)
        in every subplot.
    flamingo_label : str
        Legend label for the FLAMINGO curve.
    flamingo_basePath : str
        hdfstream directory holding the FLAMINGO SOAP catalogs; passed through to
        ``load_flamingo_halos``.
    cache_dir :
        Passed through to ``load_flamingo_halos``.
    ztol : float
        Max redshift difference between a requested redshift and the closest
        available FLAMINGO snapshot. A request with no snapshot within ``ztol``
        raises.
    rmin, rmax, nbins : float, float, int
        Log-spaced separation binning (Mpc/h) for the pair counts.
    type_str : str
        Correlation-function subscript in the y-axis label (e.g. 'gg').
    abundance : float
        Target number density (Mpc/h)^-3 used by ``get_threshold`` on MAP
        multiplicity / FLAMINGO mass.
    max_p, nthreads :
        Passed through to ``get_xi``.
    cmaps, clo, chi, panel_size, save, str_identifier :
        Same as ``compare_correlation_functions``.
    map_default : dict, optional
        Same as ``compare_correlation_functions``.
    default_label : str
        Legend label for the ``map_default`` gray dashed curve.
    plot_ratio : bool
        If True, plot every curve as its ratio to the FLAMINGO correlation
        function of that column instead of ``r^2 xi(r)`` next to it. The
        FLAMINGO curve is then drawn as a horizontal line at 1 instead of its
        own curve. Requires ``plot_flamingo=True``.
    xscale : {'log', 'linear'}
        Scale of the x-axis (separation ``r``).

    Returns
    -------
    (fig, axes) : the Figure and the 2D ndarray of Axes.
    """
    redshifts = np.atleast_1d(redshifts).astype(float)
    nrows = len(maplog_tuples)
    ncols = len(redshifts)

    if labels is not None:
        assert len(labels) == nrows, "labels must have one tuple per maplog tuple (row)"
        for row, (mls, labs) in enumerate(zip(maplog_tuples, labels)):
            assert len(labs) == len(mls), \
                "labels[{}] must match the number of maplogs in row {}".format(row, row)

    if plot_ratio and not plot_flamingo:
        raise ValueError("plot_ratio=True requires plot_flamingo=True to provide "
                         "the reference FLAMINGO correlation function")

    r_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
    r       = r_edges[:-1] + np.diff(r_edges) / 2.0

    # The gray dashed reference (map_default) is the single result everything is
    # compared against. Its correlation function depends only on the column
    # redshift, so compute it once per column here and reuse it across all rows.
    default_xi = default_poisson = None
    if map_default is not None:
        default_xi      = [None] * ncols
        default_poisson = [None] * ncols
        for col, z_req in enumerate(redshifts):
            ts, _ = map_step_for_redshift(map_default, z_req)
            default_xi[col], default_poisson[col] = get_map_xi(
                map_default, ts, r_edges, abundance=abundance,
                max_p=max_p, nthreads=nthreads)

    # FLAMINGO is resolved to its own snapshot per column (like the TNG overlay
    # in compare_correlation_functions), so its correlation function depends on
    # the column redshift: compute it once per column here and reuse it across
    # all rows rather than recomputing it in every subplot.
    flamingo_xi = None
    if plot_flamingo:
        flamingo_zs = np.array(sorted(flamingo_snapshots.keys()))
        flamingo_xi = [None] * ncols
        for col, z_req in enumerate(redshifts):
            j = np.argmin(np.abs(flamingo_zs - z_req))
            if abs(flamingo_zs[j] - z_req) > ztol:
                raise ValueError(
                    "requested redshift {:.3f} has no FLAMINGO snapshot within {:g}; "
                    "available FLAMINGO redshifts are {}".format(
                        z_req, ztol, ', '.join('{:.2f}'.format(z) for z in flamingo_zs)))
            step             = flamingo_snapshots[float(flamingo_zs[j])]
            flam             = load_flamingo_halos(step, root_dir=root_dir,
                                                    basePath=flamingo_basePath, cache_dir=cache_dir)
            flamingo_boxsize = flam['boxsize'] * u.Mpc
            flamingo_thresh  = get_threshold(flam['mass'], flamingo_boxsize, guess=1e-3,
                                             abundance=abundance)
            msk_flamingo     = flam['mass'] >= flamingo_thresh
            twopt            = get_xi(flam['pos'][msk_flamingo, :], r_edges, flamingo_boxsize,
                                      max_p=max_p, nthreads=nthreads)
            xi_flamingo      = twopt.get_corr(return_sep=False)
            poisson_flamingo = np.sqrt(twopt.D1D2.wcounts) / twopt.R1R2.wcounts
            flamingo_xi[col] = (xi_flamingo, poisson_flamingo)

    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False,
                             figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
    fig.subplots_adjust(wspace=0, hspace=0)

    # All panels share one y-axis, so the y-limit has to account for the
    # tallest curve in *any* row/column, not just whichever panel happens to
    # set it. Track the running max here and apply it once after every panel
    # is drawn instead of autoscaling off of row 0 alone.
    data_ymax = 0.0

    for row, maplogs in enumerate(maplog_tuples):
        cmap   = colormaps[cmaps[row % len(cmaps)]]
        n_map  = len(maplogs)  # curves drawn from the colormap
        # one shade per maplog, held constant across the columns
        fracs  = clo + (chi - clo) * np.linspace(0, 1, max(n_map, 1))
        for col, z_req in enumerate(redshifts):
            ax = axes[row][col]
            ref = flamingo_xi[col] if plot_ratio else None

            # gray dashed reference, precomputed once per column
            if map_default is not None:
                xi_d, poisson_d = default_xi[col], default_poisson[col]
                y_d, yerr_d = _xi_plot_values(r, xi_d, poisson_d, plot_ratio, ref)
                ax.plot(r, y_d, '--', lw=2, color='gray', label=default_label)
                ax.fill_between(r, y_d - yerr_d, y_d + yerr_d, color='gray', alpha=0.3)
                if not plot_ratio:
                    data_ymax = max(data_ymax, np.nanmax(y_d + yerr_d))

            for m, maplog in enumerate(maplogs):
                ts, zm      = map_step_for_redshift(maplog, z_req)
                xi, poisson = get_map_xi(maplog, ts, r_edges, abundance=abundance,
                                         max_p=max_p, nthreads=nthreads)
                label = str(labels[row][m]) if labels is not None else ''
                color = cmap(fracs[m])
                y, yerr = _xi_plot_values(r, xi, poisson, plot_ratio, ref)
                ax.plot(r, y, '-', lw=2, color=color, label=label)
                ax.fill_between(r, y - yerr, y + yerr, color=color, alpha=0.3)
                if not plot_ratio:
                    data_ymax = max(data_ymax, np.nanmax(y + yerr))

            # --- optional FLAMINGO halos: mass-thresholded, precomputed once
            # per column above and reused across rows ---
            if plot_flamingo:
                xi_f, poisson_f = flamingo_xi[col]
                if plot_ratio:
                    # this is the reference curve itself: ratio to itself is 1
                    ax.axhline(1.0, lw=2, color='red', ls='--', label=flamingo_label)
                else:
                    y_f, yerr_f = _xi_plot_values(r, xi_f, poisson_f, plot_ratio, ref)
                    ax.plot(r, y_f, lw=2, color='red', ls='--', label=flamingo_label)
                    ax.fill_between(r, y_f - yerr_f, y_f + yerr_f, color='red', alpha=0.3)
                    data_ymax = max(data_ymax, np.nanmax(y_f + yerr_f))

            ax.set(xscale=xscale)
            ax.grid()
            if col == 0:
                ax.legend(fontsize='small')
            if row == 0:
                ax.set_title(f'z = {z_req:.2f}')

    # All panels share one y-axis (sharey=True), so this sets the range for
    # the whole grid. Computed from the actual max plotted across every
    # row/column rather than autoscaling off of row 0 alone.
    if plot_ratio:
        axes[0, 0].set_ylim(0.8, 1.2)
    else:
        axes[0, 0].set_ylim(0, data_ymax * 1.05)

    # shared axis labels on the outer edge only
    for ax in axes[-1, :]:
        ax.set_xlabel(r'r [$h^{-1}Mpc$]')
    if plot_ratio:
        for ax in axes[:, 0]:
            ax.set_ylabel(r'$\xi_{%s}/\xi_{%s,\rm FLAMINGO}$' % (type_str, type_str))

    if save:
        fig.savefig('../Images/compare_correlation_functions_flamingo_n{:.2e}_'.format(abundance)
                    + str_identifier + _ratio_suffix(plot_ratio) + '.png',
                    bbox_inches='tight', transparent=True)
    plt.show()
    return fig, axes


def compare_crosscorrelation_functions(mdir_tuples, redshifts, labels=None,
                                        plot_tng2=False,
                                        ztol=5e-3,
                                        rmin=2e-2, rmax=40.0, nbins=30, type_str='gm',
                                        abundance=5.01e-3, max_p=np.inf, nthreads=26,
                                        cmaps=('Blues_r', 'Greens_r', 'Purples_r', 'Oranges_r'),
                                        clo=0., chi=0.7, panel_size=(4, 3),
                                        save=False, str_identifier='',
                                        map_default=None, default_label='',
                                        subsample_type='P8', tng_dm_max_p=1e7,
                                        plot_ratio=False, xscale='log',
                                        show_redshift_in_legend=False,
                                        ylim=None, yticks=None):
    """Compare MAP x matter cross-correlation functions across runs on a shared grid.

    Identical in structure and behavior to ``compare_correlation_functions`` (same
    ``nrows x ncols`` grid, per-row colormap shading, shared axes, ``map_default``
    gray-dashed reference, optional TNG overlay, ``save``/legend/title conventions)
    except every curve is a cross-correlation with the matter field rather than an
    autocorrelation:

    - MAP positions are drawn from the fair particle subsample (``subsamples/z*``,
      via ``read_subsample_dir``) instead of the MapLog tree, because the subsample
      also contains an unbiased fair sample of *all* particles to use as the matter
      tracer (see ``get_map_matter_xi``); the MapLog table alone has no such matter
      field to cross with. Consequently each row's entries are run directories
      (``mdir``), not pre-loaded ``read_maplog_dir`` dicts.
    - The optional TNG overlay (``plot_tng2``) cross-correlates TNG300-2 galaxies
      (stellar-mass-thresholded subhalos, as in ``compare_correlation_functions``)
      against a downsampled set of TNG300-2 dark-matter particles obtained with
      ``get_particle_pos_tng``. There is no ``plot_tng1`` option here: the TNG300-1
      snapshot particle outputs are too large for the per-panel DM downsampling this
      needs.

    Parameters
    ----------
    mdir_tuples : sequence of tuple of (str or Path)
        One tuple per row; each tuple is a set of run directories (each containing
        ``subsamples/z*/...``) to overplot. Every entry is drawn as a colored curve.
    redshifts : float or sequence of float
        Target redshift(s); one column per entry.
    labels : sequence of tuple of str, optional
        Same shape as ``mdir_tuples``. ``labels[row][i]`` is prepended (before the
        redshift) to the legend label of the corresponding run's curve.
    plot_tng2 : bool
        If True, also overplot the TNG300-2 galaxy x dark-matter cross-correlation
        (red dashed) in every subplot, using the snapshot matched to the column's
        redshift. Red is reserved for TNG, which is why it is absent from ``cmaps``.
        Like ``map_default``, this depends only on the column (redshift), so it
        is computed once per column and reused across rows rather than
        recomputed in every subplot.
    ztol : float
        Max redshift difference between a requested redshift and the closest
        available TNG snapshot (only used when ``plot_tng2``); see
        ``compare_correlation_functions``.
    rmin, rmax, nbins : float, float, int
        Log-spaced separation binning (Mpc/h) for the pair counts.
    type_str : str
        Correlation-function subscript in the y-axis label (default ``'gm'`` for
        galaxy-matter, since every curve here is a cross-correlation).
    abundance : float
        Target number density (Mpc/h)^-3 used by ``get_threshold`` on MAP
        multiplicity / TNG stellar mass.
    max_p, nthreads :
        Passed through to ``get_xi``.
    cmaps, clo, chi, panel_size, save, str_identifier :
        Same as ``compare_correlation_functions``.
    map_default : str or Path, optional
        A single run directory whose MAP x matter cross-correlation is drawn as a
        gray dashed reference in every subplot. Its curve is computed once per
        column and reused across rows.
    default_label : str
        Legend label for the ``map_default`` gray dashed curve.
    show_redshift_in_legend : bool
        If True, append ``(z=...)`` (the requested redshift) to the legend label
        of every MAP curve and of ``map_default``. Default False: legends carry
        only the descriptive labels, and the redshift is in the column titles.
    subsample_type : str
        Subsample tag passed to ``read_subsample_dir`` (e.g. ``'P8'``).
    tng_dm_max_p : int
        Max number of TNG300-2 dark-matter particles to downsample to per panel
        (passed to ``get_particle_pos_tng``); only used when ``plot_tng2``.
    plot_ratio : bool
        If True, plot every curve as its ratio to a reference cross-correlation
        of that column instead of ``r^2 xi(r)`` next to it. The reference is the
        TNG300-2 curve if ``plot_tng2`` is enabled; otherwise ``map_default`` is
        used (as in ``compare_correlation_functions``). The reference curve itself
        is drawn as a horizontal line at 1 instead of its own curve (with dotted
        lines at 0.9/1.1 when it is ``map_default``). Requires ``plot_tng2`` or
        ``map_default``.
    xscale : {'log', 'linear'}
        Scale of the x-axis (separation ``r``).
    ylim : tuple (ymin, ymax), optional
        y-axis limits applied to every panel (they share a y-axis). Overrides the
        automatic limits, including the fixed ratio-mode range. Either entry may
        be None to leave that end automatic.
    yticks : sequence of float, optional
        y tick locations applied to every panel (they share a y-axis).

    Returns
    -------
    (fig, axes) : the Figure and the 2D ndarray of Axes.
    """
    redshifts = np.atleast_1d(redshifts).astype(float)
    nrows = len(mdir_tuples)
    ncols = len(redshifts)

    if labels is not None:
        assert len(labels) == nrows, "labels must have one tuple per run tuple (row)"
        for row, (dirs, labs) in enumerate(zip(mdir_tuples, labels)):
            assert len(labs) == len(dirs), \
                "labels[{}] must match the number of run directories in row {}".format(row, row)

    r_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
    r       = r_edges[:-1] + np.diff(r_edges) / 2.0

    # only TNG300-2: TNG300-1's snapshot particle outputs are too large for the
    # per-panel dark-matter downsampling this cross-correlation needs
    tng_specs = []
    if plot_tng2:
        tng_specs.append((TNG300_2_BASEPATH, 'TNG300-2', dict(color='red', ls='--')))

    if plot_ratio and not tng_specs and map_default is None:
        raise ValueError("plot_ratio=True requires plot_tng2 or map_default to provide "
                         "the reference cross-correlation function")
    ref_basePath = tng_specs[0][0] if tng_specs else None
    # with no TNG overlay, the ratio is taken to map_default instead
    ratio_uses_default = plot_ratio and not tng_specs

    # resolve each column redshift to a TNG snapshot (within ztol of an available one)
    star_type = il.snapshot.partTypeNum('star') if tng_specs else None
    tng_col_snap = {}  # basePath -> array of snapnos, one per column
    for basePath, _, _ in tng_specs:
        tng_snapnos, tng_zs = get_available_tng_snapshots(basePath)
        col_snap = np.empty(ncols, dtype=int)
        for col, z_req in enumerate(redshifts):
            k = np.argmin(np.abs(tng_zs - z_req))
            if abs(tng_zs[k] - z_req) > ztol:
                raise ValueError(
                    "requested redshift {:.3f} has no TNG snapshot within {:g} "
                    "(2 decimal places); available TNG redshifts for {} are {}".format(
                        z_req, ztol, basePath, ', '.join('{:.2f}'.format(z) for z in tng_zs)))
            col_snap[col] = int(tng_snapnos[k])
        tng_col_snap[basePath] = col_snap

    # The gray dashed reference (map_default) is the single result everything is
    # compared against. Its cross-correlation depends only on the column redshift,
    # so compute it once per column here and reuse it across all rows.
    default_xi = default_poisson = None
    if map_default is not None:
        default_xi      = [None] * ncols
        default_poisson = [None] * ncols
        for col, z_req in enumerate(redshifts):
            default_xi[col], default_poisson[col] = get_map_matter_xi(
                map_default, z_req, r_edges, abundance=abundance,
                max_p=max_p, nthreads=nthreads, subsample_type=subsample_type)

    # Each TNG comparison also depends only on the column redshift (not the
    # row), so compute its cross-correlation once per column here and reuse
    # it across all rows rather than recomputing it in every subplot.
    tng_xi = {}  # basePath -> [(xi_tng, poisson_tng), ...] one per column
    for basePath, _, _ in tng_specs:
        col_xi = [None] * ncols
        for col in range(ncols):
            sn           = int(tng_col_snap[basePath][col])
            tng_subhalos = load_tng_subhalos(basePath, sn)
            m_tng        = tng_subhalos['SubhaloMassType'][:, star_type]  # 1e10 Msun
            # native TNG box size (header is ckpc/h, SubhaloPos is scaled to Mpc/h)
            tng_boxsize  = il.groupcat.loadHeader(basePath, sn)['BoxSize'] / 1e3 * u.Mpc
            tng_thresh   = get_threshold(m_tng, tng_boxsize, guess=1e-3, abundance=abundance)
            msk_tng      = m_tng >= tng_thresh
            tng_dm_pos   = get_particle_pos_tng(basePath, sn, 'dm', max_p=tng_dm_max_p)
            twopt        = get_xi(tng_subhalos['SubhaloPos'][msk_tng, :], r_edges,
                                  tng_boxsize, pos2=tng_dm_pos, max_p=max_p, nthreads=nthreads)
            xi_tng       = twopt.get_corr(return_sep=False)
            poisson_tng  = np.sqrt(twopt.D1D2.wcounts) / twopt.R1R2.wcounts
            col_xi[col] = (xi_tng, poisson_tng)
        tng_xi[basePath] = col_xi

    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False,
                             figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
    fig.subplots_adjust(wspace=0, hspace=0)

    for row, mdirs in enumerate(mdir_tuples):
        cmap   = colormaps[cmaps[row % len(cmaps)]]
        n_dir  = len(mdirs)  # curves drawn from the colormap
        # one shade per run, held constant across the columns
        fracs  = clo + (chi - clo) * np.linspace(0, 1, max(n_dir, 1))
        for col, z_req in enumerate(redshifts):
            ax = axes[row][col]
            if ratio_uses_default:
                ref = (default_xi[col], default_poisson[col])
            else:
                ref = tng_xi[ref_basePath][col] if plot_ratio else None
            z_suffix = f' (z={z_req:.3f})' if show_redshift_in_legend else ''

            # gray dashed reference, precomputed once per column
            if map_default is not None:
                if ratio_uses_default:
                    # ratio-to-itself is exactly 1: draw a flat reference line
                    ax.axhline(1.0, ls='--', lw=2, color='gray', label=default_label + z_suffix)
                    ax.axhline(0.9, ls=':', lw=2.5, color='gray', alpha=0.8)
                    ax.axhline(1.1, ls=':', lw=2.5, color='gray', alpha=0.8)
                else:
                    xi_d, poisson_d = default_xi[col], default_poisson[col]
                    y_d, yerr_d = _xi_plot_values(r, xi_d, poisson_d, plot_ratio, ref)
                    ax.plot(r, y_d, '--', lw=2, color='gray', label=default_label + z_suffix)
                    ax.fill_between(r, y_d - yerr_d, y_d + yerr_d, color='gray', alpha=0.3)

            for m, mdir in enumerate(mdirs):
                xi, poisson = get_map_matter_xi(mdir, z_req, r_edges, abundance=abundance,
                                                max_p=max_p, nthreads=nthreads,
                                                subsample_type=subsample_type)
                label = (str(labels[row][m]) if labels is not None else '') + z_suffix
                color = cmap(fracs[m])
                y, yerr = _xi_plot_values(r, xi, poisson, plot_ratio, ref)
                ax.plot(r, y, '-', lw=2, color=color, label=label)
                ax.fill_between(r, y - yerr, y + yerr, color=color, alpha=0.3)

            # --- optional TNG: galaxies cross dark matter, precomputed once
            # per column above and reused across rows ---
            for basePath, tng_label, style in tng_specs:
                xi_tng, poisson_tng = tng_xi[basePath][col]
                if plot_ratio and basePath == ref_basePath:
                    ax.axhline(1.0, lw=2, label=tng_label, **style)
                    continue
                y_t, yerr_t = _xi_plot_values(r, xi_tng, poisson_tng, plot_ratio, ref)
                ax.plot(r, y_t, lw=2, label=tng_label, **style)
                ax.fill_between(r, y_t - yerr_t, y_t + yerr_t, color=style['color'], alpha=0.3)

            ax.set(xscale=xscale)
            ax.grid()
            if col == 0:
                ax.legend(fontsize='small')
            if row == 0:
                ax.set_title('z = {:.2f}'.format(z_req))

    # shared axis labels on the outer edge only
    for ax in axes[-1, :]:
        ax.set_xlabel(r'r [Mpc/h]')
    for ax in axes[:, 0]:
        if plot_ratio:
            ref_name = 'ref' if ratio_uses_default else 'TNG'
            ax.set_ylabel(r'$\xi_{%s}/\xi_{%s,\rm %s}$' % (type_str, type_str, ref_name))
            ax.set_ylim([0.7, 1.3] if ratio_uses_default else [0.8, 1.2])
        else:
            ax.set_ylabel(r'$r^2\xi_{%s}(r)$' % type_str)
    if ylim is not None:
        axes[0, 0].set_ylim(ylim)   # y is shared across all panels
    if yticks is not None:
        axes[0, 0].set_yticks(yticks)

    if save:
        fig.savefig('../Images/compare_crosscorrelation_functions_n{:.2e}_'.format(abundance)
                    + str_identifier + _ratio_suffix(plot_ratio) + '.png',
                    bbox_inches='tight', transparent=True)
    plt.show()
    return fig, axes


def apply_zspace_poles(pos, vel, BoxSize, z, smu_edges, ells=(0, 2, 4),
                       H0=67.74, Omega_M=0.3089, Omega_DE=0.6911,
                       max_p=np.inf, nthreads=26):
    """Apply line-of-sight (z) redshift-space distortions and project to multipoles.

    Mirrors the "projected" cells of analysis.ipynb: the peculiar z-velocity is
    converted to a comoving (Mpc/h) line-of-sight shift ``v_z (1+z) / H(z) * h`` and
    added to the z position, then the redshift-space catalog is correlated in
    ``(s, mu)`` mode and projected onto the Legendre multipoles ``ells``.

    ``H(z) = H0 * sqrt((1+z)^3 Omega_M + Omega_DE)`` in km/s/Mpc and ``h = H0/100``,
    matching the ``HubbleNow`` based shift used in the notebook.

    Returns ``(s, poles)`` where ``poles`` has shape ``(len(ells), nbins)``.
    """
    Hz = H0 * np.sqrt((1.0 + z)**3 * Omega_M + Omega_DE)  # km/s/Mpc
    h  = H0 / 100.0
    x  = np.copy(pos)
    x[:, 2] += vel[:, 2]* (1.0+z) / Hz * h #convert to comoving 
    twopt = get_xi(x, smu_edges, BoxSize, mode='smu', max_p=max_p, nthreads=nthreads)
    s, poles = project_to_multipoles(twopt, ells=ells, return_sep=True, return_cov=False)
    return s, poles


def read_gm_dict_for_maplog(maplog, redshift,
                            template='../Images/{sim}_tng/gm_dict_z{z:.3f}.pkl'):
    """Load the MAP -> TNG halo match (``gm_dict``) belonging to a maplog.

    The run's name is taken from the maplog's own header (``SimName``), so callers
    never carry gm_dicts around next to their maplogs: the file is
    ``template.format(sim=SimName, z=redshift)`` (``match_maps_to_tng`` output).
    """
    path = template.format(sim=maplog['header']['SimName'], z=redshift)
    if not Path(path).is_file():
        raise FileNotFoundError(f"{path} not found; make the match first with match_maps_to_tng")
    with open(path, 'rb') as f:
        return pickle.load(f)


def map_satellite_mask(pids, mult, gm_dict):
    """Which of a set of MAPs are satellites, from the MAP -> TNG halo match ``gm_dict``.

    ``pids``/``mult`` are the pids and multiplicities of the MAPs under
    consideration (already abundance-thresholded). Within each matched halo
    (``{group_id: MapGalaxyRec}``, see ``match_maps_to_tng``), the
    highest-multiplicity MAP among them is the *central* and the rest are
    *satellites* -- the convention of ``halowise_map_galaxies``. A MAP in no
    matched halo is not a satellite (so it counts with the centrals).
    Returns a boolean array like ``pids``.
    """
    pid2idx = dict(zip(pids, range(len(pids))))
    is_sat  = np.zeros(len(pids), dtype=bool)
    for rec in gm_dict.values():
        idx = np.array([pid2idx[mp] for mp in rec.map_pids if mp in pid2idx], dtype=int)
        if idx.size > 1:
            is_sat[idx] = True
            is_sat[idx[np.argmax(mult[idx])]] = False
    return is_sat


def map_rsd_velocity(maplog, msk, gm_dict=None):
    """Velocity [km/s] used to shift the MAPs selected by ``msk`` into redshift space.

    By default every MAP uses the dark-matter velocity estimate
    ``vel_dm = vel_map + rel_vel_vector`` (``rel_vel_vector`` reconstructed from
    ``maplog['rel_vel']``/``['rel_vel_healpix']``, see ``rel_vel_healpix_to_vector``).
    If ``gm_dict`` is given, satellite MAPs (``map_satellite_mask``) instead keep
    their own reported velocity ``vel_map`` -- a satellite moves relative to the
    local dark matter -- while centrals still use ``vel_dm``. Needs
    ``maplog['pid']`` (``read_maplog_dir(..., compute_pids=True)``) in that case.
    """
    vel_map = maplog['vel'][msk, :]
    vel_dm  = vel_map + rel_vel_healpix_to_vector(maplog['rel_vel_healpix'][msk], maplog['rel_vel'][msk])
    if gm_dict is None:
        return vel_dm
    if 'pid' not in maplog:
        raise KeyError("maplog has no 'pid'; read it with read_maplog_dir(..., compute_pids=True)")
    is_sat = map_satellite_mask(maplog['pid'][msk], maplog['mult'][msk], gm_dict)
    return np.where(is_sat[:, None], vel_map, vel_dm)


def get_map_zspace_poles(maplog, timestep, smu_edges, z, ells=(0, 2, 4),
                         abundance=5.01e-3, H0=67.74, Omega_M=0.3089,
                         Omega_DE=0.6911, max_p=np.inf, nthreads=26, gm_dict=None):
    """Redshift-space MAP multipoles for one maplog at one timeslice timestep.

    Like ``get_map_xi`` but, instead of the real-space monopole, selects the
    timeslice nodes, abundance-thresholds them on multiplicity, applies the
    line-of-sight RSD shift, and returns the redshift-space Legendre multipoles
    ``ells`` (see ``apply_zspace_poles``). ``z`` is the redshift at which the RSD
    Hubble factor is evaluated.

    The RSD shift uses a dark-matter velocity estimate rather than the MAP
    node's own velocity: ``maplog['rel_vel']`` / ``maplog['rel_vel_healpix']``
    give the magnitude and Nside=4 NEST direction of the MAP's velocity
    relative to the local dark matter (see ``rel_vel_healpix_to_vector``), so
    ``vel_dm = vel_map + rel_vel_vector`` recovers a dark-matter velocity
    estimate for the node. With ``gm_dict`` (the MAP -> TNG halo match), satellite
    MAPs instead use their reported ``vel`` and only centrals use ``vel_dm`` (see
    ``map_rsd_velocity``).

    Returns ``(s, poles)`` with ``poles`` of shape ``(len(ells), nbins)``.
    """
    BoxSize   = maplog['header']['BoxSize'] * u.Mpc
    timeslice = maplog['control'] == int(timestep) * 4 + 3
    thresh    = get_threshold(maplog['mult'][timeslice], BoxSize, guess=1e1, abundance=abundance)
    msk       = timeslice & (maplog['mult'] >= thresh)

    vel_rsd   = map_rsd_velocity(maplog, msk, gm_dict)

    return apply_zspace_poles(maplog['pos'][msk, :], vel_rsd,
                              BoxSize, z, smu_edges, ells=ells, H0=H0,
                              Omega_M=Omega_M, Omega_DE=Omega_DE,
                              max_p=max_p, nthreads=nthreads)


def compare_zspace_correlation_functions(maplog_tuples, redshifts, labels=None,
                                          plot_tng=False,
                                          basePath='/flare/Abacus/ajohnson/TNG300/TNG300-2/output/',
                                          ztol=5e-3,
                                          rmin=2e-2, rmax=40.0, nbins=30, mu_nbins=20,
                                          type_str='gg', ells=(0, 2, 4),
                                          abundance=5.01e-3, max_p=np.inf, nthreads=26,
                                          H0=67.74, Omega_M=0.3089, Omega_DE=0.6911,
                                          cmaps=('Blues_r', 'Greens_r', 'Purples_r', 'Oranges_r'),
                                          clo=0., chi=0.7, panel_size=(4, 3),
                                          save=False, str_identifier='',
                                          map_default=None, default_label='',
                                          plot_ratio=False, xscale='log', gm_template=None):
    """Compare MAP redshift-space multipole correlation functions on a shared grid.

    Identical in operation to ``compare_correlation_functions`` -- same row/column
    grid (one row per maplog tuple, one column per redshift), same per-row
    single-color shading, optional gray-dashed ``map_default`` reference and
    red-dashed TNG overlay -- except that it computes the first three even
    redshift-space multipoles (``ells``, default ``(0, 2, 4)``) instead of the
    real-space monopole, and produces *one figure per multipole*.

    Following the "projected" cells of analysis.ipynb, each catalog's peculiar
    z-velocity is mapped to a line-of-sight comoving shift before correlating in
    ``(s, mu)`` mode and projecting onto Legendre polynomials (see
    ``apply_zspace_poles``). Each subplot draws ``s^2 * xi_ell(s)``. (No Poisson
    band is drawn, since the multipole projection of a single realization carries
    no per-bin Poisson estimate.)

    For MAP catalogs, the RSD shift uses a dark-matter velocity estimate rather
    than the MAP node's own velocity: ``vel_dm = vel_map + rel_vel_vector``,
    where ``rel_vel_vector`` is reconstructed from ``maplog['rel_vel']`` /
    ``maplog['rel_vel_healpix']`` via ``rel_vel_healpix_to_vector`` (see
    ``get_map_zspace_poles``). TNG galaxies (when ``plot_tng``) keep using their
    own ``SubhaloVel``, since TNG has no analogous MAP/DM split.

    If ``gm_template`` is given (e.g. ``'../Images/{sim}_tng/gm_dict_z{z:.3f}.pkl'``,
    the ``match_maps_to_tng`` output; each maplog's ``header['SimName']`` fills
    ``{sim}`` and the column redshift ``{z}``), MAPs are split into centrals and
    satellites with it (``map_satellite_mask``, as in ``make_HOD_plot``): only
    centrals are shifted with the dark-matter velocity estimate; satellites use
    their reported velocity. Maplogs then need ``compute_pids=True``. TNG
    galaxies are unaffected.

    Returns ``(figs, axes_list)``: lists with one entry per multipole, each entry
    the ``(fig, axes)`` pair for that multipole's grid.

    Parameters
    ----------
    mu_nbins : int
        Number of (uniform) ``mu`` bins on ``[-1, 1]`` used for the ``(s, mu)``
        pair counts before projection.
    ells : sequence of int
        Even Legendre multipole orders to compute; one figure is produced per
        entry. Default ``(0, 2, 4)`` (the first three even multipoles).
    H0, Omega_M, Omega_DE : float
        Cosmology used to evaluate ``H(z)`` for the RSD shift (km/s/Mpc for H0).
    plot_ratio : bool
        If True, plot every curve as its ratio to the TNG300 multipole of that
        column/multipole instead of ``s^2 xi_ell(s)`` next to it. The TNG curve
        is then drawn as a horizontal line at 1 instead of its own curve.
        Requires ``plot_tng=True``. Note ratios of the ``ell>0`` multipoles can
        blow up near their zero crossings.
    xscale : {'log', 'linear'}
        Scale of the x-axis (separation ``s``).

    See ``compare_correlation_functions`` for the remaining shared parameters.
    """
    redshifts = np.atleast_1d(redshifts).astype(float)
    ells      = tuple(ells)
    nrows = len(maplog_tuples)
    ncols = len(redshifts)
    nells = len(ells)

    if plot_ratio and not plot_tng:
        raise ValueError("plot_ratio=True requires plot_tng=True to provide the "
                         "reference TNG300 multipoles")

    if labels is not None:
        assert len(labels) == nrows, "labels must have one tuple per maplog tuple (row)"
        for row, (mls, labs) in enumerate(zip(maplog_tuples, labels)):
            assert len(labs) == len(mls), \
                "labels[{}] must match the number of maplogs in row {}".format(row, row)

    s_edges  = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
    mu_edges = np.linspace(-1.0, 1.0, mu_nbins + 1)
    smu_edges = (s_edges, mu_edges)

    # resolve each column redshift to a TNG snapshot (within ztol of an available one)
    tng_col_snap = tng_col_z = None
    if plot_tng:
        star_type           = il.snapshot.partTypeNum('star')
        tng_snapnos, tng_zs = get_available_tng_snapshots(basePath)
        tng_col_snap        = np.empty(ncols, dtype=int)
        tng_col_z           = np.empty(ncols)
        for col, z_req in enumerate(redshifts):
            k = np.argmin(np.abs(tng_zs - z_req))
            if abs(tng_zs[k] - z_req) > ztol:
                raise ValueError(
                    "requested redshift {:.3f} has no TNG snapshot within {:g} "
                    "(2 decimal places); available TNG redshifts are {}".format(
                        z_req, ztol, ', '.join('{:.2f}'.format(z) for z in tng_zs)))
            tng_col_snap[col] = int(tng_snapnos[k])
            tng_col_z[col]    = tng_zs[k]

    # The gray dashed reference (map_default) depends only on the column redshift,
    # so compute its multipoles once per column and reuse them across all rows.
    default_s = default_poles = None
    if map_default is not None:
        default_s     = [None] * ncols
        default_poles = [None] * ncols
        for col, z_req in enumerate(redshifts):
            ts, zm = map_step_for_redshift(map_default, z_req)
            gm = read_gm_dict_for_maplog(map_default, z_req, gm_template) if gm_template else None
            default_s[col], default_poles[col] = get_map_zspace_poles(
                map_default, ts, smu_edges, zm, ells=ells, abundance=abundance,
                H0=H0, Omega_M=Omega_M, Omega_DE=Omega_DE,
                max_p=max_p, nthreads=nthreads, gm_dict=gm)

    # TNG multipoles depend only on the column too; precompute once per column
    # (and reuse across rows and across the per-multipole figures).
    tng_s = tng_poles = None
    if plot_tng:
        tng_s     = [None] * ncols
        tng_poles = [None] * ncols
        for col in range(ncols):
            sn           = int(tng_col_snap[col])
            tng_subhalos = load_tng_subhalos(basePath, sn)
            m_tng        = tng_subhalos['SubhaloMassType'][:, star_type]  # 1e10 Msun
            tng_boxsize  = il.groupcat.loadHeader(basePath, sn)['BoxSize'] / 1e3 * u.Mpc
            tng_thresh   = get_threshold(m_tng, tng_boxsize, guess=1e-3, abundance=abundance)
            msk_tng      = m_tng >= tng_thresh
            tng_s[col], tng_poles[col] = apply_zspace_poles(
                tng_subhalos['SubhaloPos'][msk_tng, :],
                tng_subhalos['SubhaloVel'][msk_tng, :],
                tng_boxsize, tng_col_z[col], smu_edges, ells=ells,
                H0=H0, Omega_M=Omega_M, Omega_DE=Omega_DE,
                max_p=max_p, nthreads=nthreads)

    # one figure per multipole; each is an nrows x ncols grid
    figs = [None] * nells
    axes_list = [None] * nells
    for e in range(nells):
        figs[e], axes_list[e] = plt.subplots(
            nrows, ncols, sharex=True, sharey=True, squeeze=False,
            figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
        figs[e].subplots_adjust(wspace=0, hspace=0)

    for row, maplogs in enumerate(maplog_tuples):
        cmap   = colormaps[cmaps[row % len(cmaps)]]
        n_map  = len(maplogs)
        fracs  = clo + (chi - clo) * np.linspace(0, 1, max(n_map, 1))
        for col, z_req in enumerate(redshifts):

            # gray dashed reference, precomputed once per column
            if map_default is not None:
                s_d, poles_d = default_s[col], default_poles[col]
                for e in range(nells):
                    y_d = poles_d[e] / tng_poles[col][e] if plot_ratio else s_d**2 * poles_d[e]
                    axes_list[e][row][col].plot(
                        s_d, y_d, '--', lw=2,
                        color='gray', label=default_label)

            for m, maplog in enumerate(maplogs):
                ts, zm = map_step_for_redshift(maplog, z_req)
                gm = read_gm_dict_for_maplog(maplog, z_req, gm_template) if gm_template else None
                s, poles = get_map_zspace_poles(
                    maplog, ts, smu_edges, zm, ells=ells, abundance=abundance,
                    H0=H0, Omega_M=Omega_M, Omega_DE=Omega_DE,
                    max_p=max_p, nthreads=nthreads, gm_dict=gm)
                label = str(labels[row][m]) if labels is not None else ''
                color = cmap(fracs[m])
                for e in range(nells):
                    y = poles[e] / tng_poles[col][e] if plot_ratio else s**2 * poles[e]
                    axes_list[e][row][col].plot(
                        s, y, '-', lw=2, color=color, label=label)

            # optional TNG galaxies, precomputed once per column
            if plot_tng:
                s_t, poles_t = tng_s[col], tng_poles[col]
                for e in range(nells):
                    if plot_ratio:
                        axes_list[e][row][col].axhline(1.0, lw=2, color='red', ls='--', label='TNG')
                    else:
                        axes_list[e][row][col].plot(
                            s_t, s_t**2 * poles_t[e], '--', lw=2,
                            color='red', label='TNG')

            for e in range(nells):
                ax = axes_list[e][row][col]
                ax.set(xscale=xscale)
                ax.grid()
                if col == ncols - 1:
                    ax.legend(fontsize='small')
                if row == 0:
                    ax.set_title('z = {:.2f}'.format(z_req))

    # per-figure titles and shared axis labels on the outer edge only
    for e, ell in enumerate(ells):
        fig, axes = figs[e], axes_list[e]
        #fig.suptitle(r'$\ell = {:d}$'.format(ell))
        for ax in axes[-1, :]:
            ax.set_xlabel(r'$s\;[\mathrm{Mpc}/h]$')
        for ax in axes[:, 0]:
            if plot_ratio:
                ax.set_ylabel(r'$\xi_{%s}^{(\ell=%d)}/\xi_{%s,\rm TNG}^{(\ell=%d)}$' % (type_str, ell, type_str, ell))
            else:
                ax.set_ylabel(r'$s^2\,\xi_{%s}^{(\ell=%d)}(s)$' % (type_str, ell))
        if save:
            fig.savefig('../Images/compare_zspace_correlation_functions_l{:d}_n{:.2e}_'.format(
                ell, abundance) + str_identifier + _ratio_suffix(plot_ratio) + '.png',
                bbox_inches='tight', transparent=True)
    plt.show()
    return figs, axes_list


def apply_projected_xi(pos, vel, BoxSize, z, rp_edges, pi_edges, H0=67.74, Omega_M=0.3089,
                       Omega_DE=0.6911, max_p=np.inf, nthreads=26, pimax=None):
    r"""Apply line-of-sight (z) redshift-space distortions, then compute the
    projected correlation function :math:`w_p(r_p)`.

    Same RSD shift as ``apply_zspace_poles`` (peculiar z-velocity converted to a
    comoving Mpc/h line-of-sight shift ``v_z * (1+z) / H(z) * h`` and added to the
    z position), but the shifted catalog is then correlated in ``(r_p, pi)`` mode
    (transverse/line-of-sight separation) instead of ``(s, mu)``, and integrated
    over the line-of-sight separation ``pi`` via pycorr's ``project_to_wp``:
    :math:`w_p(r_p) = \int \xi(r_p, \pi)\,d\pi` (summed over ``pi_edges``, or
    restricted to ``pimax`` if given). Integrating out the line-of-sight
    direction is what makes :math:`w_p(r_p)` (approximately) insensitive to RSD
    -- it is computed on the RSD-shifted catalog specifically to check that the
    projection recovers the real-space clustering despite the shift.

    Returns ``(rp, wp)`` on the bin centers implied by ``rp_edges``.
    """
    Hz = H0 * np.sqrt((1.0 + z)**3 * Omega_M + Omega_DE)  # km/s/Mpc
    h  = H0 / 100.0
    x  = np.copy(pos)
    x[:, 2] += vel[:, 2] * (1.0 + z) / Hz * h  # convert to comoving
    twopt  = get_xi(x, (rp_edges, pi_edges), BoxSize, mode='rppi', max_p=max_p, nthreads=nthreads)
    rp, wp = project_to_wp(twopt, pimax=pimax, return_sep=True, return_cov=False)
    return rp, wp


def get_map_projected_xi(maplog, timestep, rp_edges, pi_edges, z, abundance=5.01e-3,
                         H0=67.74, Omega_M=0.3089, Omega_DE=0.6911,
                         max_p=np.inf, nthreads=26, pimax=None, gm_dict=None):
    """Redshift-space MAP projected correlation function :math:`w_p(r_p)` for
    one maplog at one timeslice timestep.

    Like ``get_map_zspace_poles``, but projects onto :math:`w_p(r_p)` instead of
    Legendre multipoles (see ``apply_projected_xi``), and the RSD shift uses a
    dark-matter velocity estimate rather than the MAP node's own velocity:
    ``maplog['rel_vel']`` / ``maplog['rel_vel_healpix']`` give the magnitude and
    Nside=4 NEST direction of the MAP's velocity relative to the local dark
    matter (see ``rel_vel_healpix_to_vector``), so
    ``vel_dm = vel_map + rel_vel_vector`` recovers a dark-matter velocity
    estimate for the node. ``z`` is the redshift at which the RSD Hubble factor
    is evaluated. With ``gm_dict`` (the MAP -> TNG halo match), satellite MAPs
    instead use their reported ``vel`` and only centrals use ``vel_dm`` (see
    ``map_rsd_velocity``). Returns ``(rp, wp)`` on the bin centers implied by
    ``rp_edges``.
    """
    BoxSize   = maplog['header']['BoxSize'] * u.Mpc
    timeslice = maplog['control'] == int(timestep) * 4 + 3
    thresh    = get_threshold(maplog['mult'][timeslice], BoxSize, guess=1e1, abundance=abundance)
    msk       = timeslice & (maplog['mult'] >= thresh)
    vel_rsd   = map_rsd_velocity(maplog, msk, gm_dict)

    return apply_projected_xi(maplog['pos'][msk, :], vel_rsd, BoxSize, z, rp_edges, pi_edges,
                              H0=H0, Omega_M=Omega_M, Omega_DE=Omega_DE,
                              max_p=max_p, nthreads=nthreads, pimax=pimax)


def compare_projected_correlation_functions(maplog_tuples, redshifts, labels=None,
                                             plot_tng=False,
                                             basePath='/flare/Abacus/ajohnson/TNG300/TNG300-2/output/',
                                             ztol=5e-3,
                                             rp_min=2e-2, rp_max=40.0, rp_nbins=30,
                                             pi_max=40.0, pi_nbins=40,
                                             type_str='gg',
                                             abundance=5.01e-3, max_p=np.inf, nthreads=26,
                                             H0=67.74, Omega_M=0.3089, Omega_DE=0.6911,
                                             cmaps=('Blues_r', 'Greens_r', 'Purples_r', 'Oranges_r'),
                                             clo=0., chi=0.7, panel_size=(4, 3),
                                             save=False, str_identifier='',
                                             map_default=None, default_label='',
                                             plot_ratio=False, xscale='log', gm_template=None):
    r"""Compare MAP projected correlation functions :math:`w_p(r_p)` on a shared
    grid.

    Identical in operation to ``compare_zspace_correlation_functions`` -- same
    row/column grid (one row per maplog tuple, one column per redshift), same
    per-row single-color shading, optional gray-dashed ``map_default`` reference
    and red-dashed TNG overlay, same line-of-sight RSD shift -- except that
    instead of correlating in ``(s, mu)`` mode and projecting onto Legendre
    multipoles, it correlates in ``(r_p, pi)`` mode and integrates over the
    line-of-sight separation ``pi`` to get the standard projected correlation
    function :math:`w_p(r_p) = \int \xi(r_p,\pi)\,d\pi` (see
    ``apply_projected_xi``), which is why it produces a single figure rather
    than one per multipole. Because the ``pi`` integration washes out RSD, this
    is a check that ``w_p(r_p)`` from the RSD-shifted catalog recovers the
    real-space clustering. Each subplot draws ``r_p * w_p(r_p)`` (the ``plot_ratio``
    curves are unaffected by this scaling, since it cancels in the ratio). (No
    Poisson error band is drawn, since pycorr's ``project_to_wp`` does not
    return a per-bin Poisson estimate.)

    ``gm_template`` splits MAPs into centrals and satellites exactly as in
    ``compare_zspace_correlation_functions`` (satellites keep their reported
    velocity; centrals use the dark-matter estimate; TNG unaffected).

    The RSD shift also uses a different velocity than
    ``compare_zspace_correlation_functions``: instead of the MAP node's own
    velocity (``maplog['vel']``), it uses a dark-matter velocity estimate
    ``vel_dm = vel_map + rel_vel_healpix_to_vector(rel_vel_healpix, rel_vel)``,
    i.e. the MAP velocity with the MAP-relative-to-local-DM velocity subtracted
    off (see ``get_map_projected_xi``). TNG galaxies (when ``plot_tng``) keep
    using their own ``SubhaloVel``, since TNG has no analogous MAP/DM split.

    Returns ``(fig, axes)``.

    Parameters
    ----------
    rp_min, rp_max, rp_nbins : float, float, int
        Log-spaced transverse-separation (``r_p``, Mpc/h) binning.
    pi_max, pi_nbins : float, int
        The line-of-sight separation ``pi`` is binned on ``pi_nbins`` uniform
        bins spanning ``[-pi_max, pi_max]`` (Mpc/h) before being integrated over
        via ``project_to_wp`` to form ``w_p(r_p)``.
    H0, Omega_M, Omega_DE : float
        Cosmology used to evaluate ``H(z)`` for the RSD shift (km/s/Mpc for H0).
    xscale : {'log', 'linear'}
        Scale of the x-axis (separation ``r_p``).

    See ``compare_zspace_correlation_functions`` for the remaining shared
    parameters.
    """
    redshifts = np.atleast_1d(redshifts).astype(float)
    nrows = len(maplog_tuples)
    ncols = len(redshifts)

    if plot_ratio and not plot_tng:
        raise ValueError("plot_ratio=True requires plot_tng=True to provide the "
                         "reference TNG300 correlation function")

    if labels is not None:
        assert len(labels) == nrows, "labels must have one tuple per maplog tuple (row)"
        for row, (mls, labs) in enumerate(zip(maplog_tuples, labels)):
            assert len(labs) == len(mls), \
                "labels[{}] must match the number of maplogs in row {}".format(row, row)

    rp_edges = np.logspace(np.log10(rp_min), np.log10(rp_max), rp_nbins + 1)
    rp       = rp_edges[:-1] + np.diff(rp_edges) / 2.0
    pi_edges = np.linspace(-pi_max, pi_max, pi_nbins + 1)

    # resolve each column redshift to a TNG snapshot (within ztol of an available one)
    tng_col_snap = tng_col_z = None
    if plot_tng:
        star_type           = il.snapshot.partTypeNum('star')
        tng_snapnos, tng_zs = get_available_tng_snapshots(basePath)
        tng_col_snap        = np.empty(ncols, dtype=int)
        tng_col_z           = np.empty(ncols)
        for col, z_req in enumerate(redshifts):
            k = np.argmin(np.abs(tng_zs - z_req))
            if abs(tng_zs[k] - z_req) > ztol:
                raise ValueError(
                    "requested redshift {:.3f} has no TNG snapshot within {:g} "
                    "(2 decimal places); available TNG redshifts are {}".format(
                        z_req, ztol, ', '.join('{:.2f}'.format(z) for z in tng_zs)))
            tng_col_snap[col] = int(tng_snapnos[k])
            tng_col_z[col]    = tng_zs[k]

    # The gray dashed reference (map_default) depends only on the column redshift,
    # so compute it once per column here and reuse it across all rows.
    default_wp = None
    if map_default is not None:
        default_wp = [None] * ncols
        for col, z_req in enumerate(redshifts):
            ts, zm = map_step_for_redshift(map_default, z_req)
            gm = read_gm_dict_for_maplog(map_default, z_req, gm_template) if gm_template else None
            _, default_wp[col] = get_map_projected_xi(
                map_default, ts, rp_edges, pi_edges, zm, abundance=abundance,
                H0=H0, Omega_M=Omega_M, Omega_DE=Omega_DE,
                max_p=max_p, nthreads=nthreads, gm_dict=gm)

    # TNG depends only on the column too; precompute once per column and reuse
    # across rows.
    tng_wp = None
    if plot_tng:
        tng_wp = [None] * ncols
        for col in range(ncols):
            sn           = int(tng_col_snap[col])
            tng_subhalos = load_tng_subhalos(basePath, sn)
            m_tng        = tng_subhalos['SubhaloMassType'][:, star_type]  # 1e10 Msun
            tng_boxsize  = il.groupcat.loadHeader(basePath, sn)['BoxSize'] / 1e3 * u.Mpc
            tng_thresh   = get_threshold(m_tng, tng_boxsize, guess=1e-3, abundance=abundance)
            msk_tng      = m_tng >= tng_thresh
            _, tng_wp[col] = apply_projected_xi(
                tng_subhalos['SubhaloPos'][msk_tng, :],
                tng_subhalos['SubhaloVel'][msk_tng, :],
                tng_boxsize, tng_col_z[col], rp_edges, pi_edges,
                H0=H0, Omega_M=Omega_M, Omega_DE=Omega_DE,
                max_p=max_p, nthreads=nthreads)

    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False,
                             figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
    fig.subplots_adjust(wspace=0, hspace=0)

    for row, maplogs in enumerate(maplog_tuples):
        cmap   = colormaps[cmaps[row % len(cmaps)]]
        n_map  = len(maplogs)
        fracs  = clo + (chi - clo) * np.linspace(0, 1, max(n_map, 1))
        for col, z_req in enumerate(redshifts):
            ax = axes[row][col]

            # gray dashed reference, precomputed once per column
            if map_default is not None:
                y_d = default_wp[col] / tng_wp[col] if plot_ratio else rp * default_wp[col]
                ax.plot(rp, y_d, '--', lw=2, color='gray', label=default_label)

            for m, maplog in enumerate(maplogs):
                ts, zm  = map_step_for_redshift(maplog, z_req)
                gm      = read_gm_dict_for_maplog(maplog, z_req, gm_template) if gm_template else None
                _, wp   = get_map_projected_xi(
                    maplog, ts, rp_edges, pi_edges, zm, abundance=abundance,
                    H0=H0, Omega_M=Omega_M, Omega_DE=Omega_DE,
                    max_p=max_p, nthreads=nthreads, gm_dict=gm)
                label = str(labels[row][m]) if labels is not None else ''
                color = cmap(fracs[m])
                y = wp / tng_wp[col] if plot_ratio else rp * wp
                ax.plot(rp, y, '-', lw=2, color=color, label=label)

            # optional TNG galaxies, precomputed once per column
            if plot_tng:
                if plot_ratio:
                    ax.axhline(1.0, lw=2, color='red', ls='--', label='TNG')
                else:
                    ax.plot(rp, rp * tng_wp[col], '--', lw=2, color='red', label='TNG')

            ax.set(xscale=xscale)
            ax.grid()
            if col == 0:
                ax.legend(fontsize='small')
            if row == 0:
                ax.set_title('z = {:.2f}'.format(z_req))

    # shared axis labels on the outer edge only
    for ax in axes[-1, :]:
        ax.set_xlabel(r'$r_p\;[\mathrm{Mpc}/h]$')
    for ax in axes[:, 0]:
        if plot_ratio:
            ax.set_ylabel(r'$w_{p,%s}/w_{p,%s,\rm TNG}$' % (type_str, type_str))
        else:
            ax.set_ylabel(r'$r_p\,w_{p,%s}(r_p)\;[(\mathrm{Mpc}/h)^2]$' % type_str)

    if save:
        fig.savefig('../Images/compare_projected_correlation_functions_n{:.2e}_'.format(
            abundance) + str_identifier + _ratio_suffix(plot_ratio) + '.png',
            bbox_inches='tight', transparent=True)
    plt.show()
    return fig, axes


def read_state_map_fraction(rdir, npart=None):
    """Read the per-step MAP-particle fraction history from a run directory.

    Reads every ``log/step*/step*.state`` file in ``rdir`` and returns the
    redshift, full-step number, and ``total_MAP_particles`` of each step,
    together with the total particle count ``np`` used to normalise the MAP
    counts. Records are returned sorted by step.

    Parameters
    ----------
    rdir : str or Path
        Run directory containing ``abacus.par`` and the ``log/`` tree.
    npart : int, optional
        Total particle count to normalise by. If None, it is read once from
        ``NP`` in ``abacus.par``.

    Returns
    -------
    dict with keys ``step``, ``redshift``, ``total_MAP_particles``, ``frac``
    (= ``total_MAP_particles / np``), and ``np``.
    """
    if npart is None:
        npart = InputFile(str(Path(rdir) / Path('abacus.par')))['NP']

    step = []
    redshift = []
    total = []
    for f in glob(str(Path(rdir) / Path("log/step*/step*.state"))):
        state = InputFile(f)
        if 'total_MAP_particles' in state:
            step.append(state['FullStepNumber'])
            redshift.append(state['Redshift'])
            total.append(state['total_MAP_particles'])

    step     = np.array(step)
    redshift = np.array(redshift)
    total    = np.array(total, dtype=float)
    order    = np.argsort(step)

    return {'step':                step[order],
            'redshift':            redshift[order],
            'total_MAP_particles': total[order],
            'frac':                total[order] / npart,
            'np':                  npart}


def compare_map_fractions(dir_tuples, labels=None, npart=None,
                          xaxis='redshift', default_dir=None,
                          cmaps=('Blues_r', 'Greens_r', 'Purples_r', 'Oranges_r'),
                          clo=0., chi=0.7, panel_size=(8, 4),
                          save=False, str_identifier=''):
    """Compare the MAP-particle fraction history across run directories.

    For every directory all ``log/step*/step*.state`` files are read and the
    fraction ``total_MAP_particles / np`` is plotted against redshift (or
    scale factor). The figure is a column of subplots, one per tuple, sharing
    both axes. Within a subplot the directories of that tuple are drawn in
    shades of a single-color colormap chosen per tuple from ``cmaps`` (cycled
    if there are more tuples than colormaps). The near-white end of each
    colormap is excluded by sampling on ``[clo, chi]``, matching the colour
    scheme of ``compare_correlation_functions``.

    Parameters
    ----------
    dir_tuples : sequence of tuple of (str or Path)
        One tuple per subplot; each tuple is a set of run directories to
        overplot. Every entry is drawn as a colored curve.
    labels : sequence of tuple of str, optional
        Same shape as ``dir_tuples``. ``labels[row][i]`` is the legend label of
        the corresponding directory's curve.
    npart : int, optional
        Total particle count used to normalise the MAP counts. If None, it is
        read once per directory from ``NP`` in that directory's ``abacus.par``.
    xaxis : {'redshift', 'scale_factor'}
        Quantity on the x-axis. ``'scale_factor'`` plots 1/(1+z).
    default_dir : str or Path, optional
        Reference run directory. Its ``N_MAP / N_DM`` history is computed once
        and overplotted in every subplot as a gray dashed line, providing a
        common baseline for comparison.
    cmaps : sequence of str
        Per-tuple single-color colormaps, cycled by tuple.
    clo, chi : float
        Lower/upper fraction of each colormap to sample (avoids near-white ends).
    panel_size : tuple
        (width, height) in inches of a single subplot; the figure scales with
        the number of tuples.
    save : bool
        If True, save the figure under Images/.
    str_identifier : str
        Filename suffix used when ``save`` is True.

    Returns
    -------
    (fig, axes) : the Figure and the 1D ndarray of Axes.
    """
    nrows = len(dir_tuples)

    if labels is not None:
        assert len(labels) == nrows, "labels must have one tuple per dir tuple (row)"
        for row, (dirs, labs) in enumerate(zip(dir_tuples, labels)):
            assert len(labs) == len(dirs), \
                "labels[{}] must match the number of directories in row {}".format(row, row)

    fig, axes = plt.subplots(nrows, 1, sharex=True, sharey=True, squeeze=False,
                             figsize=(panel_size[0], panel_size[1] * nrows))
    axes = axes[:, 0]
    fig.subplots_adjust(hspace=0)

    # Compute the reference curve once and reuse it across all subplots.
    default_xy = None
    if default_dir is not None:
        d = read_state_map_fraction(default_dir, npart=npart)
        if xaxis == 'scale_factor':
            default_xy = (1.0 / (1.0 + d['redshift']), d['frac'] * 100)
        else:
            default_xy = (d['redshift'], d['frac'] * 100)

    for row, dirs in enumerate(dir_tuples):
        ax    = axes[row]
        cmap  = colormaps[cmaps[row % len(cmaps)]]
        n_dir = len(dirs)
        # one shade per directory, matching compare_correlation_functions
        fracs = clo + (chi - clo) * np.linspace(0, 1, max(n_dir, 1))
        # The gray dashed reference goes down first (widest) so it sits behind.
        if default_xy is not None:
            ax.plot(default_xy[0], default_xy[1], '--', lw=2,
                    color='gray')
        # Taper the line width so curves drawn later don't fully hide earlier
        # ones plotted in (nearly) the same place.
        lws = np.linspace(3.0, 2.0, max(n_dir, 1))
        for m, rdir in enumerate(dirs):
            data  = read_state_map_fraction(rdir, npart=npart)
            if xaxis == 'scale_factor':
                x = 1.0 / (1.0 + data['redshift'])
            else:
                x = data['redshift']
            label = str(labels[row][m]) if labels is not None else ''
            color = cmap(fracs[m])
            ax.plot(x, data['frac']*100, '-', lw=lws[m], color=color, label=label)

        ax.grid()
        if labels is not None or default_xy is not None:
            ax.legend(fontsize='small')

    if xaxis == 'scale_factor':
        axes[-1].set_xlabel(r'scale factor $a$')
    else:
        axes[-1].set_xlabel(r'redshift $[z]$')
        # log scale with high redshift on the left
        axes[-1].set_xscale('log')
        axes[-1].invert_xaxis()
    for ax in axes:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
        ax.set_ylabel(r'$N_{MAP} / N_{DM}$')

    if save:
        ids = ''
        if len(str_identifier)>0:
            ids += "_"+str_identifier
        fig.savefig(f'../Images/compare_map_fractions'+ids+'.png', bbox_inches='tight')
    plt.show()
    return fig, axes


def read_event_counts(rdir):
    """Per-step number of MAP particles gained by formation and by accretion.

    Reads the aggregated ``history.asdf`` of one run (gather it first with
    ``python -m abacus.post.history <rdir>``) and returns ``(z, n_form, n_accr)``
    sorted by increasing redshift, where ``n_form`` is ``size_form_used`` and
    ``n_accr`` is ``size_accr_used - N_accr_used``: the accretion group size also
    counts the pre-existing MAP particle, so subtracting the number of accretion
    events leaves only the particles that are new to a MAP (both ``size_*_used``
    fields already fold in the used_mix subset). Same quantities as the notebook's
    ``plot_npart``.
    """
    asdf_path = Path(rdir) / "history.asdf"
    if not asdf_path.is_file():
        raise FileNotFoundError(
            f"{asdf_path} not found; gather it first with "
            f"`python -m abacus.post.history {rdir}`")
    with asdf.open(asdf_path, lazy_load=True, memmap=False) as af:
        map_collision = af['timing']['gfc_report']['map_collision']
        data_names = map_collision['data_names']   # name -> slot index, robust to schema changes
        mc = map_collision['data'].sum(axis=1)     # sum over ranks -> (n_steps, n_fields)
        z = np.asarray(af['state']['Redshift'])

    n_form = mc[:, data_names['size_form_used']]
    n_accr = mc[:, data_names['size_accr_used']] - mc[:, data_names['N_accr_used']]
    order = np.argsort(z)
    return z[order], n_form[order], n_accr[order]


def _formation_nodes_one_file(f):
    """Position and redshift of every Formation node in one maplog file.

    Runs in a worker process (module-level so it pickles). A file's node stream holds
    the whole tree of every final MAP and #Formation == #Merger + 1 per tree, so the
    Formation nodes of a file are every MAP birth that file's ranks logged.
    """
    asdf.get_config().validate_on_read = False
    tbl = read_asdf(f, load=['pos', 'timestep'], select={'node_type': 'formation'})
    step_to_z = np.asarray(tbl.meta['Step_to_Z'])
    ts = np.asarray(tbl['timestep']).astype(np.int64)
    return {'pos': np.asarray(tbl['pos']).astype(np.float32),
            'redshift': step_to_z[ts - 1].astype(np.float32),
            'header': tbl.meta}


def read_formation_events(rdir, nworkers=16):
    """Every MAP formation event of a run: ``(pos, redshift, header)``.

    ``pos`` is ``(N, 3)`` box-centered [Mpc/h] where the new MAP was formed;
    ``redshift`` is the redshift of the step that formed it (``Step_to_Z[timestep - 1]``,
    as for ``read_merge_events``). ``read_maplog_dir`` keeps only head nodes, so this
    parses the full node stream of each ``maplogs/*/maplog_z*_*.asdf`` file instead.
    """
    files = sorted(glob(str(Path(rdir) / "maplogs/*/maplog_z*_*.asdf")))
    if not files:
        raise FileNotFoundError("no maplog files under {}".format(rdir))
    with ProcessPoolExecutor(max_workers=nworkers) as ex:
        results = list(ex.map(_formation_nodes_one_file, files))
    return (np.concatenate([r['pos'] for r in results]),
            np.concatenate([r['redshift'] for r in results]),
            results[0]['header'])


def compare_formation_event_correlation_functions(maplogs, redshift_bins, labels=None,
                                                  rmin=0.02, rmax=20., nbins=20, max_p=5e6,
                                                  nworkers=16, nthreads=26, plot_ratio=False,
                                                  panel_size=(4.5, 3.5), cmap='viridis',
                                                  ylim=None, save=False, str_identifier=''):
    """Auto-correlation function of the MAP formation events in each redshift bin, per run.

    One panel per bin in ``redshift_bins``; every run is drawn in each panel as
    ``r^2 xi(r)`` (Poisson band) of the positions of all Formation nodes whose redshift
    ``z`` falls in the bin (``z_lo <= z < z_hi``). No abundance matching: a run's
    events are all of its births in the bin, so runs differ in both clustering and
    number (the event count is in the legend).

    Parameters
    ----------
    maplogs : sequence of (maplog dict or str or Path)
        The runs to compare. A maplog from ``read_maplog_dir`` only holds head nodes,
        so the formation nodes are re-read from the run directory it came from
        (``maplog['header']['WorkingDirectory']``); a path to a run directory is used as is.
    redshift_bins : sequence of (z_lo, z_hi)
        Redshift bins, one panel each.
    labels : sequence of str, optional
        Legend label per run. Default: the run's ``SimName`` / directory name.
    rmin, rmax, nbins : linear separation bins in Mpc/h (``nbins`` edges, as the
        optimizer's ``R_EDGES``).
    max_p : int
        Events per bin above this are randomly subsampled (``get_xi``); ``np.inf`` for none.
    plot_ratio : bool or int
        As in ``compare_correlation_functions``, against the first run: the ratio
        ``xi / xi_0`` (or, for ``2``, its square root) instead of ``r^2 xi``.
    ylim : tuple, optional
        y-limits shared by all panels.
    save, str_identifier :
        Saved to ``../Images/compare_formation_event_correlation_functions[_<id>].png``.

    Returns ``(fig, axes, data)`` with ``data[run][(z_lo, z_hi)] = (N_events, xi, poisson)``.
    """
    bins = [(float(lo), float(hi)) for lo, hi in redshift_bins]
    rdirs = [Path(m['header']['WorkingDirectory']) if isinstance(m, dict) else Path(m)
             for m in maplogs]
    if labels is None:
        labels = [d.name for d in rdirs]
    assert len(labels) == len(rdirs), "labels must have one entry per run"

    r_edges = np.linspace(rmin, rmax, nbins)
    r = r_edges[:-1] + np.diff(r_edges) / 2.0

    data = {}
    for label, rdir in zip(labels, rdirs):
        pos, z, header = read_formation_events(rdir, nworkers=nworkers)
        BoxSize = header['BoxSize'] * u.Mpc
        data[label] = {}
        for lo, hi in bins:
            sel = (z >= lo) & (z < hi)
            if np.count_nonzero(sel) < 2:
                data[label][(lo, hi)] = (int(np.count_nonzero(sel)), None, None)
                continue
            twopt = get_xi(pos[sel], r_edges, BoxSize, max_p=max_p, nthreads=nthreads)
            data[label][(lo, hi)] = (int(np.count_nonzero(sel)),
                                     np.asarray(twopt.get_corr(return_sep=False)),
                                     np.sqrt(twopt.D1D2.wcounts) / twopt.R1R2.wcounts)

    fig, axes = plt.subplots(1, len(bins), sharex=True, sharey=True, squeeze=False,
                             figsize=(panel_size[0] * len(bins), panel_size[1]))
    fig.subplots_adjust(wspace=0.04)
    colors = colormaps[cmap](np.linspace(0.05, 0.85, max(len(labels), 1)))
    for ax, (lo, hi) in zip(axes[0], bins):
        ref = data[labels[0]][(lo, hi)][1:]
        for label, color in zip(labels, colors):
            n, xi, poisson = data[label][(lo, hi)]
            name = '{} (N={:.3g})'.format(label, n)
            if xi is None:
                ax.plot([], [], color=color, label=name)
                continue
            if plot_ratio and ref[0] is None:
                continue
            y, yerr = _xi_plot_values(r, xi, poisson, plot_ratio=plot_ratio,
                                      ref=ref if plot_ratio else None)
            ax.plot(r, y, lw=2, color=color, label=name)
            ax.fill_between(r, y - yerr, y + yerr, color=color, alpha=0.3)
        if plot_ratio:
            ax.axhline(1.0, color='gray', ls=':', lw=1)
        ax.set_title(r'${:g} \leq z < {:g}$'.format(lo, hi), fontsize='medium')
        ax.set_xlabel(r'$r\;\;[h^{-1}Mpc]$')
        ax.grid()
        ax.legend(fontsize='x-small')
    if ylim is not None:
        axes[0, 0].set_ylim(ylim)
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_ylabel(r'$r^2\xi_{form}(r)$' if not plot_ratio else
                          r'$\xi_{form}\,/\,\xi_{form,0}$' if plot_ratio is True else
                          r'$\sqrt{\xi_{form}\,/\,\xi_{form,0}}$')

    if save:
        ids = "_" + str_identifier if len(str_identifier) > 0 else ''
        fig.savefig('../Images/compare_formation_event_correlation_functions' + ids
                    + '.png', bbox_inches='tight', transparent=True)
    plt.show()
    return fig, axes, data


def compare_formation_events(dir_tuples, labels=None, default_dir=None,
                             cmaps=('Blues_r', 'Greens_r', 'Purples_r', 'Oranges_r'),
                             clo=0., chi=0.7, panel_size=(8, 4),
                             save=False, str_identifier='', cumsum=0, plot_ratio=False,
                             ylim=None, yscale=None):
    """Compare the per-step MAP formation and accretion histories across runs.

    For every run directory, plots the number of MAP particles that enter a MAP
    at each timestep through formation (dots) and through accretion (``+``)
    against redshift (see ``read_event_counts``). Layout and conventions follow
    ``compare_map_fractions``: one row of subplots per tuple, sharing both axes,
    with the directories of a tuple drawn in shades of a single-color colormap
    chosen per tuple from ``cmaps`` (sampled on ``[clo, chi]``). Formation is
    drawn in the left column and accretion in the right (a ``(nrows, 2)`` layout),
    so a run has the same colour in both; there is no combined "total" curve.
    With ``cumsum == 2`` (the total only) there is nothing to split and a single
    column is drawn.

    Parameters
    ----------
    dir_tuples : sequence of tuple of (str or Path)
        One tuple per subplot; each is a set of run directories (each containing
        ``history.asdf``) to overplot.
    labels : sequence of tuple of str, optional
        Same shape as ``dir_tuples``; ``labels[row][i]`` is the legend label of
        the corresponding run (shown in the legend of the left column).
    default_dir : str or Path, optional
        Reference run drawn in every subplot as gray dashed curves, computed
        once.
    cmaps, clo, chi, panel_size, save, str_identifier :
        As in ``compare_map_fractions`` (saved to
        ``Images/compare_formation_events[_<str_identifier>].png``).

    cumsum : int {0, 1, 2}
        0 (or False): per-step counts. 1 (or True): the running total of each
        curve from high to low redshift, normalised to ``NP/100`` (``NP`` read
        from each run's ``abacus.par``; the same normalisation as the notebook's
        ``plot_npart``) and shown as a percentage on a linear y-axis -- the
        percent of that reference particle count that has entered MAPs through
        formation (left) and accretion (right) by that redshift. 2: the same
        cumulative quantity but only the total (formation + accretion), one
        curve per run.
    plot_ratio : bool
        If True, plot every curve divided by the corresponding ``default_dir`` curve
        (formation by formation, accretion by accretion, or total by total for
        ``cumsum == 2``) instead of the curves themselves, on a linear y-axis with
        the reference drawn as a gray dashed line at 1. The reference is
        interpolated linearly in redshift onto each run's own steps (the runs'
        step grids need not match). Requires ``default_dir``. For the per-step
        counts (``cumsum == 0``) steps where either count is zero have no ratio and
        are dropped, and the result is noisy since the two runs' steps are not
        the same events.
    ylim : tuple (ymin, ymax), optional
        y-axis limits (shared by all subplots); a ``None`` entry leaves that end
        automatic. If ``None`` (default): automatic, except with ``plot_ratio``
        where it is ``(0.5, 1.5)`` -- a ratio is unbounded wherever one run has
        (nearly) no events and the other does, as at high redshift, which would
        otherwise set the scale. Pass ``(None, None)`` for fully automatic limits.
    yscale : {'linear', 'log'}, optional
        y-axis scale. Default: log for the per-step counts, linear otherwise
        (cumulative curves and ratios).

    Returns
    -------
    (fig, axes) : the Figure and the ``(nrows, ncols)`` ndarray of Axes (``ncols`` is 2, or 1
    for ``cumsum == 2``).
    """
    nrows = len(dir_tuples)
    if plot_ratio and default_dir is None:
        raise ValueError("plot_ratio=True requires default_dir (the reference run)")
    if labels is not None:
        assert len(labels) == nrows, "labels must have one tuple per dir tuple (row)"
        for row, (dirs, labs) in enumerate(zip(dir_tuples, labels)):
            assert len(labs) == len(dirs), \
                "labels[{}] must match the number of directories in row {}".format(row, row)

    ncols = 1 if cumsum == 2 else 2          # formation | accretion
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, squeeze=False,
                             figsize=(panel_size[0] * ncols, panel_size[1] * nrows))
    fig.subplots_adjust(hspace=0, wspace=0.04)

    def _read(rdir):
        """(z, n_form, n_accr) for one run; with ``cumsum``, each is accumulated
        in chronological order (high z first) and normalised to NP/100, so the
        values are percentages. ``cumsum == 2`` returns (z, total) instead."""
        z, n_form, n_accr = read_event_counts(rdir)
        if cumsum:
            total = InputFile(str(Path(rdir) / 'abacus.par'))['NP'] / 100
            z, n_form, n_accr = z[::-1], n_form[::-1], n_accr[::-1]
            n_form, n_accr = np.cumsum(n_form) / total, np.cumsum(n_accr) / total
            if cumsum == 2:
                return z, n_form + n_accr
        return z, n_form, n_accr

    ms = 6
    def _draw(axrow, z, *n, color, ls, lw, label):
        """Formation on ``axrow[0]`` and accretion on ``axrow[1]`` in one colour (or, for
        ``cumsum == 2``, the single total curve on the only axis); for the per-step counts drop
        empty steps, which can't be drawn on the log axis."""
        if cumsum == 2:
            axrow[0].plot(z, n[0], ls=ls, lw=lw, marker='.', ms=ms, color=color, label=label)
            return
        n_form, n_accr = n
        f, a = (n_form >= 0, n_accr >= 0) if cumsum else (n_form > 0, n_accr > 0)
        axrow[0].plot(z[f], n_form[f], ls=ls, lw=lw, marker='.', ms=ms, color=color, label=label)
        axrow[1].plot(z[a], n_accr[a], ls=ls, lw=lw, marker='.', ms=ms, color=color)

    # the reference run is shared by every subplot, so read it once
    default = _read(default_dir) if default_dir is not None else None

    def _to_ratio(z, *n):
        """Divide each of ``n`` by the reference's matching curve interpolated onto
        ``z``; entries with no valid ratio (zero/undefined either side) become NaN."""
        zr   = default[0]
        sort = np.argsort(zr)
        out  = []
        for n_i, ref_i in zip(n, default[1:]):
            ref = np.interp(z, zr[sort], ref_i[sort])
            ok  = (ref > 0) if cumsum else (ref > 0) & (n_i > 0)
            out.append(np.where(ok, n_i / np.where(ref > 0, ref, 1), np.nan))
        return (z, *out)

    for row, dirs in enumerate(dir_tuples):
        axrow = axes[row]
        cmap  = colormaps[cmaps[row % len(cmaps)]]
        fracs = clo + (chi - clo) * np.linspace(0, 1, max(len(dirs), 1))
        if plot_ratio:
            for ax in axrow:
                ax.axhline(1.0, color='gray', ls='--', lw=2)
        elif default is not None:
            _draw(axrow, *default, color='gray', ls='--', lw=2, label='')
        for m, rdir in enumerate(dirs):
            label = str(labels[row][m]) if labels is not None else ''
            data  = _to_ratio(*_read(rdir)) if plot_ratio else _read(rdir)
            _draw(axrow, *data, color=cmap(fracs[m]), ls='-', lw=1.5, label=label)

        for ax in axrow:
            ax.grid()
        if axrow[0].get_legend_handles_labels()[0]:
            axrow[0].legend(fontsize='small', markerscale=1.5)

    if ncols == 2:
        axes[0, 0].set_title('formation', fontsize='medium')
        axes[0, 1].set_title('accretion', fontsize='medium')
    for ax in axes[-1]:
        ax.set(xscale='log', xlabel=r'redshift $[z]$')
    axes[-1, 0].set_yscale(yscale if yscale is not None
                           else 'linear' if (cumsum or plot_ratio) else 'log')
    axes[-1, 0].invert_xaxis()                   # high-z left, low-z right (axes are shared)
    if plot_ratio:
        ylabel = (r'$\Sigma \mathcal{M}_{MAP}$ / reference' if cumsum
                  else r'$N_{MAP}$ per timestep / reference')
    elif cumsum:
        for ax in axes.flat:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
            ax.set_yticks([10, 20, 30])   # none at the panel edges, where adjacent subplots' labels would collide
        ylabel = r'$\Sigma \mathcal{M}_{MAP}\,/\,N_p$'
    else:
        ylabel = r'$N_{MAP}$ per timestep'
    if ylim is None and plot_ratio:
        ylim = (0.5, 1.5)
    if ylim is not None:
        axes[-1, 0].set_ylim(ylim)
    # one label for the whole figure, not one per subplot, placed just left of the
    # widest tick labels (a fixed position would collide with them)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    x_left = min(ax.yaxis.get_tightbbox(renderer).x0 for ax in axes[:, 0])
    x_left = fig.transFigure.inverted().transform((x_left, 0))[0]
    fig.supylabel(ylabel, x=x_left - 0.01, ha='right')

    if save:
        ids = ''
        if len(str_identifier) > 0:
            ids += "_" + str_identifier
        fig.savefig('../Images/compare_formation_events' + ids + _ratio_suffix(plot_ratio) + '.png',
                    bbox_inches='tight', transparent=True)
    plt.show()
    return fig, axes


def read_event_energies(rdir):
    """Per-step ``v^2 * mass`` of MAP formation and accretion events for one run.

    Reads the aggregated ``history.asdf`` (gather it first with
    ``python -m abacus.post.history <rdir>``) and returns ``(z, E_form, E_accr)``
    sorted by increasing redshift, where for each step

    ``E_form = <v_form>^2 * size_form_used``
    ``E_accr = <v_accr>^2 * (size_accr_used - N_accr_used)``

    ``<v>`` is the used-event mean velocity ``v_*_used / N_*_used`` (the quantities
    ``plot_vels`` draws: new ``<v_rms>`` for formation, ``sqrt(<v_rel^2>)`` for
    accretion), and the mass is the number of particles entering MAPs that step in
    each channel (as ``read_event_counts``). Units are (km/s)^2 times particle
    count; steps with no events of a type contribute zero.
    """
    asdf_path = Path(rdir) / "history.asdf"
    if not asdf_path.is_file():
        raise FileNotFoundError(
            f"{asdf_path} not found; gather it first with "
            f"`python -m abacus.post.history {rdir}`")
    with asdf.open(asdf_path, lazy_load=True, memmap=False) as af:
        map_collision = af['timing']['gfc_report']['map_collision']
        names = map_collision['data_names']
        mc = map_collision['data'].sum(axis=1)     # sum over ranks -> (n_steps, n_fields)
        z = np.asarray(af['state']['Redshift'])

    def energy(v_name, n_name, mass):
        n = mc[:, names[n_name]]
        with np.errstate(divide='ignore', invalid='ignore'):
            vmean = np.where(n > 0, mc[:, names[v_name]] / n, 0.0)
        return vmean ** 2 * mass

    E_form = energy('v_form_used', 'N_form_used', mc[:, names['size_form_used']])
    E_accr = energy('v_accr_used', 'N_accr_used',
                    mc[:, names['size_accr_used']] - mc[:, names['N_accr_used']])
    order = np.argsort(z)
    return z[order], E_form[order], E_accr[order]


def compare_energies(dir_tuples, labels=None, default_dir=None,
                     cmaps=('Blues_r', 'Greens_r', 'Purples_r', 'Oranges_r'),
                     clo=0., chi=0.7, panel_size=(8, 4),
                     save=False, str_identifier='', combine=False, yscale='log'):
    """Compare the cumulative kinetic "energy" of MAP formation and accretion across runs.

    Layout and conventions follow ``compare_formation_events`` / ``compare_velocities``:
    a column of subplots, one per tuple, sharing both axes, runs in shades of a single
    colour per tuple, a gray dashed ``default_dir`` in every subplot, and one y-label for
    the figure. Each step contributes ``v^2 * M`` for formation events (``v`` the mean
    formation velocity, ``M`` the particles entering new MAPs) and for accretion events
    (``v`` the mean relative velocity, ``M`` the new particles accreted) -- see
    ``read_event_energies``. The curves are the running totals from high to low redshift,
    divided by ``NP`` (from each run's ``abacus.par``) so the units are (km/s)^2, a mean
    per simulation particle.

    Parameters
    ----------
    dir_tuples : sequence of tuple of (str or Path)
        One tuple per subplot; each is a set of run directories (each containing
        ``history.asdf``) to overplot.
    labels : sequence of tuple of str, optional
        Same shape as ``dir_tuples``; legend label of each run (on its formation curve; the
        top subplot also gets a marker key, dot = formation, ``+`` = accretion).
    default_dir : str or Path, optional
        Reference run drawn in every subplot as gray dashed curves.
    combine : bool
        False (default): formation (dots) and accretion (``+``) are drawn separately,
        in the same colour per run. True: one curve per run, their sum.
    yscale : {'log', 'linear'}
        Scale of the y-axis. On a log axis the leading zeros of the running totals
        (before any events) are not drawn.
    cmaps, clo, chi, panel_size, save, str_identifier :
        As in ``compare_formation_events`` (saved to
        ``Images/compare_energies[_<str_identifier>].png``).

    Returns
    -------
    (fig, axes) : the Figure and the 1D ndarray of Axes.
    """
    nrows = len(dir_tuples)
    if labels is not None:
        assert len(labels) == nrows, "labels must have one tuple per dir tuple (row)"
        for row, (dirs, labs) in enumerate(zip(dir_tuples, labels)):
            assert len(labs) == len(dirs), \
                "labels[{}] must match the number of directories in row {}".format(row, row)

    fig, axes = plt.subplots(nrows, 1, sharex=True, sharey=True, squeeze=False,
                             figsize=(panel_size[0], panel_size[1] * nrows))
    axes = axes[:, 0]
    fig.subplots_adjust(hspace=0)

    def _read(rdir):
        """(z, cumulative E_form / NP, cumulative E_accr / NP), chronological (high z first);
        (z, total) when ``combine``."""
        z, E_form, E_accr = read_event_energies(rdir)
        npart = InputFile(str(Path(rdir) / 'abacus.par'))['NP']
        z, E_form, E_accr = z[::-1], E_form[::-1], E_accr[::-1]
        cf, ca = np.cumsum(E_form) / npart, np.cumsum(E_accr) / npart
        return (z, cf + ca) if combine else (z, cf, ca)

    ms = 6
    def _draw(ax, z, *E, color, ls, lw, label):
        pos = [e > 0 for e in E] if yscale == 'log' else [np.ones(len(e), bool) for e in E]
        ax.plot(z[pos[0]], E[0][pos[0]], ls=ls, lw=lw, marker='.', ms=ms, color=color, label=label)
        if not combine:
            ax.plot(z[pos[1]], E[1][pos[1]], ls=ls, lw=lw, marker='+', ms=ms, color=color, label='_nolegend_')

    default = _read(default_dir) if default_dir is not None else None
    for row, dirs in enumerate(dir_tuples):
        ax    = axes[row]
        cmap  = colormaps[cmaps[row % len(cmaps)]]
        fracs = clo + (chi - clo) * np.linspace(0, 1, max(len(dirs), 1))
        if default is not None:
            _draw(ax, *default, color='gray', ls='--', lw=2, label='')
        for m, rdir in enumerate(dirs):
            label = str(labels[row][m]) if labels is not None else ''
            _draw(ax, *_read(rdir), color=cmap(fracs[m]), ls='-', lw=1.5, label=label)
        ax.grid()
        if row == 0 and not combine:
            ax.plot([], [], ls='none', marker='.', ms=ms, color='k', label='formation')
            ax.plot([], [], ls='none', marker='+', ms=ms, color='k', label='accretion')
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize='small', markerscale=1.5)

    axes[-1].set(xscale='log', yscale=yscale, xlabel=r'redshift $[z]$')
    axes[-1].invert_xaxis()                      # high-z left, low-z right
    if yscale != 'log':
        axes[0].set_ylim(bottom=0)               # running totals are non-negative
    ylabel = r'$\Sigma\,v^2 M\,/\,N_p$ [(km/s)$^2$]'
    # one label for the whole figure, just left of the widest tick labels
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    x_left = min(ax.yaxis.get_tightbbox(renderer).x0 for ax in axes)
    x_left = fig.transFigure.inverted().transform((x_left, 0))[0]
    fig.supylabel(ylabel, x=x_left - 0.01, ha='right')

    if save:
        ids = "_" + str_identifier if len(str_identifier) > 0 else ''
        fig.savefig('../Images/compare_energies' + ids + '.png', bbox_inches='tight', transparent=True)
    plt.show()
    return fig, axes


def plot_vels(vel_dir, save=True):
    """Mean formation/accretion/merger velocities vs. redshift for one run.

    Reads the aggregated ``history.asdf`` (gather it first with
    ``python -m abacus.post.history <vel_dir>`` if missing -- new schema,
    everything lives under the "timing" group, not "data") and plots the
    used-event mean velocity of each collision type (new/accreted/merger)
    against redshift, with a secondary temperature axis (``T =
    0.5*mu*m_p*v^2/k_B``).

    Parameters
    ----------
    vel_dir : str or Path
        Run directory containing ``history.asdf``.
    save : bool
        If True, save the figure to ``Images/velocities_<vel_dir name>.png``
        -- keyed by the directory's own name, so batch runs over many
        simulations don't collide on the same ``Images/`` path.

    Returns
    -------
    (fig, ax)
    """
    specs = [
        (r"new $\langle v_{rms} \rangle$", colormaps['Blues'](0.7), '.', 'v_form_used', 'N_form_used'),
        (r"accreted $\sqrt{\langle v_{rel}^2\rangle}$", colormaps['Oranges'](0.7), '.', 'v_accr_used', 'N_accr_used'),
        (r"merger $\sqrt{\langle v_{rel}^2\rangle}$", colormaps['Greens'](0.7), '.', 'v_merg_used', 'N_merg_used'),
    ]

    asdf_path = Path(vel_dir) / "history.asdf"
    if not asdf_path.is_file():
        raise FileNotFoundError(
            f"{asdf_path} not found; gather it first with "
            f"`python -m abacus.post.history {vel_dir}`")
    with asdf.open(asdf_path, lazy_load=True, memmap=False) as af:
        map_collision = af['timing']['gfc_report']['map_collision']
        data_names = map_collision['data_names']  # name -> slot index, robust to schema changes
        mc = map_collision['data'].sum(axis=1)     # sum over ranks -> (n_steps, n_fields)
        z_all = np.asarray(af['state']['Redshift'])

    fig, ax = plt.subplots(1, figsize=(9, 5))
    for title, c, marker, v_name, n_name in specs:
        n = mc[:, data_names[n_name]]
        keep = n > 0                                 # drop steps with no collisions of this type
        vmean = mc[keep, data_names[v_name]] / n[keep]
        z = z_all[keep]
        order = np.argsort(z)
        ax.plot(z[order], vmean[order], marker=marker, ls='none', ms=6, color=c, label=title)

    ax.set(xscale='log', xlabel='redshift [$z$]', ylabel=r'$v\;\;[km/s]$')
    ax.invert_xaxis()                                # high-z left, low-z right
    ax.grid()
    ax.legend(markerscale=2)

    ax1 = ax.secondary_yaxis('right', functions=(
        lambda v: (0.5 * mu * m_p * (v * u.km / u.s) ** 2 / k_B).to(u.K).value,
        lambda T: np.sqrt((2 * (T * u.K) * k_B / mu / m_p).to(u.km ** 2 / u.s ** 2).value)))
    ax1.set_ylabel(r'T [K]', rotation=270, labelpad=13)
    ax1.set(yscale='log')

    if save:
        fig.savefig(f'Images/velocities_{Path(vel_dir).name:s}.png', bbox_inches='tight', transparent=True)
    plt.show()
    return fig, ax


# (title, history field of summed velocity, history field of event count) per
# collision category, as in plot_vels; the key names the output file / return dict
_VELOCITY_SPECS = {
    'form': (r"new $\langle v_{rms} \rangle$",                'v_form_used', 'N_form_used'),
    'accr': (r"accreted $\sqrt{\langle v_{rel}^2\rangle}$",    'v_accr_used', 'N_accr_used'),
    'merg': (r"merger $\sqrt{\langle v_{rel}^2\rangle}$",      'v_merg_used', 'N_merg_used'),
}


def read_collision_velocities(rdir):
    """Per-step mean collision velocity of each category for one run.

    Reads the aggregated ``history.asdf`` (gather it first with
    ``python -m abacus.post.history <rdir>``) and returns
    ``{'form'|'accr'|'merg': (z, vmean)}`` (used events only), sorted by increasing
    redshift with steps that have no collisions of that type dropped -- the same
    quantities ``plot_vels`` draws.
    """
    asdf_path = Path(rdir) / "history.asdf"
    if not asdf_path.is_file():
        raise FileNotFoundError(
            f"{asdf_path} not found; gather it first with "
            f"`python -m abacus.post.history {rdir}`")
    with asdf.open(asdf_path, lazy_load=True, memmap=False) as af:
        map_collision = af['timing']['gfc_report']['map_collision']
        data_names = map_collision['data_names']  # name -> slot index, robust to schema changes
        mc = map_collision['data'].sum(axis=1)     # sum over ranks -> (n_steps, n_fields)
        z_all = np.asarray(af['state']['Redshift'])

    out = {}
    for key, (_, v_name, n_name) in _VELOCITY_SPECS.items():
        n = mc[:, data_names[n_name]]
        keep = n > 0
        z = z_all[keep]
        vmean = mc[keep, data_names[v_name]] / n[keep]
        order = np.argsort(z)
        out[key] = (z[order], vmean[order])
    return out


def compare_velocities(dir_tuples, labels=None, default_dir=None,
                       cmaps=('Blues_r', 'Greens_r', 'Purples_r', 'Oranges_r'),
                       clo=0., chi=0.7, panel_size=(8, 4),
                       save=False, str_identifier=''):
    """Compare MAP collision velocities vs. redshift across runs.

    The ``plot_vels`` quantities (used-event mean velocity of new / accreted /
    merger collisions, drawn as dots with a secondary temperature axis
    ``T = 0.5*mu*m_p*v^2/k_B``) compared between simulations in the style of
    ``compare_map_fractions``: a column of subplots, one per tuple, sharing both
    axes, with the runs of a tuple in shades of a single-color colormap chosen
    per tuple from ``cmaps`` (sampled on ``[clo, chi]``). One figure is made for
    each velocity category, so three figures are produced.

    Parameters
    ----------
    dir_tuples : sequence of tuple of (str or Path)
        One tuple per subplot; each is a set of run directories (each containing
        ``history.asdf``) to overplot.
    labels : sequence of tuple of str, optional
        Same shape as ``dir_tuples``; legend label of each run.
    default_dir : str or Path, optional
        Reference run drawn in every subplot as gray dots, read once.
    cmaps, clo, chi, panel_size, save, str_identifier :
        As in ``compare_map_fractions``. Saved to
        ``Images/compare_velocities_<form|accr|merg>[_<str_identifier>].png``.

    Returns
    -------
    figs, axes : dicts keyed by ``'form'``, ``'accr'``, ``'merg'`` holding each
        category's Figure and 1D ndarray of Axes.
    """
    nrows = len(dir_tuples)
    if labels is not None:
        assert len(labels) == nrows, "labels must have one tuple per dir tuple (row)"
        for row, (dirs, labs) in enumerate(zip(dir_tuples, labels)):
            assert len(labs) == len(dirs), \
                "labels[{}] must match the number of directories in row {}".format(row, row)

    # every run is needed by every figure, so read each history once
    default = read_collision_velocities(default_dir) if default_dir is not None else None
    data = [[read_collision_velocities(d) for d in dirs] for dirs in dir_tuples]

    v2T = lambda v: (0.5 * mu * m_p * (v * u.km / u.s) ** 2 / k_B).to(u.K).value
    T2v = lambda T: np.sqrt((2 * (T * u.K) * k_B / mu / m_p).to(u.km ** 2 / u.s ** 2).value)

    figs, all_axes = {}, {}
    for key, (title, _, _) in _VELOCITY_SPECS.items():
        fig, axes = plt.subplots(nrows, 1, sharex=True, sharey=True, squeeze=False,
                                 figsize=(panel_size[0], panel_size[1] * nrows))
        axes = axes[:, 0]
        fig.subplots_adjust(hspace=0)

        for row, runs in enumerate(data):
            ax    = axes[row]
            cmap  = colormaps[cmaps[row % len(cmaps)]]
            fracs = clo + (chi - clo) * np.linspace(0, 1, max(len(runs), 1))
            if default is not None:
                ax.plot(*default[key], marker='.', ls='none', ms=6, color='gray')
            for m, run in enumerate(runs):
                label = str(labels[row][m]) if labels is not None else ''
                ax.plot(*run[key], marker='.', ls='none', ms=6, color=cmap(fracs[m]), label=label)
            ax.grid()
            if labels is not None:
                ax.legend(fontsize='small', markerscale=2)

            ax1 = ax.secondary_yaxis('right', functions=(v2T, T2v))
            ax1.set_ylabel(r'T [K]', rotation=270, labelpad=13)
            ax1.set(yscale='log')
            ax.set_ylabel(r'$v\;\;[km/s]$')

        axes[0].set_title(title)
        axes[-1].set(xscale='log', xlabel='redshift [$z$]')
        axes[-1].invert_xaxis()                  # high-z left, low-z right

        if save:
            ids = "_" + str_identifier if len(str_identifier) > 0 else ''
            fig.savefig(f'../Images/compare_velocities_{key}' + ids + '.png',
                        bbox_inches='tight', transparent=True)
        figs[key], all_axes[key] = fig, axes
    plt.show()
    return figs, all_axes


def get_particle_pos_tng(basePath,snapno,partType,max_p=5e7, return_vel=False):
    pos = np.empty((int(max_p),3),dtype=float)
    vel = np.empty((int(max_p),3),dtype=float)
    subset = il.snapshot.getSnapOffsets(basePath, snapno, 0, "Subhalo")
    with h5.File(il.snapshot.snapPath(basePath, snapno), 'r') as f:
        header = dict(f['Header'].attrs.items())
        nPart = il.snapshot.getNumPart(header)
    scale_factor = header['Time']
    downsample = max_p/nPart[il.snapshot.partTypeNum(partType)]
    startpoint=0
    chunk_len=500000
    totstored=0
    while(startpoint < nPart[il.snapshot.partTypeNum(partType)]):
        len_range=np.min([chunk_len, nPart[il.snapshot.partTypeNum(partType)]-startpoint])
        subset['offsetType'][il.snapshot.partTypeNum(partType)] = startpoint
        subset['lenType'][il.snapshot.partTypeNum(partType)]    = len_range
        fields = ['Coordinates', 'Velocities'] if return_vel else ['Coordinates']
        data_tmp = il.snapshot.loadSubset(basePath, snapno, partType, fields=fields, subset=subset)
        startpoint=startpoint+len_range
        n2store = int(np.floor(downsample*len_range))
        idx = rng.choice(len_range, size=n2store)
        if return_vel:
            pos[totstored:totstored+n2store,:] = data_tmp['Coordinates'][idx]/1e3 #convert tp Mpc/h
            # TNG stores velocities in km/sqrt(a)/s; multiply by sqrt(a) for peculiar velocity in km/s
            vel[totstored:totstored+n2store,:] = data_tmp['Velocities'][idx] * np.sqrt(scale_factor) #sqrt(a) required for correct units
        else:
            pos[totstored:totstored+n2store,:] = data_tmp[idx]/1e3
        totstored += n2store
    if return_vel:
        return pos[:totstored,:], vel[:totstored,:]
    else:
        return pos[:totstored,:]
