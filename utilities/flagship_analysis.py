"""Read-only analysis of the live flagship run's per-step timing files.

Reads <log>/stepNNNN/stepNNNN.time.bin (one raw PerformanceSummary struct per MPI rank, laid out by
<log>/timing_schema.json) and the sibling stepNNNN.state text files.  Nothing here writes, renames or
locks anything under the log directory: every access is open(..., 'r'), np.fromfile, Path.stat() or
Path.iterdir().  Only step directories with min_step <= N <= max_step are ever opened, so set max_step
below the step that is being written.

Usage from the notebook (the data cache lives in the object `fd`, not in this module, so
importlib.reload(flagship_analysis) after an edit keeps it):

    import importlib, flagship_analysis as fa
    importlib.reload(fa)
    if 'fd' not in globals():
        fd = fa.open_run(max_step=211)
    fa.configure(fd, max_step=211)
    fa.read_steps(fd)                      # reads only steps not yet cached
    fa.plot_overview(fd)
    fa.plot_groups(fd, axis='x')
    fa.group_heatmaps(fd, ['busy', 'slab_pipeline.near_force_time'], axis='x')
    fa.plot_components(fd, 'slab_pipeline')    # any struct: 'spinning', 'io_stats', 'near_force', ...
    fa.plot_wait_vs_distance(fd, axis='x')    # MPI wait (max rank, summed flags) beyond the busy spread vs node separation
    fa.find_slow_ranks(fd)    # ranks behind offending x AND z groups, kick+drift by default (needs both axes)

Rank grid: rank = x*size_z + z (checked per file).  Group axes:
    'x': fixed x, all z -- size_z ranks (comm_1d_z), which sweep slab by slab in lock step
    'z': fixed z, all x -- size_x ranks (comm_1d_x), which are pipelined along x
"""

import itertools
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

DEFAULT_LOG = Path("/tmp/ajohnson/AbacusAurora_run/Outputs/lgarrison/AbacusAurora_flagship_c000_ph400/log")
DEFAULT_JOB_OUT = Path("/home/lgarrison/AbacusAurora/job/out")   # job logs, scanned for checkpoint events
# Node placement.  The timing files do NOT contain hostnames (their node_string is ".<MPI rank>"), so the
# rank -> node map comes from a job's hostfile: rank r is on line r // ranks_per_node (the launch is
# `mpirun --hostfile <nodefile> -ppn 2`, ranks placed in hostfile order).  Placement differs from job to job,
# so the node names are only valid for the steps that ran in that job: job 8948658 restarted from the step-176
# checkpoint and ran steps 177..211.
DEFAULT_NODEFILE = Path("/home/lgarrison/AbacusAurora/job/out/8948658/nodefile_part_000")
DEFAULT_NODE_STEPS = (177, 211)

FLAG_STATS = ('min', 'mean', 'max', 'sum')   # reductions over all ranks, kept for every field
GROUP_STATS = ('min', 'mean', 'max')         # reductions over each group's ranks
GROUP_SKIP = ('gfc_report',)                 # field-name prefixes left out of group_fields='scalars'
# Derived per-rank fields, reduced like the real ones.  busy = the time a rank actually spent working in the
# pipeline step: slab_pipeline_time minus the time spent spinning minus the wait at the barrier that follows the
# event loop (mpi_barrier_time = the rank's slack, which is INSIDE slab_pipeline_time).  Without the barrier term,
# busy would differ between ranks only by their spin time, because that barrier makes slab_pipeline_time nearly
# the same on every rank.
# MPI-wait spin timers grouped by the communication pattern they wait on (src/Timestep/pipeline.cpp, parallel.cpp):
#   manifest      one send per step to x-1 (same z) on comm_manifest = dup(comm_1d_x), received from x+1
#   xtranspose    the convolution's multipole/taylor transposes (ParallelConvolution): Ialltoallv rounds on
#                 comm_multipoles/comm_taylors = dup(comm_1d_x), i.e. all pairs of the fixed-z group
#   ztranspose    2D multipole/taylor transposes: Ialltoallv on comm_multipoles_z/comm_taylors_z = dup(comm_1d_z),
#                 all pairs of the fixed-x group
#   zneighbor     neighbor/accdens exchange with the z neighbors, on comm_1d_z
# 'partners' says which ranks a rank exchanges with, for pattern_distance.  There is no manifest-SEND spin timer
# in the schema: waiting_for_mpi_manifest_time is the only manifest wait.
WAIT_PATTERNS = {
    'manifest':   dict(partners='x_ring', flags=['spinning.waiting_for_mpi_manifest_time']),
    'xtranspose': dict(partners='x_all', flags=['spinning.waiting_for_mpi_multipole_recv_time',
                                                'spinning.waiting_for_mpi_multipole_send_time',
                                                'spinning.waiting_for_mpi_taylor_recv_time',
                                                'spinning.waiting_for_mpi_taylor_send_time']),
    'ztranspose': dict(partners='z_all', flags=['spinning.waiting_for_mpi_taylor_2d_time',
                                                'spinning.waiting_for_mpi_multipole_2d_time']),
    'zneighbor':  dict(partners='z_ring', flags=['spinning.waiting_for_mpi_neighbor_recv_2d_time',
                                                 'spinning.waiting_for_mpi_accdens_recv_2d_time']),
}
# patterns reduced over each kind of group: x groups (fixed x) only do z-direction work; z groups (fixed z)
# do the x-direction work and, through their ranks, the z-direction work too
PATTERNS_BY_AXIS = {'x': ['ztranspose', 'zneighbor'],
                    'z': ['manifest', 'xtranspose', 'ztranspose', 'zneighbor']}


def _wait_sum(names):
    def f(r):
        tot = 0.0
        for n in names:
            a = r
            for k in n.split('.'):
                a = a[k]
            tot = tot + np.asarray(a, dtype=np.float64)
        return tot
    return f


# Spin flags for the rack-contention analysis: what a rank waits on to RECEIVE (ingress) or to SEND (egress).
# The all-to-all 2D flags cover both directions, so they are in both lists.  The schema has only two send flags.
RACK_SPIN_FLAGS = {
    'ingress': ['spinning.waiting_for_mpi_manifest_time', 'spinning.waiting_for_mpi_multipole_recv_time',
                'spinning.waiting_for_mpi_taylor_recv_time', 'spinning.waiting_for_mpi_neighbor_recv_2d_time',
                'spinning.waiting_for_mpi_accdens_recv_2d_time', 'spinning.waiting_for_mpi_multipole_2d_time',
                'spinning.waiting_for_mpi_taylor_2d_time'],
    'egress': ['spinning.waiting_for_mpi_multipole_send_time', 'spinning.waiting_for_mpi_taylor_send_time',
               'spinning.waiting_for_mpi_multipole_2d_time', 'spinning.waiting_for_mpi_taylor_2d_time'],
}


GROUP_DERIVED_VERSION = 7    # bump when a GROUP_DERIVED definition changes; read_steps then rebuilds the cached
                             # group reductions (re-reading every step once)
GROUP_DERIVED = {
    'busy': lambda r: np.asarray(r['slab_pipeline_time'], dtype=np.float64)
                      - np.asarray(r['spinning']['spinning_total_time'], dtype=np.float64)
                      - np.asarray(r['mpi_barrier_time'], dtype=np.float64),
    # each rank's MPI wait per pattern (sum of the pattern's flags), for the group reductions' max/mean
    **{'wait_' + k: _wait_sum(v['flags']) for k, v in WAIT_PATTERNS.items()},
    **{'wait_' + k: _wait_sum(v) for k, v in RACK_SPIN_FLAGS.items()},
}


# ----------------------------------------------------------------------------------------------
# schema -> numpy dtype
# ----------------------------------------------------------------------------------------------
def _fixed_string_dtype(f):
    return np.dtype({'names': ['size', 'buf'], 'formats': ['<u8', f'S{f["max"]}'],
                     'offsets': [0, 8], 'itemsize': f['itemsize']})


def dtype_from_schema(fields, itemsize):
    """Struct dtype built strictly from the explicit offsets in the schema."""
    names, formats, offsets = [], [], []
    for f in fields:
        if f.get('kind') == 'struct':
            base = dtype_from_schema(f['fields'], f['itemsize'])
        elif f.get('kind') == 'fixed_string':
            base = _fixed_string_dtype(f)
        else:
            base = np.dtype(f['dtype'])
        names.append(f['name'])
        formats.append((base, tuple(f['shape'])) if 'shape' in f else base)
        offsets.append(f['offset'])
    return np.dtype({'names': names, 'formats': formats, 'offsets': offsets, 'itemsize': itemsize})


def iter_leaves(fields, path=(), shape=()):
    """Yield (path tuple, base dtype, array shape, is_fixed_string) for every leaf column."""
    for f in fields:
        fshape = shape + tuple(f.get('shape', ()))
        if f.get('kind') == 'struct':
            yield from iter_leaves(f['fields'], path + (f['name'],), fshape)
        elif f.get('kind') == 'fixed_string':
            yield path + (f['name'],), np.dtype(f'S{f["max"]}'), fshape, True
        else:
            yield path + (f['name'],), np.dtype(f['dtype']), fshape, False


def _dig(rec, path):
    a = rec
    for p in path:
        a = a[p]
    return a


# ----------------------------------------------------------------------------------------------
# the run object
# ----------------------------------------------------------------------------------------------
class Run(SimpleNamespace):
    """The run object `fd`: a plain attribute bag holding the config and every cache.  Its repr is a
    one-liner, so a cell that ends in `fd` does not dump the cached data into the notebook."""

    def __repr__(self):
        d = self.__dict__
        return (f'<flagship run: {len(d.get("data", {}))} steps cached, steps {d.get("min_step")}..{d.get("max_step")}, '
                f'{d.get("nranks")} ranks, group axes {d.get("group_axes")}>')


def _ensure(fd):
    """Give an `fd` made by an older version of this module any attributes added since."""
    d = fd.__dict__
    for k in ('data', 'strings', 'state', 'rank', 'groups', 'group_args', 'topo', 'skipped'):
        d.setdefault(k, {})
    d.setdefault('nranks', None)
    d.setdefault('group_axes', ('x',))
    d.setdefault('group_fields', 'scalars')
    d.setdefault('keep_rank_fields', [])
    d.setdefault('job_out', DEFAULT_JOB_OUT)
    d.setdefault('cp_cache', {})
    d.setdefault('nodefile', DEFAULT_NODEFILE)
    d.setdefault('ranks_per_node', 2)
    d.setdefault('node_steps', DEFAULT_NODE_STEPS)
    d.setdefault('node_cache', {})
    _set_group_paths(fd)
    for a in fd.group_axes:
        fd.groups.setdefault(a, {})
        fd.group_args.setdefault(a, {})
    return fd


def _set_group_paths(fd):
    scalar = {p for p, base, shape, is_fs in fd.leaves if not is_fs and shape == ()}
    if fd.group_fields == 'scalars':
        fd.group_paths = sorted(p for p in scalar if not any('.'.join(p).startswith(s) for s in GROUP_SKIP))
    else:
        fd.group_paths = [tuple(n.split('.')) for n in fd.group_fields]
        for p in fd.group_paths:
            assert p in scalar, f'group_fields: {".".join(p)} is not a scalar timing field'


def open_run(max_step, log_dir=DEFAULT_LOG, min_step=0, group_axes=('x',), group_fields='scalars',
             keep_rank_fields=(), nodefile=DEFAULT_NODEFILE, ranks_per_node=2, node_steps=DEFAULT_NODE_STEPS,
             verbose=False):
    """Load the schema and return the run object `fd` (no step files are read yet).

    max_step   : last step directory that may be opened (inclusive); required, so that a step that
                 is still being written is never touched.
    group_axes : which group reductions to build: 'x' (fixed x), 'z' (fixed z), or both.
    group_fields: 'scalars' (every non-array, non-string timing field except GROUP_SKIP) or a list
                 of dotted names.  Costs about n_fields*3*n_groups*4 bytes per step.
    keep_rank_fields: dotted names whose full per-rank values are also kept (n_ranks*8 B per step).
    nodefile   : the job hostfile that maps ranks to nodes (see DEFAULT_NODEFILE; None = no node names).
    ranks_per_node: ranks per hostfile line (2 for the flagship).
    node_steps : (first, last) steps that ran with that nodefile; the node-placement analyses use only these
                 steps by default."""
    assert max_step is not None, 'set max_step to the last step dir that is safe to read'
    log = Path(log_dir)
    with open(log / 'timing_schema.json', 'r') as fp:
        schema = json.load(fp)
    assert schema['format'] == 'abacus-timing-schema' and schema['schema_version'] == 1
    dtype = dtype_from_schema(schema['fields'], schema['itemsize'])
    assert dtype.itemsize == schema['itemsize']
    leaves = list(iter_leaves(schema['fields']))
    fd = Run(log=log, schema=schema, dtype=dtype, leaves=leaves, min_step=min_step,
             max_step=max_step, group_axes=tuple(group_axes), group_fields=group_fields,
             keep_rank_fields=list(keep_rank_fields), nodefile=nodefile and Path(nodefile),
             ranks_per_node=ranks_per_node, node_steps=node_steps)
    _ensure(fd)
    if verbose:
        print(f'schema hash {schema["schema_hash"]} | itemsize {dtype.itemsize} | {len(leaves)} leaf fields')
    return fd


def configure(fd, **kw):
    """Change run options on an existing `fd` (min_step, max_step, group_axes, group_fields,
    keep_rank_fields, nodefile, ranks_per_node, node_steps).  Adding a group axis or changing the group fields makes the next read_steps
    re-read the steps that lack them."""
    ok = {'min_step', 'max_step', 'group_axes', 'group_fields', 'keep_rank_fields', 'job_out', 'nodefile',
          'ranks_per_node', 'node_steps'}
    bad = set(kw) - ok
    assert not bad, f'unknown option(s) {sorted(bad)}; valid: {sorted(ok)}'
    if 'max_step' in kw:
        assert kw['max_step'] is not None, 'max_step must be set'
    for k, v in kw.items():
        setattr(fd, k, tuple(v) if k == 'group_axes' else v)
    if 'nodefile' in kw and kw['nodefile'] is not None:
        fd.nodefile = Path(kw['nodefile'])
    fd.node_cache = {}
    if 'job_out' in kw:
        fd.cp_cache = {}                 # re-scan the logs on the next checkpoint_steps()
    _ensure(fd)     # returns None on purpose: a notebook cell that ends in this call must not echo `fd`


# ----------------------------------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------------------------------
def list_steps(fd):
    """Sorted [(step, time.bin path)] for step dirs in [min_step, max_step] that have a time.bin."""
    out = []
    for d in fd.log.iterdir():
        m = re.fullmatch(r'step(\d+)', d.name)
        if m and d.is_dir() and fd.min_step <= int(m.group(1)) <= fd.max_step:
            fn = d / f'{d.name}.time.bin'
            if fn.is_file():
                out.append((int(m.group(1)), fn))
    return sorted(out)


def read_state_numeric(step_fn):
    """All numeric `key = value` pairs in the sibling stepNNNN.state text file."""
    st = step_fn.parent / (step_fn.parent.name + '.state')
    out = {}
    try:
        with open(st, 'r') as fp:
            for line in fp:
                m = re.match(r'\s*([A-Za-z_]\w*)\s*=\s*(\S+)\s*$', line)
                if m:
                    try:
                        out[m.group(1)] = float(m.group(2))
                    except ValueError:
                        pass
    except OSError:
        pass
    return out


def _reduce(a):
    a = a.astype(np.float64, copy=False)
    return np.stack([a.min(0), a.mean(0), a.max(0), a.sum(0)])


def read_steps(fd, verbose=True):
    """Read every .time.bin in range that is not yet cached, one at a time, and keep every leaf
    field reduced across ranks to (min, mean, max, sum), plus the group reductions for fd.group_axes.
    Array-valued leaves keep their trailing shape; fixed strings keep rank 0's value.  uint64 is
    stored as float64 (exact below 2^53)."""
    _ensure(fd)
    if getattr(fd, 'derived_version', None) != GROUP_DERIVED_VERSION:
        if any(fd.groups.get(a) for a in fd.groups):
            print('the cached group reductions use an older definition of the derived fields (e.g. busy): '
                  'rebuilding them, which re-reads every cached step once')
        for a in list(fd.groups):
            fd.groups[a] = {}
        for a in list(fd.group_args):
            fd.group_args[a] = {}
        fd.derived_version = GROUP_DERIVED_VERSION
        _ensure(fd)
    steps = list_steps(fd)
    if verbose:
        print(f'{len(steps)} step dirs in [{fd.min_step}, {fd.max_step}] with a time.bin'
              + (f'; first {steps[0][0]}, last {steps[-1][0]}' if steps else ''))
    for step, fn in steps:
        if step in fd.data and all(step in fd.groups[a] for a in fd.group_axes):
            continue
        size = fn.stat().st_size
        if size == 0 or size % fd.dtype.itemsize:
            fd.skipped[step] = f'size {size} not a multiple of itemsize (still being written?)'
            continue
        nr = size // fd.dtype.itemsize
        if fd.nranks is None:
            fd.nranks = nr
        if nr != fd.nranks:
            fd.skipped[step] = f'{nr} ranks != {fd.nranks}'
            continue
        rec = np.fromfile(fn, dtype=fd.dtype)       # one read-only pass over the whole file
        ranks_ok = np.array_equal(rec['rank'], np.arange(nr, dtype=rec['rank'].dtype))
        step_ok = bool((rec['full_step_number'] == step).all())
        size_x, size_z = int(rec['mpi_size_x'][0]), int(rec['mpi_size_z'][0])
        grid_ok = (size_x * size_z == nr and (rec['mpi_size_x'] == size_x).all()
                   and (rec['mpi_size_z'] == size_z).all()
                   and np.array_equal(rec['rank_z'], np.arange(nr) % size_z))     # i.e. rank = x*size_z + z
        if not (ranks_ok and step_ok and grid_ok):
            fd.skipped[step] = (f'validation failed (rank column ok={ranks_ok}, full_step_number ok={step_ok}, '
                                f'rank grid ok={grid_ok})')
            continue

        stats, strs = {}, {}
        for path, base, shape, is_fs in fd.leaves:
            col = _dig(rec, path)
            if is_fs:
                strs[path] = np.asarray(col['buf'])[0]
            else:
                stats[path] = _reduce(np.asarray(col))
        fd.data[step], fd.strings[step] = stats, strs
        fd.state[step] = read_state_numeric(fn)
        for name in fd.keep_rank_fields:
            fd.rank.setdefault(name, {})[step] = np.array(_dig(rec, tuple(name.split('.'))))
        for k in list(WAIT_PATTERNS) + list(RACK_SPIN_FLAGS):   # every rank's summed wait per pattern / direction (float32)
            fd.rank.setdefault('wait_' + k, {})[step] = GROUP_DERIVED['wait_' + k](rec).astype(np.float32)

        if not fd.topo:
            fd.topo.update(size_x=size_x, size_z=size_z,
                           node=np.array([b.decode(errors='replace')
                                          for b in np.asarray(rec['node_string']['buf'])]))
        for axn in fd.group_axes:
            red = 1 if axn == 'x' else 0            # reduce over z for fixed-x groups, over x for fixed-z
            gstat, garg = {}, {}
            for path in fd.group_paths:
                gstat[path] = _group_reduce(np.asarray(_dig(rec, path), dtype=np.float64).reshape(size_x, size_z), red)
            for name, fdv in GROUP_DERIVED.items():
                g = fdv(rec).reshape(size_x, size_z)
                gstat[(name,)] = _group_reduce(g, red)
                if name == 'busy':
                    garg['busy_argmax'] = g.argmax(red).astype(np.int16)    # busiest rank within each group
            bar_arg = np.asarray(rec['mpi_barrier_time'], dtype=np.float64).reshape(size_x, size_z).argmin(red)
            garg['barrier_argmin'] = bar_arg.astype(np.int16)                  # least slack
            for k, v in WAIT_PATTERNS.items():          # each pattern's wait on the least-slack rank of the group
                w = GROUP_DERIVED['wait_' + k](rec).reshape(size_x, size_z)
                c = (np.take_along_axis(w, bar_arg[:, None], axis=1)[:, 0] if red == 1
                     else np.take_along_axis(w, bar_arg[None, :], axis=0)[0])
                gstat[('wait_' + k + '_crit',)] = np.stack([c, c, c]).astype(np.float32)   # same value for every stat
            fd.groups[axn][step], fd.group_args[axn][step] = gstat, garg
        del rec

    if verbose:
        print(f'{len(selected(fd))} steps loaded ({fd.nranks} ranks); skipped {len(fd.skipped)}: '
              f'{dict(list(fd.skipped.items())[:5])}')
        if fd.topo:
            print(f'rank grid {fd.topo["size_x"]} x {fd.topo["size_z"]} (x, z); group axes {fd.group_axes}: '
                  f'{len(fd.group_paths) + len(GROUP_DERIVED)} fields x {GROUP_STATS} per group')
    # returns None on purpose (see configure)


def _group_reduce(g, red):
    return np.stack([g.min(red), g.mean(red), g.max(red)]).astype(np.float32)   # (3, n_groups)


# ----------------------------------------------------------------------------------------------
# accessors
# ----------------------------------------------------------------------------------------------
def selected(fd):
    """Loaded steps inside [min_step, max_step], sorted."""
    return np.array(sorted(s for s in fd.data if fd.min_step <= s <= fd.max_step))


def get(fd, name, stat='max'):
    """Array over selected steps of the all-rank reduction of field `name` (dotted path), shape
    (n_steps,) + leaf shape.  stat in FLAG_STATS, e.g. get(fd, 'slab_pipeline.kick_time', 'mean')."""
    path, k = tuple(name.split('.')), FLAG_STATS.index(stat)
    return np.array([fd.data[s][path][k] for s in selected(fd)])


def state_col(fd, key):
    """Array over selected steps of a numeric .state key (nan where absent)."""
    return np.array([fd.state[s].get(key, np.nan) for s in selected(fd)])


def group_steps(fd, axis='x'):
    """Selected steps that have group reductions for `axis`, sorted (the rows of group_get)."""
    return np.array([s for s in selected(fd) if s in fd.groups.get(axis, {})])


def group_get(fd, name, stat='max', axis='x'):
    """(n_steps, n_groups) array of field `name` (dotted path, or 'busy') reduced over each group's
    ranks with `stat` in GROUP_STATS.  axis 'x': one column per x index (size_z ranks each).

    `name` may also be a ratio 'numerator/denominator' of two fields, e.g.
    'io_stats.total_write_bytes/io_stats.total_write_time'; each side is reduced with `stat`, and the
    result is NaN where the denominator is not positive.  With stat='mean' a ratio of two sums over
    the same ranks is exactly the group's aggregate ratio (mean/mean = sum/sum); with 'min' or 'max'
    it is a ratio of two different ranks' values and means little."""
    if '/' in name:
        num, den = (s.strip() for s in name.split('/', 1))
        a = group_get(fd, num, stat, axis).astype(np.float64)
        b = group_get(fd, den, stat, axis).astype(np.float64)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(b > 0, a / b, np.nan)
    path, k = tuple(name.split('.')), GROUP_STATS.index(stat)
    return np.array([fd.groups[axis][s][path][k] for s in group_steps(fd, axis)])


def node_names(fd):
    """Array (n_ranks,) of node names (e.g. 'x4516c6s0b0n0') from fd.nodefile: rank r is on hostfile line
    r // fd.ranks_per_node.  Valid for the steps that ran with that nodefile (fd.node_steps).  Needs
    read_steps(fd) to have run (for the rank count)."""
    if fd.nodefile is None:
        raise ValueError("no node names: the timing files hold no hostnames; set nodefile= (open_run or configure)")
    if not fd.topo:
        raise ValueError('read_steps(fd) first (the rank count is needed)')
    nr = fd.topo['size_x'] * fd.topo['size_z']
    key = (str(fd.nodefile), fd.ranks_per_node, nr)
    if key not in fd.node_cache:
        with open(fd.nodefile, 'r') as fp:
            hosts = [ln.split()[0].split('.')[0] for ln in fp if ln.strip()]
        if len(hosts) * fd.ranks_per_node != nr:
            raise ValueError(f'{fd.nodefile} has {len(hosts)} hosts x {fd.ranks_per_node} ranks/node '
                             f'!= {nr} ranks')
        if not any(_NODE_RE.fullmatch(h) for h in hosts):
            raise ValueError(f'{fd.nodefile}: no host name looks like x4516c6s0b0n0 (first: {hosts[:2]})')
        fd.node_cache[key] = np.repeat(np.array(hosts), fd.ranks_per_node)
    return fd.node_cache[key]


def _step_mask(fd, steps, step_range):
    """Boolean mask over `steps` for a step_range option: 'nodefile' (default) = fd.node_steps, None = all,
    or an inclusive (first, last)."""
    if step_range is None:
        return np.ones(len(steps), bool)
    lo, hi = fd.node_steps if (isinstance(step_range, str) and step_range == 'nodefile') else step_range
    return (steps >= lo) & (steps <= hi)


def group_ranks(fd, g, axis='x'):
    """World ranks in group g: axis 'x' -> all z at x=g; axis 'z' -> all x at z=g."""
    sx, sz = fd.topo['size_x'], fd.topo['size_z']
    return np.arange(sz) + g * sz if axis == 'x' else np.arange(sx) * sz + g


def find(fd, substr=''):
    """Print every available timing field (dotted name, shape) containing substr."""
    for path, base, shape, is_fs in fd.leaves:
        n = '.'.join(path)
        if substr in n:
            print(f'{n:60s} {str(base):8s} {shape}{"  (string)" if is_fs else ""}')


# ----------------------------------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------------------------------
_CP_RE = re.compile(r'^# (\S+ \S+) Checkpoint step (\d+) (written|flushed)')
_ROW_RE = re.compile(r'^\s*(\d+)\s+\d+\.\d+\s+[\d.]+\s+[\d.]+\s')


def _parse_checkpoint_logs(job_out, run_name):
    """Checkpoint events from the job logs <job_out>/*/sim_*.out whose first 200 lines name `run_name`.
    The log carries '# <date> <time> Checkpoint step N written ...' (the state at the end of step N is
    in the node-local cache) and '... flushed ...' (safe on the parallel file system) lines between the
    per-step table rows.  A flush line sits between rows M and M+1, so it completed during step M+1.
    Returns (sorted steps written, sorted steps during which a flush completed, job ids scanned)."""
    written, flush_done, jobs = set(), set(), []
    try:
        files = sorted(Path(job_out).glob('*/sim_*.out'))
    except OSError:
        return [], [], []
    for f in files:
        try:
            with open(f, 'r', errors='replace') as fp:
                head = [next(fp, '') for _ in range(200)]
                if not any(run_name in line for line in head):
                    continue
                jobs.append(f.parent.name)
                last = None
                for line in itertools.chain(head, fp):
                    m = _CP_RE.match(line)
                    if m:
                        if m.group(3) == 'written':
                            written.add(int(m.group(2)))
                        elif last is not None:
                            flush_done.add(last + 1)
                        continue
                    r = _ROW_RE.match(line)
                    if r:
                        last = int(r.group(1))
        except OSError:
            continue
    return sorted(written), sorted(flush_done), jobs


def checkpoint_steps(fd, source='auto', refresh=False, verbose=False):
    """Steps at which a checkpoint was written ('written': the state at the end of that step) and the
    steps during which a flush of one to the parallel file system completed ('flush_done').
    source: 'logs'   parse the "Checkpoint step N written/flushed" lines in the job logs under fd.job_out
                     (only logs whose text names this run);
            'timers' steps where the SCR Start_output timer is nonzero (no flush information);
            'auto'   logs if they yield any checkpoint, else timers.
    Cached on fd; refresh=True re-reads.  Returns {'written': [...], 'flush_done': [...], 'source': str}."""
    _ensure(fd)
    if refresh or source not in fd.cp_cache:
        out = None
        if source in ('logs', 'auto'):
            w, f, jobs = _parse_checkpoint_logs(fd.job_out, fd.log.parent.name)
            if w or source == 'logs':
                out = dict(written=w, flush_done=f, source=f'job logs ({len(jobs)} matching under {fd.job_out})')
        if out is None:
            s = selected(fd)
            v = get(fd, 'scr_start_output_time', 'max') if len(s) else np.array([])
            out = dict(written=[int(x) for x in s[v > 0.05]], flush_done=[], source='scr_start_output_time > 0.05 s')
        fd.cp_cache[source] = out
    out = fd.cp_cache[source]
    if verbose:
        print(f'checkpoints from {out["source"]}: written at steps {out["written"]}; flush completed during {out["flush_done"]}')
    return out


def _mark_checkpoints(ax, fd, steps, mode='written'):
    """Light gray dotted vertical lines at the checkpoint steps inside the plotted step range.
    mode 'written' marks the steps checkpoints were written; 'both' also marks (sparser dots) the steps
    during which a flush completed; None/False draws nothing.  Never lets a missing log break a plot."""
    if not mode or len(steps) < 2:
        return
    try:
        cp = checkpoint_steps(fd)
    except Exception:
        return
    lo, hi = steps[0], steps[-1]
    for n in cp['written']:
        if lo <= n <= hi:
            ax.axvline(n, color='0.82', ls=':', lw=0.9, zorder=5)
    if mode == 'both':
        for n in cp['flush_done']:
            if lo <= n <= hi:
                ax.axvline(n, color='0.82', ls=(0, (1, 4)), lw=0.9, alpha=0.8, zorder=5)


def _add_top_z(a, steps, z):
    """Top axis: the redshift at the same evenly spaced round-number step ticks as the bottom axis,
    each top tick labelled with the redshift of its step.  Skipped without a finite Redshift."""
    from matplotlib.ticker import MaxNLocator
    if len(steps) < 2 or not np.isfinite(z).all():
        return
    st = MaxNLocator(nbins=7, integer=True, steps=[1, 2, 2.5, 5, 10]).tick_values(steps[0], steps[-1])
    st = st[(st >= steps[0]) & (st <= steps[-1])]
    a.set_xlim(steps[0], steps[-1])
    a.set_xticks(st)
    top = a.twiny()                    # shares the y axis; its x axis sits on top
    top.set_xlim(a.get_xlim())
    top.set_xticks(st)
    top.set_xticklabels([f'{v:.3g}' for v in np.interp(st, steps, z)])
    top.set_xlabel('z')


def plot_overview(fd):
    """Six-panel overview of the all-rank reductions vs step, with redshift along the top."""
    import matplotlib.pyplot as plt
    fsteps = selected(fd)
    zs = state_col(fd, 'Redshift')
    have_z = len(zs) > 1 and np.isfinite(zs).all()
    if not have_z:
        print('top (redshift) axis skipped: need >1 step with a finite Redshift state key')
    x = fsteps
    fig, ax = plt.subplots(2, 3, figsize=(15, 7), sharex=False)
    ax = ax.ravel()
    for n, lab in [('singlestep_total_time', 'step wall'), ('slab_pipeline_time', 'slab pipeline'),
                   ('convolution_time', 'convolution'), ('finish_io_time', 'finish IO')]:
        ax[0].plot(x, get(fd, n, 'max'), label=lab)
    ax[0].set_ylabel('s (max rank)'); ax[0].set_title('Step timing', pad=24); ax[0].set_yscale('log'); ax[0].legend(fontsize=7)
    # io_stats summary.  Left axis (solid): IO-thread seconds per step for the slowest rank, from the io_stats
    # times (summed over the IO threads, so they can exceed wall time; total_write includes the checksum time).
    # Right axis (dashed): aggregate bandwidth in GB per IO-thread-second = bytes summed over all ranks / time
    # summed over all ranks.  A series that is zero at every step cannot be drawn on a log axis and is skipped.
    io_times = [('total_read_time', 'read'), ('total_write_time', 'write (incl. checksum)'), ('checksum_time', 'checksum'),
                ('asdf_aggregate_time', 'ASDF aggregate'), ('asdf_crc32_time', 'ASDF crc32'),
                ('asdf_file_write_time', 'ASDF file write')]
    io_bw = [('total_read_bytes', 'total_read_time', 'read'), ('total_write_bytes', 'total_write_time', 'write'),
             ('asdf_bytes_out', 'asdf_file_write_time', 'ASDF file write')]
    io_color = {n: f'C{k}' for k, (n, _) in enumerate(io_times)}
    handles = []
    for n, lab in io_times:
        y = get(fd, 'io_stats.' + n, 'max')
        if np.any(y > 0):
            handles += ax[1].plot(x, y, color=io_color[n], label=lab)
    ax[1].set_ylabel('IO-thread s (max rank)'); ax[1].set_yscale('log')
    ax1b = ax[1].twinx()
    for nb, nt, lab in io_bw:
        bsum, tsum = get(fd, 'io_stats.' + nb, 'sum'), get(fd, 'io_stats.' + nt, 'sum')
        with np.errstate(divide='ignore', invalid='ignore'):
            y = np.where(tsum > 0, bsum / tsum / 1e9, np.nan)
        if np.isfinite(y).any() and np.nanmax(y) > 0:
            handles += ax1b.plot(x, y, color=io_color[nt], ls='--', lw=1.0, label=f'{lab} GB/s')
    ax1b.set_ylabel('GB / IO-thread-s (all ranks)'); ax1b.set_yscale('log')
    ax[1].set_title('IO stats', pad=24)
    if handles:
        ax[1].legend(handles=handles, fontsize=6, ncol=2, loc='best', handlelength=1.6, columnspacing=0.8)
    for n, lab in [('spinning.spinning_total_time', 'spinning'), ('spinning.waiting_for_gpu_time', 'waiting GPU'),
                   ('spinning.waiting_for_io_time', 'waiting IO')]:
        ax[2].plot(x, get(fd, n, 'mean'), label=lab)
    ax[2].plot(x, get(fd, 'slab_pipeline_time', 'max') / get(fd, 'slab_pipeline_time', 'mean'),
               label='pipeline max/mean', ls='--')
    ax[2].set_title('Spin / imbalance (mean over ranks)', pad=24); ax[2].set_yscale('log'); ax[2].legend(fontsize=7)
    ax[3].plot(x, get(fd, 'rms_fractional_overdensity', 'max'), label='rms overdensity')
    ax[3].plot(x, get(fd, 'rms_velocity', 'max'), label='rms velocity')
    ax[3].set_title('Clustering sanity', pad=24); ax[3].set_yscale('log'); ax[3].legend(fontsize=7)
    GiB = 2**30
    ax[4].plot(x, get(fd, 'max_rss_bytes', 'max') / GiB, label='max RSS')
    ax[4].plot(x, get(fd, 'memwatch.shm_used_bytes_max', 'max') / GiB, label='ramdisk used (max)')
    ax[4].plot(x, get(fd, 'memwatch.mem_available_bytes_min', 'min') / GiB, label='MemAvailable (min)')
    ax[4].set_ylabel('GiB'); ax[4].set_title('Memory (worst rank)', pad=24); ax[4].legend(fontsize=7)
    ax[5].plot(x, get(fd, 'max_cell_size', 'max'), label='max cell size')
    ax[5].plot(x, get(fd, 'mapforest.overflow_laps', 'max'), label='mapforest overflow laps (want 0)')
    ax[5].set_title('Cell occupancy / MAP forest', pad=24); ax[5].set_yscale('symlog'); ax[5].legend(fontsize=7)
    for a in ax:
        a.set_xlabel('step')
        if have_z:
            _add_top_z(a, fsteps, zs)
    plt.tight_layout()
    plt.show()

    names = [('singlestep_total_time', 'max'), ('slab_pipeline_time', 'max'), ('convolution_time', 'max'),
             ('spinning.spinning_total_time', 'mean'), ('max_rss_bytes', 'max')]
    print('step      z ' + ' '.join(f'{n[:18]:>18s}' for n, _ in names))
    cols = [get(fd, n, s) for n, s in names]
    for i in range(max(0, len(fsteps) - 10), len(fsteps)):
        print(f'{fsteps[i]:4d} {zs[i]:6.3f} ' + ' '.join(f'{c[i]:18.5g}' for c in cols))


# Presentation defaults for plot_components, by struct name.
COMPONENT_TITLES = {'slab_pipeline': 'Pipeline stages'}
COMPONENT_UNITS = {'slab_pipeline': 's', 'spinning': 's'}
# Top-level fields drawn with a struct's own members (in black/gray).  The barrier wait that follows the event
# loop is a top-level field, but it is inside slab_pipeline_time, so it belongs with the pipeline stages.
COMPONENT_EXTRAS = {'slab_pipeline': [('mpi_barrier_time', 'mpi barrier (post-pipeline)')]}


def plot_components(fd, struct, stat='max', extra=None, unit=None, title=None):
    """Every first-level member of a struct vs step in one figure: all scalar fields directly inside `struct`
    (e.g. 'slab_pipeline', 'spinning', 'io_stats', 'near_force'; a field added to the schema appears
    automatically), reduced over all ranks with `stat` ('max', 'mean', 'min'), log y, redshift along the top.
    Arrays and nested structs inside the struct are not plotted (they are named in the note).
    extra: top-level fields to draw with the members, as names or (name, label) pairs; the default comes from
    COMPONENT_EXTRAS (for 'slab_pipeline' it is mpi_barrier_time, drawn in black, so that the lines sum to
    slab_pipeline_time with unaccounted_pipeline_time as the leftover; that sum holds for stat='mean', since a
    sum of per-stage maxima comes from different ranks).  Pass extra=() for none.  unit and title default from
    COMPONENT_UNITS and COMPONENT_TITLES.  A series that is zero at every step cannot be drawn on a log axis;
    it is named in the note."""
    import matplotlib.pyplot as plt
    structs = sorted({p[0] for p, base, shape, is_fs in fd.leaves if len(p) > 1})
    if struct not in structs:
        raise ValueError(f'unknown struct {struct!r}; the top-level structs are: {", ".join(structs)}')
    fsteps = selected(fd)
    zs = state_col(fd, 'Redshift')
    members = [p for p, base, shape, is_fs in fd.leaves
               if p[0] == struct and len(p) == 2 and shape == () and not is_fs]
    skipped = sorted({p[1] for p, base, shape, is_fs in fd.leaves
                      if p[0] == struct and (len(p) > 2 or shape != () or is_fs)})
    extras = [(e, e) if isinstance(e, str) else tuple(e) for e in (COMPONENT_EXTRAS.get(struct, []) if extra is None else extra)]
    unit = COMPONENT_UNITS.get(struct) if unit is None else unit
    title = COMPONENT_TITLES.get(struct, struct.replace('_', ' ')) if title is None else title
    cmap = plt.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(12, 6.5))
    empty = []
    for k, p in enumerate(members):
        y = get(fd, '.'.join(p), stat)
        if not np.any(y > 0):
            empty.append(p[-1].replace('_time', ''))
            continue
        ax.plot(fsteps, y, label=p[-1].replace('_time', '').replace('_', ' '), color=cmap(k % 20),
                ls='-' if k < 10 else '--' if k < 20 else ':', lw=1.4)
    for j, (name, label) in enumerate(extras):
        y = get(fd, name, stat)
        if np.any(y > 0):
            ax.plot(fsteps, y, label=label, color=['k', '0.35', '0.55', '0.7'][j % 4], lw=1.8)
        else:
            empty.append(name.replace('_time', ''))
    ax.set_yscale('log')
    ax.set_xlabel('step'); ax.set_ylabel(f'{unit} ({stat} over ranks)' if unit else f'{stat} over ranks')
    ax.set_title(title, pad=24)
    note = []
    if empty:
        note.append('all zero, not drawn: ' + ', '.join(empty))
    if skipped:
        note.append('not plotted (array or nested struct): ' + ', '.join(skipped))
    if note:
        ax.text(0.01, 0.99, '\n'.join(note), transform=ax.transAxes, ha='left', va='top', fontsize=7, color='0.4')
    if len(ax.lines):
        ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False,
                  ncol=1 if len(ax.lines) <= 24 else 2)
    if len(fsteps) > 1 and np.isfinite(zs).all():
        _add_top_z(ax, fsteps, zs)
    plt.tight_layout()
    plt.show()


def group_slabs(fd, axis='x'):
    """Slabs processed by each group, for normalizing extensive quantities (time, particles, bytes).
    Slabs are split over the x-ranks as slabs[j] = floor(cpd*(j+1)/size_x) - floor(cpd*j/size_x), so an
    x-group owns 37 or 38 of them (cpd=8775, size_x=232) and its work differs by ~2.7% for that reason alone.
    Every rank of an x-group owns the same slabs, so dividing by this is exact for any statistic over the
    group.  A z-group spans all x-ranks, whose mean is cpd/size_x slabs for every z-group, so it returns that
    constant (dividing by it changes nothing once a heatmap is made relative)."""
    cpd = int(get(fd, 'cpd', 'max')[0])
    sx, sz = fd.topo['size_x'], fd.topo['size_z']
    if axis == 'x':
        j = np.arange(sx)
        return ((cpd * (j + 1)) // sx - (cpd * j) // sx).astype(np.float64)
    return np.full(sz, cpd / sx)


def _check_norm(norm):
    if norm not in ('median', 'mean'):
        raise ValueError(f"norm must be 'median' or 'mean', not {norm!r}")


def _reference(v, norm):
    """Per-step reference over the groups (axis 1) that a relative heatmap divides by: the median (robust to
    a few slow groups, the default) or the mean.  A zero or undefined reference becomes 1 (row left unscaled)."""
    import warnings
    _check_norm(norm)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')              # all-NaN rows, e.g. a ratio with no writes that step
        ref = np.nanmedian(v, axis=1, keepdims=True) if norm == 'median' else np.nanmean(v, axis=1, keepdims=True)
    ref[~(ref != 0)] = 1.0
    return ref


def group_critical(fd, axis='x', norm='median', per_slab=False):
    """Per-step critical group.  slack = mpi_barrier_time, the wait at the barrier after the pipeline:
    the rank that finishes last waits ~0, so the group whose MINIMUM barrier time is smallest is that
    step's critical group.  (busy = time actually working: pipeline - spinning - barrier wait; see
    GROUP_DERIVED.)  rel is busy max over each group's ranks (divided by the group's slab count if
    per_slab) over the per-step median (or mean, norm='mean') across groups.  Returns a dict with steps,
    rel, bar_min, crit (critical group per step) and counts (fraction of steps each group is critical)."""
    steps = group_steps(fd, axis)
    n_groups = fd.topo['size_x'] if axis == 'x' else fd.topo['size_z']
    _check_norm(norm)
    busy_max = group_get(fd, 'busy', 'max', axis)
    bar_min = group_get(fd, 'mpi_barrier_time', 'min', axis)
    crit = bar_min.argmin(axis=1)
    scaled = busy_max / group_slabs(fd, axis)[None, :] if per_slab else busy_max
    return dict(steps=steps, n_groups=n_groups, busy_max=busy_max, bar_min=bar_min,
                rel=scaled / _reference(scaled, norm), crit=crit,
                counts=np.bincount(crit, minlength=n_groups) / len(steps))


def plot_groups(fd, axis='x', top=8, checkpoints='written', norm='median', per_slab=True):
    """Group-versus-step view: relative busy time, slack, the critical group per step and how often
    each group is critical; then spread inside groups, what the critical group waits on, and load;
    then a table of the most often critical groups with where they run.  Reads only cached data.
    The busy-time heatmap and the load scatter compare groups, so by default (per_slab=True) they use work
    per slab, which removes the 37-vs-38-slab stripes; norm is the per-step reference over groups
    ('median' or 'mean').  Slack, the spread inside groups, and the waits are not slab-normalized."""
    import matplotlib.pyplot as plt
    c = group_critical(fd, axis, norm=norm, per_slab=per_slab)
    gsteps, n_groups, rel, bar_min, crit, counts = c['steps'], c['n_groups'], c['rel'], c['bar_min'], c['crit'], c['counts']
    gz = np.array([fd.state[s].get('Redshift', np.nan) for s in gsteps])
    print(f'busy and load comparisons: {"per slab, " if per_slab else ""}relative to the {norm} group')
    bmin, bmax = get(fd, 'mpi_barrier_time', 'min'), get(fd, 'mpi_barrier_time', 'max')
    print(f'{len(gsteps)} steps x {n_groups} {axis}-groups.  mpi_barrier_time over all ranks, median over steps: '
          f'min {np.median(bmin):.3g} s, max {np.median(bmax):.3g} s')
    if bmax.max() == 0:
        print('WARNING: mpi_barrier_time is zero everywhere -- the slack definition is meaningless')
    sb = np.sort(bar_min, axis=1)
    print(f'critical group = argmin of group-min barrier wait; margin over the runner-up: median '
          f'{np.median(sb[:, 1] - sb[:, 0]):.3g} s (small => the choice is ambiguous)')
    print(f'critical group unchanged from the previous step in {np.mean(crit[1:] == crit[:-1]):.0%} of steps; '
          f'{np.count_nonzero(counts)} different groups were critical at least once')

    yy = np.arange(n_groups)
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    vmin, vmax = np.nanpercentile(rel, [1, 99.5])
    m = ax[0, 0].pcolormesh(gsteps, yy, rel.T, shading='auto', vmin=vmin, vmax=vmax, cmap='magma')
    fig.colorbar(m, ax=ax[0, 0], label=f'busy (max in group){" per slab" if per_slab else ""} / {norm} over groups')
    ax[0, 0].set_title(f'Relative busy time per {axis}-group' + (' (per slab)' if per_slab else ''), pad=24)
    m = ax[0, 1].pcolormesh(gsteps, yy, bar_min.T, shading='auto', vmax=np.nanpercentile(bar_min, 99), cmap='viridis')
    fig.colorbar(m, ax=ax[0, 1], label='min barrier wait in group [s]')
    ax[0, 1].set_title('Slack per group (dark = critical)', pad=24)
    ax[1, 0].plot(gsteps, crit, '.', ms=3)
    ax[1, 0].set_title('Critical group at each step', pad=24)
    ax[1, 1].bar(yy, counts)
    ax[1, 1].set_xlabel(f'{axis} group'); ax[1, 1].set_ylabel('fraction of steps critical')
    ax[1, 1].set_title('How often each group is critical')
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.set_ylabel(f'{axis} group'); a.set_xlabel('step'); _add_top_z(a, gsteps, gz)
    for a in (ax[0, 0], ax[0, 1]):                  # the two heatmaps
        _mark_checkpoints(a, fd, gsteps, checkpoints)
    plt.tight_layout(); plt.show()

    # ---- second view: spread inside groups, what the critical group waits on, and load ----
    busy_min = group_get(fd, 'busy', 'min', axis)
    spread = c['busy_max'] - busy_min
    wait_names = ['spinning.checking_preconditions_time', 'spinning.not_enough_ram_time',
                  'spinning.waiting_for_io_time', 'spinning.waiting_for_gpu_time']
    wait_names += ['.'.join(p) for p, base, shape, is_fs in fd.leaves
                   if p[0] == 'spinning' and p[-1].startswith('waiting_for_mpi') and shape == ()]
    rows = np.arange(len(gsteps))
    crit_w, typ_w = [], []
    for n in wait_names:
        w = group_get(fd, n, 'mean', axis)
        crit_w.append(w[rows, crit].mean()); typ_w.append(np.median(w, axis=1).mean())
    crit_w, typ_w = np.array(crit_w), np.array(typ_w)
    show = np.maximum(crit_w, typ_w) > 1e-3
    part = group_get(fd, 'num_particles_node', 'mean', axis)
    if per_slab:
        part = part / group_slabs(fd, axis)[None, :]
    rel_p = part / _reference(part, norm)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    ax[0].plot(gsteps, spread.mean(axis=1), label='mean over groups')
    ax[0].plot(gsteps, np.median(spread, axis=1), label='median')
    ax[0].plot(gsteps, spread.max(axis=1), label='worst group')
    ax[0].set_ylabel('busy max - min inside the group [s]'); ax[0].set_xlabel('step')
    ax[0].set_title('Imbalance among the ranks of a group', pad=24); ax[0].legend(fontsize=7); _add_top_z(ax[0], gsteps, gz)
    yb = np.arange(show.sum())
    ax[1].barh(yb - 0.2, typ_w[show], height=0.4, label='typical group (median)')
    ax[1].barh(yb + 0.2, crit_w[show], height=0.4, label='critical group')
    ax[1].set_yticks(yb)
    ax[1].set_yticklabels([n.split('.')[-1].replace('waiting_for_', '').replace('_time', '')
                           for n, s in zip(wait_names, show) if s], fontsize=7)
    ax[1].set_xlabel('mean wait per step [s]'); ax[1].set_title('What waits: critical vs typical group'); ax[1].legend(fontsize=7)
    ax[2].hexbin(rel_p.ravel(), rel.ravel(), gridsize=60, bins='log', mincnt=1)
    ax[2].set_xlabel(f'group particles{" per slab" if per_slab else ""} / {norm}')
    ax[2].set_ylabel(f'group busy{" per slab" if per_slab else ""} / {norm}')
    ax[2].set_title(f'Load vs busy time (r = {np.corrcoef(rel_p.ravel(), rel.ravel())[0, 1]:.2f})')
    plt.tight_layout(); plt.show()

    print(f'{axis:>3s}  crit-frac  mean rel busy  world ranks      nodes  cabinets')
    for g in np.argsort(counts)[::-1][:top]:
        ranks = group_ranks(fd, g, axis)
        hosts = np.unique(node_names(fd)[ranks]); cabs = sorted({str(h)[:5] for h in hosts})
        print(f'{g:3d}  {counts[g]:9.2f}  {rel[:, g].mean():13.3f}  {ranks[0]:6d}..{ranks[-1]:<6d}  {len(hosts):5d}  '
              f'{len(cabs)} {cabs[:5]}')
    return c


# ----------------------------------------------------------------------------------------------
# MPI wait vs how far apart a group's nodes are
# ----------------------------------------------------------------------------------------------
_NODE_RE = re.compile(r'x(\d{2})(\d{2})c(\d+)s(\d+)b(\d+)n(\d+)')
# Separation of two nodes in rough network-distance units: each step up the hierarchy is about twice as far,
# and nodes in different cabinet rows are far.  Nodes whose names do not parse have no distance (NaN) and are
# left out of every statistic rather than being treated as near or far.
DISTANCE_LEVELS = {0: 'same node', 1: 'same chassis', 2: 'same cabinet', 4: 'same cabinet row', 8: 'different row'}


def _parse_nodes(names):
    """(N, 6) int array of (cabinet row, cabinet column, chassis, slot, board, node) read from Cray EX node
    names like x4516c6s0b0n0 (row 45, column 16, chassis 6, slot 0, board 0, node 0); -1 where a name does not
    parse.  Reading the first two digits as a row and the last two as a column is a guess from the names."""
    out = np.full((len(names), 6), -1, dtype=np.int64)
    for i, nm in enumerate(names):
        m = _NODE_RE.fullmatch(str(nm))
        if m:
            out[i] = [int(g) for g in m.groups()]
    return out


def _pair_distances(P):
    """Distance for every pair of nodes in P (see DISTANCE_LEVELS): 0 same node, 1 same chassis, 2 same cabinet,
    4 same cabinet row, 8 different row; NaN for a pair with a name that did not parse."""
    ok = (P >= 0).all(axis=1)
    def same(cols):
        return (P[:, None, cols] == P[None, :, cols]).all(axis=2)
    row, cab, chassis, node = same([0]), same([0, 1]), same([0, 1, 2]), same([0, 1, 2, 3, 4, 5])
    D = (8.0 * ~row + 4.0 * (row & ~cab) + 2.0 * (cab & ~chassis) + 1.0 * (chassis & ~node))
    D[~(ok[:, None] & ok[None, :])] = np.nan
    return D


def group_distance(fd, axis='x', scheme='hierarchy'):
    """How far apart the nodes of each group's communicator are, from the node names recorded in the timing
    files (the real rank-to-node placement, not the hostfile order).  Per group (axis 'x' = comm_1d_z, 'z' =
    comm_1d_x) over all pairs of its distinct nodes, returns arrays: 'max' and 'mean' of the pair distance,
    'n_nodes', 'n_cabinets', 'n_chassis', 'n_unparsed' (nodes whose name did not parse; pairs involving them
    are left out, and a group with no valid pair has NaN distance).  scheme='hierarchy' gives a pair distance
    1 same chassis, 2 same cabinet, 4 same cabinet row, 8 different row (DISTANCE_LEVELS); the levels above
    'same chassis' are a guess at the network (the real switch and dragonfly-group map is not in the names).
    scheme may instead be a function f(a, b) -> distance (NaN = unknown) of two parsed
    (row, col, chassis, slot, board, node) tuples."""
    if not (scheme == 'hierarchy' or callable(scheme)):
        raise ValueError(f"scheme must be 'hierarchy' or a function of two parsed node tuples, not {scheme!r}")
    n_groups = fd.topo['size_x'] if axis == 'x' else fd.topo['size_z']
    names = node_names(fd)
    out = {k: np.zeros(n_groups) for k in ('max', 'mean', 'n_nodes', 'n_cabinets', 'n_chassis', 'n_unparsed')}
    for g in range(n_groups):
        P = _parse_nodes(np.unique(names[group_ranks(fd, g, axis)]))
        if g == 0 and not (P >= 0).all(axis=1).any():
            raise ValueError(f'node names do not parse as x4516c6s0b0n0: {names[:2]}')
        S = (_pair_distances(P) if scheme == 'hierarchy'
             else np.array([[scheme(tuple(a), tuple(b)) for b in P] for a in P], dtype=float))
        d = S[np.triu_indices(len(P), 1)]
        d = d[np.isfinite(d)]
        out['max'][g] = d.max() if len(d) else np.nan
        out['mean'][g] = d.mean() if len(d) else np.nan
        good = P[(P >= 0).all(axis=1)]
        out['n_nodes'][g] = len(P)
        out['n_unparsed'][g] = len(P) - len(good)
        out['n_cabinets'][g] = len({tuple(r[:2]) for r in good})
        out['n_chassis'][g] = len({tuple(r[:3]) for r in good})
    return out


def _node_distance(P, a, b):
    """Hierarchy distance (see DISTANCE_LEVELS) between nodes P[a] and P[b] (index arrays); NaN where unparsed."""
    ok = (P[a] >= 0).all(axis=1) & (P[b] >= 0).all(axis=1)
    diff = lambda k: (P[a][:, :k] != P[b][:, :k]).any(axis=1)
    d = np.where(diff(1), 8.0, np.where(diff(2), 4.0, np.where(diff(3), 2.0, np.where(diff(6), 1.0, 0.0))))
    d[~ok] = np.nan
    return d


def rank_pattern_distance(fd, pattern):
    """How far apart the nodes are that a communication pattern (WAIT_PATTERNS) really connects, per group.
    For each rank, the distance (DISTANCE_LEVELS) to each of its partners in the pattern, leaving out partners on
    its own node (no network) and unparsed names: 'manifest' = ranks x-1 and x+1 (same z), 'z_ring' = z-1 and
    z+1 (same x), the transposes = every other rank of the rank's fixed-z ('xtranspose') or fixed-x
    ('ztranspose') group.  A rank's value is the max and mean over its partners; a group's is the max of the
    ranks' maxes and the mean of their means ('max', 'mean', NaN if no network pair).  The communication
    is not recorded anywhere; this follows from the communicators in the source.
    rank_pattern_distance returns the per-rank (max, mean) arrays, pattern_distance the group reductions."""
    kind = WAIT_PATTERNS[pattern]['partners']
    sx, sz = fd.topo['size_x'], fd.topo['size_z']
    names = node_names(fd)
    uniq, node = np.unique(names, return_inverse=True)
    P = _parse_nodes(uniq)
    if not (P >= 0).all(axis=1).any():
        raise ValueError(f'node names do not parse as x4516c6s0b0n0: {names[:2]}')
    X, Z = np.divmod(np.arange(sx * sz), sz)
    rmax, rmean = np.full(sx * sz, np.nan), np.full(sx * sz, np.nan)

    def fill(ranks, partners):          # partners: (len(ranks), k) world ranks
        a = np.repeat(node[ranks], partners.shape[1])
        d = _node_distance(P, a, node[partners].ravel()).reshape(partners.shape)
        d = np.where(d == 0, np.nan, d)
        ok = np.isfinite(d).any(axis=1)
        mx, mn = np.full(len(ranks), np.nan), np.full(len(ranks), np.nan)
        mx[ok], mn[ok] = np.nanmax(d[ok], axis=1), np.nanmean(d[ok], axis=1)
        rmax[ranks], rmean[ranks] = mx, mn

    if kind == 'x_ring':
        r = np.arange(sx * sz)
        fill(r, np.stack([((X - 1) % sx) * sz + Z, ((X + 1) % sx) * sz + Z], axis=1))
    elif kind == 'z_ring':
        r = np.arange(sx * sz)
        fill(r, np.stack([X * sz + (Z - 1) % sz, X * sz + (Z + 1) % sz], axis=1))
    else:
        gaxis = 'z' if kind == 'x_all' else 'x'          # the group whose members all talk to each other
        for g in range(sx if gaxis == 'x' else sz):
            r = group_ranks(fd, g, gaxis)
            fill(r, np.tile(r, (len(r), 1)))
    return rmax, rmean


def pattern_distance(fd, pattern, axis='x'):
    """Per group (axis 'x' fixed x, 'z' fixed z) the max of its ranks' max partner distance and the mean of their
    mean partner distance for a communication pattern: {'max', 'mean'} (see rank_pattern_distance)."""
    rmax, rmean = rank_pattern_distance(fd, pattern)
    n_groups = fd.topo['size_x'] if axis == 'x' else fd.topo['size_z']
    out = {'max': np.full(n_groups, np.nan), 'mean': np.full(n_groups, np.nan)}
    for g in range(n_groups):
        r = group_ranks(fd, g, axis)
        if np.isfinite(rmax[r]).any():
            out['max'][g], out['mean'][g] = np.nanmax(rmax[r]), np.nanmean(rmean[r])
    return out


def _spearman(a, b, n_perm=2000, seed=0):
    """Spearman rank correlation of a and b (average ranks for ties) and a two-sided permutation p-value."""
    def rank(x):
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        return (np.cumsum(counts) - (counts - 1) / 2.0)[inv]
    ok = np.isfinite(a) & np.isfinite(b)
    ra, rb = rank(np.asarray(a)[ok]), rank(np.asarray(b)[ok])
    if len(ra) < 3 or ra.std() == 0 or rb.std() == 0:
        return np.nan, np.nan
    rho = np.corrcoef(ra, rb)[0, 1]
    rng = np.random.default_rng(seed)
    perm = np.array([np.corrcoef(ra, rng.permutation(rb))[0, 1] for _ in range(n_perm)])
    return rho, (1 + np.sum(np.abs(perm) >= abs(rho))) / (n_perm + 1)


def plot_wait_vs_distance(fd, axis='x', patterns=None, **kwargs):
    """For each communication pattern in `patterns` (default PATTERNS_BY_AXIS[axis]; see WAIT_PATTERNS), one
    figure of the groups' MPI wait and slack against the distance between the nodes that pattern connects
    (pattern_distance).  Other keywords go to plot_wait_pattern.  Returns {pattern: result dict}."""
    pats = PATTERNS_BY_AXIS[axis] if patterns is None else ([patterns] if isinstance(patterns, str) else patterns)
    return {p: plot_wait_pattern(fd, axis, p, **kwargs) for p in pats}


def plot_spin_vs_distance(fd, axis='x', patterns=None, exclude_checkpoints=True, per_slab=False, flush_steps=10,
                          step_range='nodefile'):
    """The groups' MPI spin time on each communication pattern against the distance between the nodes the pattern
    connects (pattern_distance), two ways of representing a group's spin:
      critical: the spin (sum of the pattern's flags) of the group's least-slack rank, i.e. the rank with the
                smallest mpi_barrier_time that step, which reaches the barrier last and sets the group's pace;
      max:      the largest spin of any rank in the group, which is not hidden if the critical rank happens to
                talk to nearer nodes than another rank does.
    Each value is the group's median over the used steps (exclude_checkpoints and step_range as in
    plot_wait_pattern; per_slab divides by the group's slab count).  Rows: critical, max; columns: against the
    group's maximum partner distance (boxes + points), against its mean partner distance; the right panel is
    critical against max, coloured by distance.  Assumes a rank waiting for a peer is counted in its spin flags.
    Returns {pattern: dict}.  Patterns default to PATTERNS_BY_AXIS[axis]."""
    import warnings
    import matplotlib.pyplot as plt
    pats = PATTERNS_BY_AXIS[axis] if patterns is None else ([patterns] if isinstance(patterns, str) else patterns)
    steps = group_steps(fd, axis)
    in_range = _step_mask(fd, steps, step_range)
    keep = in_range.copy()
    if not keep.any():
        raise ValueError(f'no cached steps in step_range {step_range!r} (node_steps {fd.node_steps}); '
                         f'cached group steps are {steps[:1]}..{steps[-1:]}')
    if exclude_checkpoints:
        cp = checkpoint_steps(fd)
        for w in cp['written']:
            f = next((x for x in cp['flush_done'] if x >= w), w + flush_steps)
            keep &= ~((steps >= w) & (steps <= f))
    if not keep.any():
        raise ValueError('every step in step_range was excluded as a checkpoint window; pass exclude_checkpoints=False')
    norm = group_slabs(fd, axis)[None, :] if per_slab else 1.0
    rng = np.random.default_rng(0)
    results = {}
    for pattern in pats:
        dist = pattern_distance(fd, pattern, axis)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            crit = np.nanmedian((group_get(fd, f'wait_{pattern}_crit', 'max', axis) / norm)[keep], axis=0)
            mx = np.nanmedian((group_get(fd, f'wait_{pattern}', 'max', axis) / norm)[keep], axis=0)
        cats = np.unique(dist['max'][np.isfinite(dist['max'])])
        ylab = f'MPI spin, {pattern} [s per step{", per slab" if per_slab else ""}]'
        fig = plt.figure(figsize=(17, 9))
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.9])
        ax = np.array([[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(2)])
        axr = fig.add_subplot(gs[:, 2])
        corr = {}
        for r, (yv, nm) in enumerate(((crit, 'critical (least-slack) rank'), (mx, 'max rank'))):
            rho_m, p_m = _spearman(dist['max'], yv)
            rho_a, p_a = _spearman(dist['mean'], yv)
            corr[nm] = dict(rho_max=rho_m, p_max=p_m, rho_mean=rho_a, p_mean=p_a)
            a = ax[r, 0]
            a.boxplot([yv[(dist['max'] == c) & np.isfinite(yv)] for c in cats], positions=cats, widths=0.5,
                      showfliers=False, medianprops=dict(color='C3'))
            a.scatter(dist['max'] + rng.uniform(-0.18, 0.18, len(yv)), yv, s=9, alpha=0.5, color='C0', zorder=3)
            a.set_xticks(cats)
            a.set_xticklabels([f'{c:g}: {DISTANCE_LEVELS.get(c, "")}\n(n={int((dist["max"] == c).sum())})'
                               for c in cats], fontsize=8)
            a.set_xlabel("group's maximum partner distance"); a.set_ylabel(f'{nm}\n{ylab}', fontsize=9)
            a.set_title(f'{axis}-groups, {nm}: Spearman {rho_m:.2f}, p={p_m:.3g}', fontsize=10)
            a = ax[r, 1]
            a.scatter(dist['mean'], yv, s=12, alpha=0.6)
            ok = np.isfinite(yv) & np.isfinite(dist['mean'])
            if ok.sum() > 2 and np.ptp(dist['mean'][ok]) > 0:
                slope, icpt = np.polyfit(dist['mean'][ok], yv[ok], 1)
                xs = np.array([dist['mean'][ok].min(), dist['mean'][ok].max()])
                a.plot(xs, slope * xs + icpt, color='C3', lw=1.2)
            a.set_xlabel("group's mean partner distance"); a.set_ylabel(f'{nm}\n{ylab}', fontsize=9)
            a.set_title(f'vs mean distance: Spearman {rho_a:.2f}, p={p_a:.3g}', fontsize=10)
        sc = axr.scatter(mx, crit, c=dist['max'], cmap='viridis', s=18, alpha=0.8,
                         vmin=cats.min() if len(cats) else None, vmax=cats.max() if len(cats) else None)
        lim = np.nanmax(np.concatenate([mx, crit])) if np.isfinite(mx).any() else 1.0
        axr.plot([0, lim], [0, lim], 'k--', lw=0.8)
        if len(cats):
            cb = fig.colorbar(sc, ax=axr, ticks=cats); cb.set_label("group's maximum partner distance")
        axr.set_xlabel('max-rank spin [s]'); axr.set_ylabel('critical-rank spin [s]')
        axr.set_title('critical vs max (dashed: equal)', fontsize=10)
        plt.tight_layout(); plt.show()
        print(f'{axis}-groups, {pattern}: {len(crit)}; node names from {fd.nodefile}; {int(keep.sum())} of '
              f'{int(in_range.sum())} steps in range used; flags: {len(WAIT_PATTERNS[pattern]["flags"])} summed per rank')
        print('distance    groups   critical spin: median [IQR]      max spin: median [IQR]')
        for c in cats:
            m = (dist['max'] == c) & np.isfinite(crit)
            if m.any():
                print(f'{c:g}: {DISTANCE_LEVELS.get(c, ""):<17s}{m.sum():5d}  {np.median(crit[m]):9.4g} '
                      f'[{np.percentile(crit[m], 25):.4g}, {np.percentile(crit[m], 75):.4g}]'
                      f'    {np.median(mx[m]):9.4g} [{np.percentile(mx[m], 25):.4g}, {np.percentile(mx[m], 75):.4g}]')
        results[pattern] = dict(dist=dist, critical=crit, max=mx, corr=corr, steps=steps[keep])
    return results


def rank_racks(fd):
    """(rack index per rank, rack names): the rack is the cabinet 'x####' of the rank's node (fd.nodefile placement,
    see node_names).  Aurora's dragonfly groups are its racks (166 compute groups, 2 global links between any
    two), so the rack is the unit whose global links all off-rack traffic shares.  Raises if a name does not
    parse, since a rank with no rack would silently drop out of the contention."""
    names = node_names(fd)
    rack = []
    for nm in names:
        m = _NODE_RE.fullmatch(str(nm))
        if m is None:
            raise ValueError(f'node name {nm!r} does not parse as x4516c6s0b0n0; cannot assign a rack')
        rack.append('x' + m.group(1) + m.group(2))
    uniq, idx = np.unique(np.array(rack), return_inverse=True)
    return idx, list(uniq)


def rank_units(fd, level='rack'):
    """(unit index per rank, unit names) for level 'rack' (cabinet x####) or 'chassis' (x####c#), from the node of
    each rank (fd.nodefile placement, see node_names).  Raises if a name does not parse, since a rank with no unit
    would silently drop out of the contention."""
    if level not in ('rack', 'chassis'):
        raise ValueError(f"level must be 'rack' or 'chassis', not {level!r}")
    out = []
    for nm in node_names(fd):
        m = _NODE_RE.fullmatch(str(nm))
        if m is None:
            raise ValueError(f'node name {nm!r} does not parse as x4516c6s0b0n0; cannot assign a {level}')
        out.append('x' + m.group(1) + m.group(2) + (f'c{m.group(3)}' if level == 'chassis' else ''))
    uniq, idx = np.unique(np.array(out), return_inverse=True)
    return idx, list(uniq)


def rank_racks(fd):
    """(rack index per rank, rack names): rank_units(fd, 'rack').  Aurora's dragonfly groups are its racks (166
    compute groups, 2 global links between any two), so the rack is the unit whose global links all off-rack traffic shares."""
    return rank_units(fd, 'rack')


def rack_traffic(fd, level='rack', include=('transposes', 'manifest', 'neighbor'),
                 multipole_gb=12.0, manifest_gb=30.0, zneighbor_gb=20.0):
    """GB per step that cross the boundary of each rack (or chassis, level='chassis') to or from ranks in other
    units, summed over the communication patterns that run at the same time.  Ranks in the same unit, including
    the same node, cost nothing.  Load estimates from D. Eisenstein; per rank and step:
      transposes: 2 transposes of multipole_gb among the size_x ranks sharing z (2*multipole_gb/size_x to each) and
          among the size_z ranks sharing x (2*multipole_gb/size_z to each)
      manifest: manifest_gb to the x-1 neighbor (same z)
      neighbor: zneighbor_gb to each of z-1 and z+1 (same x)
    The loads are the same every step.  Returns {'units': names, 'n_ranks', 'unit_of', 'egress', 'ingress'} with
    per-unit arrays in GB."""
    sx, sz = fd.topo['size_x'], fd.topo['size_z']
    nr = sx * sz
    if nr != len(node_names(fd)):
        raise ValueError('rank grid and node names disagree')
    unit_of, names = rank_units(fd, level)
    nu = len(names)
    egress, ingress = np.zeros(nu), np.zeros(nu)
    X, Z = np.divmod(np.arange(nr), sz)
    src = np.arange(nr)

    def add_ring(dst, gb):
        m = unit_of[src] != unit_of[dst]
        egress[:] += gb * np.bincount(unit_of[src][m], minlength=nu)
        ingress[:] += gb * np.bincount(unit_of[dst][m], minlength=nu)

    def add_alltoall(group_of, ngroups, gb_per_pair):
        C = np.bincount(group_of * nu + unit_of, minlength=ngroups * nu).reshape(ngroups, nu)
        size = C.sum(axis=1, keepdims=True)
        off = gb_per_pair * (C * (size - C)).sum(axis=0)       # a rank sends to each partner outside its unit
        egress[:] += off
        ingress[:] += off                                      # all-to-all is symmetric

    if 'transposes' in include:
        add_alltoall(Z, sz, 2.0 * multipole_gb / sx)           # x transposes: fixed-z groups of size_x ranks
        add_alltoall(X, sx, 2.0 * multipole_gb / sz)           # z transposes: fixed-x groups of size_z ranks
    if 'manifest' in include:
        add_ring(((X - 1) % sx) * sz + Z, manifest_gb)
    if 'neighbor' in include:
        add_ring(X * sz + (Z - 1) % sz, zneighbor_gb)
        add_ring(X * sz + (Z + 1) % sz, zneighbor_gb)
    return dict(units=names, n_ranks=np.bincount(unit_of, minlength=nu), unit_of=unit_of, egress=egress, ingress=ingress)


def _rack_steps(fd, step_range, exclude_checkpoints, flush_steps):
    """Steps with cached per-rank ingress/egress waits that fall in step_range and, if exclude_checkpoints, outside
    the checkpoint-flush windows (as in plot_wait_pattern)."""
    steps = np.array([st for st in selected(fd) if 'wait_ingress' in fd.rank and st in fd.rank['wait_ingress']])
    if not len(steps):
        raise ValueError('no per-rank ingress/egress waits cached: run read_steps again (the derived-field version bump '
                         're-reads every step)')
    in_range = _step_mask(fd, steps, step_range)
    keep = in_range.copy()
    if not keep.any():
        raise ValueError(f'no cached steps in step_range {step_range!r} (node_steps {fd.node_steps})')
    if exclude_checkpoints:
        cp = checkpoint_steps(fd)
        for w in cp['written']:
            f = next((x for x in cp['flush_done'] if x >= w), w + flush_steps)
            keep &= ~((steps >= w) & (steps <= f))
    if not keep.any():
        raise ValueError('every step in step_range was excluded as a checkpoint window; pass exclude_checkpoints=False')
    return steps[keep]


def plot_rack_contention(fd, level='rack', directions=('ingress', 'egress'), combine='mean',
                         include=('transposes', 'manifest', 'neighbor'), multipole_gb=12.0, manifest_gb=30.0,
                         zneighbor_gb=20.0, exclude_checkpoints=True, flush_steps=10, step_range='nodefile',
                         n_perm=2000, fig_size=None):
    """MPI spin of each rack (or chassis, `level`) against the traffic that must cross the unit's boundary
    (rack_traffic, D. Eisenstein's load estimates), no reduction over communicators.  One point per unit.
      x: the unit's contention: GB per step to (egress) or from (ingress) other units, summed over
         all patterns running at once (`include`: any of 'transposes', 'manifest', 'neighbor').  Constant in time.
      y: the unit's spin: the ranks in it combined by `combine` ('sum', 'mean', 'max' or 'min') of each rank's
         summed spin flags (RACK_SPIN_FLAGS[direction]: recv flags + the all-to-all flags for 'ingress', send flags +
         the all-to-all flags for 'egress'), computed per step, then the median over the used steps.
    Columns: spin vs contention, contention vs ranks in the unit, spin vs ranks in the unit (partly filled units
    have fewer ranks, and the contention is ~ proportional to the rank count, which confounds the first).
    Titles give Spearman correlations with permutation p-values (n_perm shuffles).
    Steps: step_range ('nodefile' = fd.node_steps, an inclusive (first, last), or None), checkpoint windows dropped
    if exclude_checkpoints (flush_steps as in plot_wait_pattern).
    fig_size: figure (width, height) in inches; None (default) gives (17, 4.6 * number of directions).
    For the group-size effect on the max and the shape of the spin distribution see plot_rack_spin_distribution.
    Returns {direction: dict}."""
    import matplotlib.pyplot as plt
    if combine not in ('sum', 'mean', 'max', 'min'):
        raise ValueError(f"combine must be 'sum', 'mean', 'max' or 'min', not {combine!r}")
    directions = [directions] if isinstance(directions, str) else list(directions)
    for d in directions:
        if d not in RACK_SPIN_FLAGS:
            raise ValueError(f"direction must be 'ingress' or 'egress', not {d!r}")
    comb = {'sum': np.sum, 'mean': np.mean, 'max': np.max, 'min': np.min}[combine]
    tr = rack_traffic(fd, level, include, multipole_gb, manifest_gb, zneighbor_gb)
    use = _rack_steps(fd, step_range, exclude_checkpoints, flush_steps)
    unit_of, nu, n_ranks = tr['unit_of'], len(tr['units']), tr['n_ranks']
    members = [np.flatnonzero(unit_of == k) for k in range(nu)]
    results = {}
    fig, axs = plt.subplots(len(directions), 3, figsize=(17, 4.6 * len(directions)) if fig_size is None else fig_size,
                            squeeze=False)
    for row, d in enumerate(directions):
        spin = np.empty((len(use), nu))                                          # per step, unit-combined
        for i, st in enumerate(use):
            v = fd.rank['wait_' + d][st]
            spin[i] = [comb(v[m]) for m in members]
        x, y = tr[d], np.median(spin, axis=0)
        rho, p = _spearman(x, y, n_perm)
        rho_cn, p_cn = _spearman(n_ranks, x, n_perm)
        rho_sn, p_sn = _spearman(n_ranks, y, n_perm)
        ax = axs[row]
        sc = ax[0].scatter(x, y, c=n_ranks, s=30, alpha=0.85, cmap='viridis')
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 2 and np.ptp(x[ok]) > 0:
            slope, icpt = np.polyfit(x[ok], y[ok], 1)
            xs = np.array([x[ok].min(), x[ok].max()])
            ax[0].plot(xs, slope * xs + icpt, color='C3', lw=1.2)
        fig.colorbar(sc, ax=ax[0]).set_label(f'ranks on the {level}')
        ax[0].set_xlabel('GB x-ing boundary')
        ax[0].set_ylabel(f'{d} MPI spin, {combine} [s per step]')
        ax[0].set_title(f'{level} {d}: Spearman {rho:.2f}, p={p:.3g}', fontsize=10)
        ax[1].scatter(n_ranks + np.random.default_rng(0).uniform(-0.3, 0.3, nu), x, s=12, alpha=0.6)
        ax[1].set_xlabel(f'ranks on the {level}'); ax[1].set_ylabel(f'{d} contention [GB per step]')
        ax[1].set_title(f'contention vs ranks: Spearman {rho_cn:.2f}, p={p_cn:.3g}', fontsize=10)
        ax[2].scatter(n_ranks + np.random.default_rng(1).uniform(-0.3, 0.3, nu), y, s=12, alpha=0.6)
        ax[2].set_xlabel(f'ranks on the {level}'); ax[2].set_ylabel(f'{d} spin [s per step]')
        ax[2].set_title(f'spin vs ranks: Spearman {rho_sn:.2f}, p={p_sn:.3g}', fontsize=10)
        print(f'{level} {d}: {nu} units, ranks per unit {n_ranks.min()}..{n_ranks.max()} (median {int(np.median(n_ranks))}); '
              f'contention [GB per step] min {x.min():.3g}, median {np.median(x):.3g}, max {x.max():.3g}; '
              f'steps used {len(use)} ({use[0]}..{use[-1]})')
        print(f'    spin vs contention: Spearman {rho:.3f} (p={p:.3g}); contention vs ranks {rho_cn:.3f}; '
              f'spin vs ranks {rho_sn:.3f}')
        results[d] = dict(units=tr['units'], n_ranks=n_ranks, contention_gb=x, spin=y, rho=rho, p=p, steps=use)
    plt.tight_layout(); plt.show()
    return results


def plot_rack_spin_distribution(fd, directions=('ingress', 'egress'), quantiles=(0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
                                size_bins=(1, 33, 97, 129), exclude_ranks=(), exclude_checkpoints=True,
                                flush_steps=10, step_range='nodefile', fig_size=None):
    """How the distribution of the MPI spin of the ranks of a rack depends on how many ranks the rack holds.  Per
    rack and step, the quantiles of its ranks' summed spin flags (RACK_SPIN_FLAGS[direction]) and two spreads
    (max - median, and the interquartile range), then the median over the used steps.  There is no baseline: a
    rack with more ranks puts more traffic through the rack's links (rack_traffic), which is the effect of
    interest, but the max of more ranks is also larger from sampling alone, so compare the quantiles: a median
    that rises with size points to a shared resource, a max or spread that rises alone does not tell the two apart.
    Columns: (1) each quantile (dots) against the number of ranks on the rack, with the median over the racks in
    each size bin of `size_bins` (edges, rank counts) as lines; (2) the two spreads, same style; (3) the quantile
    function of the spin within a rack (spin against quantile, 0..1) as the median curve over the racks in each
    size bin, which coincide if the shape of the within-rack distribution does not depend on rack size.
    Prints the number of racks per size bin and the Spearman correlation of each statistic with rack size.
    exclude_ranks: world ranks to leave out (e.g. the known-slow ranks).  Steps and checkpoint windows as in
    plot_rack_contention; fig_size None gives (18, 4.6 * number of directions).  Returns {direction: dict}."""
    import matplotlib.pyplot as plt
    directions = [directions] if isinstance(directions, str) else list(directions)
    for d in directions:
        if d not in RACK_SPIN_FLAGS:
            raise ValueError(f"direction must be 'ingress' or 'egress', not {d!r}")
    quantiles = tuple(quantiles)
    use = _rack_steps(fd, step_range, exclude_checkpoints, flush_steps)
    rack_of, names = rank_racks(fd)
    nrack = len(names)
    alive = np.ones(len(rack_of), bool)
    alive[list(exclude_ranks)] = False
    members = [np.flatnonzero((rack_of == k) & alive) for k in range(nrack)]
    n_k = np.array([len(m) for m in members])
    ok = n_k > 0
    qs = np.unique(np.round(np.concatenate([quantiles, (0.25, 0.5, 0.75, 1.0)]), 10))
    qgrid = np.linspace(0.0, 1.0, 21)
    qall = np.unique(np.round(np.concatenate([qs, qgrid]), 10))
    pos = lambda arr: np.searchsorted(qall, np.round(arr, 10))
    iq, ig = pos(qs), pos(qgrid)
    j50, j25, j75, j100 = (int(np.searchsorted(qs, v)) for v in (0.5, 0.25, 0.75, 1.0))
    qsel = [int(np.searchsorted(qs, np.round(q, 10))) for q in quantiles]
    stat_names = [('max' if q == 1.0 else f'q{q:g}') for q in quantiles] + ['max-median', 'IQR']

    def stats(Q):                                   # Q[..., len(qs)] -> [..., len(stat_names)]
        return np.concatenate([Q[..., qsel], (Q[..., j100] - Q[..., j50])[..., None],
                               (Q[..., j75] - Q[..., j25])[..., None]], axis=-1)

    bins = [(size_bins[b], size_bins[b + 1]) for b in range(len(size_bins) - 1)]
    results = {}
    fig, axs = plt.subplots(len(directions), 3, figsize=(18, 4.6 * len(directions)) if fig_size is None else fig_size,
                            squeeze=False)
    for row, d in enumerate(directions):
        realQ = np.full((len(use), nrack, len(qall)), np.nan)
        for i, st in enumerate(use):
            v = np.asarray(fd.rank['wait_' + d][st], dtype=np.float64)
            for k, m in enumerate(members):
                if len(m):
                    realQ[i, k] = np.quantile(v[m], qall)
        real_stats = np.median(stats(realQ[..., iq]), axis=0)                         # (rack, stat)
        qcurve = np.median(realQ[..., ig], axis=0)                                    # (rack, len(qgrid))
        ax = axs[row]
        bin_sel = [ok & (n_k >= lo) & (n_k < hi) for lo, hi in bins]
        for panel, idxs in ((0, range(len(quantiles))), (1, (len(quantiles), len(quantiles) + 1))):
            for c_i, s_i in enumerate(idxs):
                c = f'C{c_i}'
                ax[panel].scatter(n_k[ok], real_stats[ok, s_i], s=10, color=c, alpha=0.45)
                xs_b = [np.median(n_k[b]) for b in bin_sel if b.any()]
                ys_b = [np.median(real_stats[b, s_i]) for b in bin_sel if b.any()]
                ax[panel].plot(xs_b, ys_b, color=c, lw=1.4, marker='o', ms=4, label=stat_names[s_i])
            ax[panel].set_xscale('log'); ax[panel].set_xlabel('ranks on the rack')
            ax[panel].legend(fontsize=7, ncol=2)
        ax[0].set_ylabel(f'{d} MPI spin quantile [s per step]'); ax[0].set_title('dots: racks; lines: median per size bin', fontsize=10)
        ax[1].set_ylabel(f'{d} spin spread [s per step]'); ax[1].set_title('spread', fontsize=10)
        for (lo, hi), sel in zip(bins, bin_sel):
            if sel.any():
                ax[2].plot(qgrid, np.median(qcurve[sel], axis=0), marker='.', label=f'{lo}-{hi - 1} ranks ({int(sel.sum())} racks)')
        ax[2].set_xlabel('quantile within the rack'); ax[2].set_ylabel(f'{d} spin [s per step]')
        ax[2].set_title('within-rack spin quantile function by rack size', fontsize=10); ax[2].legend(fontsize=7)
        print(f'{d}: {int(ok.sum())} racks, ranks per rack {n_k[ok].min()}..{n_k[ok].max()}; per size bin: '
              + ', '.join(f'{lo}-{hi - 1}: {int(b.sum())}' for (lo, hi), b in zip(bins, bin_sel))
              + f'; {len(use)} steps ({use[0]}..{use[-1]})')
        for s_i, nm in enumerate(stat_names):
            rho_s, p_s = _spearman(n_k[ok], real_stats[ok, s_i], 2000)
            print(f'    {nm:<11s} Spearman vs rack size {rho_s:6.2f} (p={p_s:.3g})')
        results[d] = dict(racks=names, n_ranks=n_k, stat_names=stat_names, real=real_stats, quantile_curve=qcurve,
                          qgrid=qgrid, steps=use)
    plt.tight_layout(); plt.show()
    return results


def plot_rank_spin_vs_distance(fd, patterns=None, exclude_checkpoints=True, flush_steps=10, step_range='nodefile',
                               max_points=6000):
    """Every rank's MPI spin on each communication pattern against the distance between the nodes it exchanges
    with, with no reduction over communicators.  Per rank: y = its median over the used steps of the sum of
    the pattern's spin flags (WAIT_PATTERNS); x = its maximum partner distance (boxes + points) and its mean
    partner distance (rank_pattern_distance; ranks with no network partner, i.e. all partners on their own node,
    are left out).  Titles give the Spearman correlation.  The 5 known-slow ranks of a run, whose compute makes
    their partners wait, show up as the high outliers.  Steps as in plot_wait_pattern.  A step cached before
    per-rank waits were stored is skipped, so a step range with none cached raises.  Returns {pattern: dict}."""
    import warnings
    import matplotlib.pyplot as plt
    pats = list(WAIT_PATTERNS) if patterns is None else ([patterns] if isinstance(patterns, str) else patterns)
    steps = np.array([s for s in selected(fd) if s in fd.rank.get('wait_' + pats[0], {})])
    if not len(steps):
        raise ValueError('no per-rank waits cached: run read_steps again (bumping GROUP_DERIVED_VERSION re-reads)')
    in_range = _step_mask(fd, steps, step_range)
    keep = in_range.copy()
    if not keep.any():
        raise ValueError(f'no cached steps in step_range {step_range!r} (node_steps {fd.node_steps})')
    if exclude_checkpoints:
        cp = checkpoint_steps(fd)
        for w in cp['written']:
            f = next((x for x in cp['flush_done'] if x >= w), w + flush_steps)
            keep &= ~((steps >= w) & (steps <= f))
    if not keep.any():
        raise ValueError('every step in step_range was excluded as a checkpoint window; pass exclude_checkpoints=False')
    use = steps[keep]
    rng = np.random.default_rng(0)
    results = {}
    for pattern in pats:
        rmax, rmean = rank_pattern_distance(fd, pattern)
        y = np.median(np.stack([fd.rank['wait_' + pattern][st] for st in use]), axis=0)
        ok = np.isfinite(rmax) & np.isfinite(y)
        cats = np.unique(rmax[ok])
        rho_m, p_m = _spearman(rmax[ok], y[ok])
        rho_a, p_a = _spearman(rmean[ok], y[ok])
        idx = np.flatnonzero(ok)
        if len(idx) > max_points:
            idx = rng.choice(idx, max_points, replace=False)
        fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
        ax[0].boxplot([y[ok & (rmax == c)] for c in cats], positions=cats, widths=0.5, showfliers=False,
                      medianprops=dict(color='C3'))
        ax[0].scatter(rmax[idx] + rng.uniform(-0.18, 0.18, len(idx)), y[idx], s=4, alpha=0.3, color='C0', zorder=3)
        ax[0].set_xticks(cats)
        ax[0].set_xticklabels([f'{c:g}: {DISTANCE_LEVELS.get(c, "")}\n(n={int((ok & (rmax == c)).sum())})'
                               for c in cats], fontsize=8)
        ax[0].set_xlabel("rank's maximum partner distance"); ax[0].set_ylabel(f'MPI spin, {pattern} [s per step]')
        ax[0].set_title(f'{pattern}, {int(ok.sum())} ranks: Spearman {rho_m:.2f}, p={p_m:.3g}', fontsize=10)
        ax[1].scatter(rmean[idx] + rng.uniform(-0.03, 0.03, len(idx)), y[idx], s=4, alpha=0.3)
        ax[1].set_xlabel("rank's mean partner distance"); ax[1].set_ylabel(f'MPI spin, {pattern} [s per step]')
        ax[1].set_title(f'vs mean distance: Spearman {rho_a:.2f}, p={p_a:.3g}', fontsize=10)
        plt.tight_layout(); plt.show()
        print(f'{pattern}: {len(use)} steps used ({use[0]}..{use[-1]}); distance   ranks   spin median [IQR]')
        for c in cats:
            m = ok & (rmax == c)
            print(f'  {c:g}: {DISTANCE_LEVELS.get(c, ""):<17s}{m.sum():6d}  {np.median(y[m]):9.4g} '
                  f'[{np.percentile(y[m], 25):.4g}, {np.percentile(y[m], 75):.4g}]')
        results[pattern] = dict(rank_max=rmax, rank_mean=rmean, spin=y, steps=use,
                                rho_max=rho_m, p_max=p_m, rho_mean=rho_a, p_mean=p_a)
    return results


def plot_wait_pattern(fd, axis, pattern, metric='excess',
                      exclude_checkpoints=True, per_slab=False, flush_steps=10, center_slack_by_slabs=True,
                      step_range='nodefile'):
    """A group's MPI wait on one communication pattern beyond what its compute imbalance explains, and its slack,
    against how far apart the nodes are that the pattern connects.

    Row 1, the wait.  For each group and step: W = the MAX over the group's ranks of the rank's wait, where
    a rank's wait is the SUM of the pattern's spin flags (WAIT_PATTERNS[pattern]['flags']; the flags can
    overlap, so the sum is an upper bound).
    spread = busy max - busy mean over the group's ranks, which is the average wait a rank would have if the
    only cause were the group's compute imbalance and it synchronized once per step (W is a max over
    ranks and spread a mean wait, so W - spread is biased upward by the spread of the waits).
      metric='excess' (default): W - spread, seconds per step: an estimate of the communication-and-latency
          part of the wait.  The group really syncs every slab, so the single-sync spread underestimates
          imbalance waiting and this overstates the communication part.
      metric='ratio': W / spread.
    Row 2, the slack.  The group's minimum barrier wait (mpi_barrier_time over its ranks) at the barrier after
    the pipeline: how much earlier than the globally slowest rank the group's busiest rank arrives, so ~0 means
    the group is critical.  It is the net effect on a group's pace of everything, not just the network.  Groups
    that own 37 slabs instead of 38 finish ~2.7% earlier, far more than any network effect, so with
    center_slack_by_slabs=True (default) each slab class is centred on its own median; this assumes the
    separations are not themselves aligned with the slab classes (true when the classes are interleaved).
    Each value is the group's median over steps.  exclude_checkpoints drops the steps from each checkpoint to
    the completion of its flush (or `flush_steps` later if none is logged), when the flush perturbs everything.
    per_slab divides the wait by the group's slab count.  The distance is pattern_distance(fd, pattern, axis),
    from the node names in fd.nodefile.  step_range: the steps used, 'nodefile' (default, fd.node_steps = the
    steps that ran on that nodefile's nodes), an inclusive (first, last), or None for every cached step
    (only sensible if the placement did not change).
    Columns: against the group's maximum partner distance (boxes and points), against its mean partner distance;
    the right panel plots the wait against the slack.  Titles give the Spearman rank correlation and a
    permutation p-value.  A hop costs about a microsecond, small against compute imbalance, so expect a weak
    signal, and note that a trend can come from node speed or from position in the allocation instead of the
    network.  Returns a dict with the per-group distance arrays, y (the wait), slack, their interquartile range,
    the correlations and the steps used."""
    import warnings
    import matplotlib.pyplot as plt
    if metric not in ('excess', 'ratio'):
        raise ValueError(f"metric must be 'excess' or 'ratio', not {metric!r}")
    fields = WAIT_PATTERNS[pattern]['flags']
    steps = group_steps(fd, axis)
    in_range = _step_mask(fd, steps, step_range)
    keep = in_range.copy()
    if not keep.any():
        raise ValueError(f'no cached steps in step_range {step_range!r} (node_steps {fd.node_steps}); '
                         f'cached group steps are {steps[:1]}..{steps[-1:]}')
    if exclude_checkpoints:
        cp = checkpoint_steps(fd)
        for w in cp['written']:
            f = next((x for x in cp['flush_done'] if x >= w), w + flush_steps)
            keep &= ~((steps >= w) & (steps <= f))
    if not keep.any():
        raise ValueError('every step in step_range was excluded as a checkpoint window; pass exclude_checkpoints=False')
    W = group_get(fd, f'wait_{pattern}', 'max', axis)                              # (n_steps, n_groups)
    spread = group_get(fd, 'busy', 'max', axis) - group_get(fd, 'busy', 'mean', axis)
    with np.errstate(divide='ignore', invalid='ignore'):
        val = W - spread if metric == 'excess' else np.where(spread > 1e-9, W / spread, np.nan)
    if per_slab:
        val = val / group_slabs(fd, axis)[None, :]
    bar_min = group_get(fd, 'mpi_barrier_time', 'min', axis)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        y = np.nanmedian(val[keep], axis=0)
        q25, q75 = np.nanpercentile(val[keep], [25, 75], axis=0)
        slack_raw = np.nanmedian(bar_min[keep], axis=0)
        slack = slack_raw.copy()
        if center_slack_by_slabs:
            sl, med0 = group_slabs(fd, axis), np.nanmedian(slack_raw)
            for u in np.unique(sl):
                m = sl == u
                slack[m] = slack_raw[m] - np.nanmedian(slack_raw[m]) + med0
    dist = pattern_distance(fd, pattern, axis)
    rho_max, p_max = _spearman(dist['max'], y)
    rho_mean, p_mean = _spearman(dist['mean'], y)
    rs_max, ps_max = _spearman(dist['max'], slack)
    rs_mean, ps_mean = _spearman(dist['mean'], slack)
    rho_ys, p_ys = _spearman(y, slack)

    ylab = (f'MPI wait ({pattern}) - busy spread [s per step{", per slab" if per_slab else ""}]' if metric == 'excess'
            else f'MPI wait ({pattern}) / busy spread')
    slab_lab = 'slack: group min barrier wait [s]' + (',\nslab-class centred' if center_slack_by_slabs else '')
    cats = np.unique(dist['max'][np.isfinite(dist['max'])])
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.9])
    ax = np.array([[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
                   [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]])
    axr = fig.add_subplot(gs[:, 2])

    def boxes(a, yv, label, rho, p):
        a.boxplot([yv[(dist['max'] == c) & np.isfinite(yv)] for c in cats], positions=cats, widths=0.5,
                  showfliers=False, medianprops=dict(color='C3'))
        a.scatter(dist['max'] + rng.uniform(-0.18, 0.18, len(yv)), yv, s=9, alpha=0.5, color='C0', zorder=3)
        a.set_xticks(cats)
        a.set_xticklabels([f'{c:g}: {DISTANCE_LEVELS.get(c, "")}\n(n={int((dist["max"] == c).sum())})' for c in cats],
                          fontsize=8)
        a.set_xlabel("group's maximum partner distance"); a.set_ylabel(label)
        a.set_title(f'vs max distance: Spearman {rho:.2f}, p={p:.3g}', fontsize=10)

    def scatter(a, yv, label, rho, p):
        a.scatter(dist['mean'], yv, s=12, alpha=0.6)
        ok = np.isfinite(yv) & np.isfinite(dist['mean'])
        if ok.sum() > 2 and np.ptp(dist['mean'][ok]) > 0:
            slope, icpt = np.polyfit(dist['mean'][ok], yv[ok], 1)
            xs = np.array([dist['mean'][ok].min(), dist['mean'][ok].max()])
            a.plot(xs, slope * xs + icpt, color='C3', lw=1.2)
        a.set_xlabel("group's mean partner distance"); a.set_ylabel(label)
        a.set_title(f'vs mean distance: Spearman {rho:.2f}, p={p:.3g}', fontsize=10)

    boxes(ax[0, 0], y, ylab, rho_max, p_max); scatter(ax[0, 1], y, ylab, rho_mean, p_mean)
    boxes(ax[1, 0], slack, slab_lab, rs_max, ps_max); scatter(ax[1, 1], slack, slab_lab, rs_mean, ps_mean)
    ax[0, 0].set_title(f'{axis}-groups, wait: ' + ax[0, 0].get_title(), fontsize=10)
    ax[1, 0].set_title(f'{axis}-groups, slack: ' + ax[1, 0].get_title(), fontsize=10)
    sc = axr.scatter(slack, y, c=dist['max'], cmap='viridis', s=18, alpha=0.8, vmin=cats.min(), vmax=cats.max())
    cb = fig.colorbar(sc, ax=axr, ticks=cats); cb.set_label("group's maximum partner distance")
    axr.set_xlabel(slab_lab); axr.set_ylabel(ylab)
    axr.set_title(f'wait vs slack: Spearman {rho_ys:.2f}, p={p_ys:.3g}', fontsize=10)
    plt.tight_layout(); plt.show()

    print(f'{axis}-groups, {pattern}: {len(y)}; node names from {fd.nodefile} (steps {fd.node_steps[0]}..{fd.node_steps[1]}); '
          f'{int(in_range.sum())} cached steps in step_range ({steps[in_range][0]}..{steps[in_range][-1]}), '
          f'{int(keep.sum())} used'
          + (f' ({int((in_range & ~keep).sum())} excluded as checkpoint windows)' if (in_range & ~keep).any() else '')
          + f'; pattern {pattern}: max over ranks of the per-rank sum of {len(fields)} flags; metric: {metric}'
          + ('; slack centred within slab classes' if center_slack_by_slabs else ''))
    print('distance    groups   wait: median [IQR of group medians]      slack: median [IQR]')
    for c in cats:
        m = (dist['max'] == c) & np.isfinite(y)
        if m.any():
            print(f'{c:g}: {DISTANCE_LEVELS.get(c, ""):<17s}{m.sum():5d}  {np.median(y[m]):9.4g} '
                  f'[{np.percentile(y[m], 25):.4g}, {np.percentile(y[m], 75):.4g}]'
                  f'    {np.nanmedian(slack[m]):9.4g} [{np.nanpercentile(slack[m], 25):.4g}, {np.nanpercentile(slack[m], 75):.4g}]')
    return dict(distance=dist, y=y, slack=slack, slack_raw=slack_raw, iqr=(q25, q75),
                rho_max=rho_max, p_max=p_max, rho_mean=rho_mean, p_mean=p_mean,
                rho_slack_max=rs_max, p_slack_max=ps_max, rho_slack_mean=rs_mean, p_slack_mean=ps_mean,
                rho_wait_slack=rho_ys, p_wait_slack=p_ys, steps=steps[keep])


# ----------------------------------------------------------------------------------------------
# which ranks (and so nodes) make a field's group maxima excessive in both the x and z reductions
# ----------------------------------------------------------------------------------------------
def _bad_groups(score, n_bad, nsigma):
    """Indices of the groups whose score is high: the n_bad highest, or those above median + nsigma robust
    sigmas (1.4826 MAD) of the scores when n_bad is None."""
    if n_bad is not None:
        return np.argsort(score)[::-1][:n_bad]
    med = np.nanmedian(score)
    sig = 1.4826 * np.nanmedian(np.abs(score - med))
    return np.where(score > med + nsigma * max(sig, 1e-12))[0]


SLOW_RANK_FIELDS = ('slab_pipeline.kick_time', 'slab_pipeline.drift_time')   # the kick and drift pipeline timings


def find_slow_ranks(fd, fields=SLOW_RANK_FIELDS, n_bad=None, nsigma=5.0, min_match=0.3,
                    rel_tol=1e-6, plot=True, step_range='nodefile'):
    """Which ranks make the group maxima of `fields` excessive in both the x and the z reduction.

    fields: one dotted field name or a list of them (default: the kick and drift pipeline timings).  Their
    times are added into one metric.  The group reductions are kept per field, so the group "max" of the
    metric is the sum of each field's group max: an upper bound on the largest per-rank sum, and equal to
    it when the same rank is the slowest in every field.

    An x group (fixed x, comm_1d_z) and a z group (fixed z, comm_1d_x) share exactly ONE rank, rank = x*size_z
    + z.  So a single slow rank shows up as one bad x group and one bad z group, and the groups' maxima are
    then the SAME number at every step in which that rank is the slowest in both.  That gives an exact test
    that needs no per-rank data: for every x group i, z group j and field, the two group maxima are equal
    (within rel_tol) when the shared rank is the slowest in both.  match[i, j] is the fraction of the metric's
    time (summed over fields, averaged over steps) carried by fields whose two maxima are equal, so a pair
    with a high match is one rank, rank i*size_z + j, that is the slowest member of both groups in the
    fields that make up the time.

    Offending groups: score of a group = its median over steps of (group max / median of the group maxima at
    that step) of the summed metric; a group is offending if it is among the n_bad highest scores, or (n_bad=None) above the median
    score by nsigma robust sigmas.  Needs group_axes=('x','z') in the run.
    step_range: the steps used, 'nodefile' (default: fd.node_steps, the steps that ran on the nodes in
    fd.nodefile, so the printed node names are right for them), an inclusive (first, last), or None for all.
    Prints the offending groups, the matched ranks (match >= min_match) with their node names and the other rank
    on the same node, and any offending group without a partner (its slowest rank changes from step to step,
    or the bad ranks are not the slowest in the other direction).  Returns a dict with the scores, the
    offending groups, match (n_x_groups, n_z_groups), the ranks table and 'nodes', the sorted node names of the matched ranks."""
    import matplotlib.pyplot as plt
    for ax_ in ('x', 'z'):
        if ax_ not in fd.group_axes:
            raise ValueError(f"needs group_axes to include 'x' and 'z'; got {tuple(fd.group_axes)}")
    steps = group_steps(fd, 'x')
    if len(steps) == 0 or not np.array_equal(steps, group_steps(fd, 'z')):
        raise ValueError("no steps with both x and z group reductions; run read_steps(fd) first")
    sx, sz = fd.topo['size_x'], fd.topo['size_z']
    fields = [fields] if isinstance(fields, str) else list(fields)
    if not fields:
        raise ValueError('fields is empty')
    FX = [group_get(fd, f, 'max', 'x').astype(np.float64) for f in fields]      # each (n_steps, size_x)
    FZ = [group_get(fd, f, 'max', 'z').astype(np.float64) for f in fields]      # each (n_steps, size_z)
    Mx, Mz = np.sum(FX, axis=0), np.sum(FZ, axis=0)
    label = ' + '.join(f.split('.')[-1].replace('_time', '') for f in fields)
    sel = _step_mask(fd, steps, step_range)
    if not sel.any():
        raise ValueError(f'no cached steps in step_range {step_range!r} (node_steps {fd.node_steps})')
    steps, Mx, Mz = steps[sel], Mx[sel], Mz[sel]
    FX, FZ = [f[sel] for f in FX], [f[sel] for f in FZ]
    score_x = np.nanmedian(Mx / np.nanmedian(Mx, axis=1, keepdims=True), axis=0)
    score_z = np.nanmedian(Mz / np.nanmedian(Mz, axis=1, keepdims=True), axis=0)
    bad_x = _bad_groups(score_x, n_bad, nsigma)
    bad_z = _bad_groups(score_z, n_bad, nsigma)

    # fraction of steps with equal maxima, for every (x group, z group) pair
    match = np.zeros((sx, sz))
    for t in range(len(steps)):
        w = np.zeros((sx, sz))                 # time carried by the fields in which the two maxima are equal
        for fx, fz in zip(FX, FZ):
            a, b = fx[t][:, None], fz[t][None, :]
            w += np.where(np.abs(a - b) <= rel_tol * np.maximum(np.abs(a), np.abs(b)), a, 0.0)
        match += w / np.maximum(Mx[t][:, None], 1e-300)
    match /= len(steps)

    node = node_names(fd)
    rows = []
    for i in range(sx):
        for j in np.where(match[i] >= min_match)[0]:
            if i in bad_x or j in bad_z:                  # only pairs that involve an offending group
                r = i * sz + j
                mates = [int(q) for q in np.where(node == node[r])[0] if q != r]
                rows.append(dict(rank=int(r), x=int(i), z=int(j), node=str(node[r]), same_node_ranks=mates,
                                 match=float(match[i, j]), score_x=float(score_x[i]), score_z=float(score_z[j]),
                                 x_flagged=bool(i in bad_x), z_flagged=bool(j in bad_z)))
    rows.sort(key=lambda d: -d['match'])
    matched_x = {d['x'] for d in rows}
    matched_z = {d['z'] for d in rows}
    orphan_x = [int(i) for i in bad_x if i not in matched_x]
    orphan_z = [int(j) for j in bad_z if j not in matched_z]
    own = None
    if all(f in fd.rank for f in fields) and rows:
        own = {d['rank']: float(np.nanmedian([sum(fd.rank[f][s][d['rank']] for f in fields) for s in steps
                                               if all(s in fd.rank[f] for f in fields)])) for d in rows}

    print(f'{label}: group max (sum over {len(fields)} field(s)), {len(steps)} steps {steps[0]}..{steps[-1]}')
    print(f'offending x groups ({len(bad_x)}): ' + ', '.join(f'{i} ({score_x[i]:.2f}x)' for i in bad_x[np.argsort(-score_x[bad_x])]))
    print(f'offending z groups ({len(bad_z)}): ' + ', '.join(f'{j} ({score_z[j]:.2f}x)' for j in bad_z[np.argsort(-score_z[bad_z])]))
    print(f'\nranks that are the slowest member of both their x and z group in >= {min_match:.0%} of steps '
          f'(and in an offending group):')
    print(f'{"rank":>6s} {"x":>4s} {"z":>4s}  {"match":>6s}  {"node":<16s} {"other rank on node":<20s} flagged')
    for d in rows:
        print(f'{d["rank"]:6d} {d["x"]:4d} {d["z"]:4d}  {d["match"]:6.0%}  {d["node"]:<16s} '
              f'{str(d["same_node_ranks"]):<20s} x:{"Y" if d["x_flagged"] else "n"} z:{"Y" if d["z_flagged"] else "n"}'
              + (f'  own median {own[d["rank"]]:.4g}' if own else ''))
    if not rows:
        print('  (none)')
    if orphan_x or orphan_z:
        print(f'\noffending groups with no matching rank: x {orphan_x}, z {orphan_z}  '
              f'(several different slow ranks, or a slow node whose two ranks alternate)')
    nodes = sorted({d['node'] for d in rows})
    if nodes:
        print(f'\nnode names ({len(nodes)}): ' + ' '.join(nodes))
    if len(nodes) > 1:
        P = _parse_nodes(nodes)
        print(f'\n{len(nodes)} nodes: {len({tuple(r[:3]) for r in P})} chassis, {len({tuple(r[:2]) for r in P})} cabinets, '
              f'{len({r[0] for r in P})} cabinet rows  (shared chassis/cabinet would point at the infrastructure)')

    if plot:
        fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw=dict(width_ratios=[1.2, 0.8, 1]))
        for a, sc, bad, nm in ((ax[0], score_x, bad_x, 'x'), (ax[1], score_z, bad_z, 'z')):
            a.plot(sc, '.', ms=3, color='0.5')
            a.plot(bad, sc[bad], 'o', color='C3', ms=5)
            for g in bad: a.annotate(str(g), (g, sc[g]), fontsize=7, xytext=(3, 3), textcoords='offset points')
            a.axhline(1, color='k', lw=0.5)
            a.set_xlabel(f'{nm} group'); a.set_ylabel('median group max / median over groups')
            a.set_title(f'{label}: {nm} groups', fontsize=10)
        bx, bz = np.array(sorted(bad_x)), np.array(sorted(bad_z))
        if len(bx) and len(bz):
            m = match[np.ix_(bx, bz)]
            im = ax[2].imshow(m, vmin=0, vmax=1, cmap='viridis', aspect='auto')
            ax[2].set_xticks(range(len(bz))); ax[2].set_xticklabels(bz, fontsize=7, rotation=90)
            ax[2].set_yticks(range(len(bx))); ax[2].set_yticklabels(bx, fontsize=7)
            for (u, v), val in np.ndenumerate(m):
                if val >= 0.05: ax[2].text(v, u, f'{val:.0%}', ha='center', va='center', fontsize=6, color='w' if val < 0.6 else 'k')
            fig.colorbar(im, ax=ax[2], label='fraction of steps with equal maxima')
        ax[2].set_xlabel('offending z group'); ax[2].set_ylabel('offending x group')
        ax[2].set_title('x group / z group share their slowest rank?', fontsize=10)
        plt.tight_layout(); plt.show()
    return dict(score_x=score_x, score_z=score_z, bad_x=bad_x, bad_z=bad_z, match=match, ranks=rows,
                orphan_x=orphan_x, orphan_z=orphan_z, nodes=nodes, steps=steps)


def group_heatmap(fd, name, stat='max', axis='x', relative=True, log=False, ax=None, cmap=None,
                  pct=(1, 99.5), z_axis=True, checkpoints='written', title_size=12, norm='median',
                  per_slab=False, ylabels=True, cbar_label=True):
    """Heatmap (y: group index, x: step) of field `name` reduced with `stat` over each group's ranks.
    relative=True divides each step by a reference over the groups (norm='median', the default, or 'mean'),
    so colour shows which groups are slow or fast relative to a typical group at that step; the median is
    not moved by a few outlier groups, the mean is.  per_slab=True first divides each group's value by the
    number of slabs it owns (see group_slabs), which removes the 37-vs-38-slab stripes from extensive
    quantities (times, particles, bytes); leave it off for waits, barrier time and bandwidths.
    log=True colours on a log scale.
    checkpoints: 'written' draws light gray dotted lines at the steps a checkpoint was written, 'both'
    also marks the steps a flush to disk completed, None draws none (see checkpoint_steps).
    title_size: font size (points) of the panel title.
    The title is the field name.  cbar_label (default True) puts a short tag on the colorbar saying how the
    values were reduced and scaled, e.g. '[max, per slab, rel]' or '[mean, rel to mean]' (the statistic
    over each group's ranks, whether divided by the group's slabs, and the per-step reference).
    ylabels (default True) draws the '<axis> group' label on the y axis.
    Returns (steps, values) with values of shape (n_steps, n_groups)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    _check_norm(norm)
    steps = group_steps(fd, axis)
    v = group_get(fd, name, stat, axis).astype(np.float64)
    if per_slab:
        v = v / group_slabs(fd, axis)[None, :]
    if relative:
        v = v / _reference(v, norm)
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure
    # The title is just the field; the tag below (statistic, per slab, relative-to) goes on the colorbar label.
    tag = (f'[{stat}{", per slab" if per_slab else ""}'
           f'{(", rel to " + norm) if relative and norm != "median" else (", rel" if relative else "")}]')
    title = name
    if not np.isfinite(v).any():                          # nothing to colour: say so instead of failing
        ax.text(0.5, 0.5, f'no data\n(all zero or undefined)\n{tag}', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title(title, pad=24 if z_axis else 6, fontsize=title_size)
        ax.set_xlabel('step')
        if ylabels:
            ax.set_ylabel(f'{axis} group')
        if own:
            plt.tight_layout(); plt.show()
        return steps, v
    lo, hi = np.nanpercentile(v, pct)
    if not hi > lo:                                       # constant field: widen so the colour scale exists
        hi = lo + (abs(lo) * 1e-6 if lo else 1e-12)
    kw = dict(shading='auto', cmap=cmap or ('magma' if relative else 'viridis'))
    if log:
        pos = v[v > 0]
        kw['norm'] = LogNorm(vmin=max(lo, pos.min()) if len(pos) else 1e-12, vmax=max(hi, 1e-12))
    else:
        kw.update(vmin=lo, vmax=hi)
    m = ax.pcolormesh(steps, np.arange(v.shape[1]), np.where(v > 0, v, np.nan).T if log else v.T, **kw)
    fig.colorbar(m, ax=ax, label=tag if cbar_label else None)
    ax.set_title(title, pad=24 if z_axis else 6, fontsize=title_size)
    ax.set_xlabel('step')
    if ylabels:
        ax.set_ylabel(f'{axis} group')
    if z_axis:
        _add_top_z(ax, steps, np.array([fd.state[s].get('Redshift', np.nan) for s in steps]))
    _mark_checkpoints(ax, fd, steps, checkpoints)
    if own:
        plt.tight_layout(); plt.show()
    return steps, v


def group_heatmaps(fd, specs, axis='x', ncols=2, relative=True, log=False, panel=(7.5, 4.6),
                   checkpoints='written', title_size=12, norm='median', per_slab=False, ylabels=True,
                   cbar_label=True):
    """Several group-vs-step heatmaps in one figure.  specs is a list whose items are a field name
    (stat 'max'), a (name, stat) pair, a (name, stat, relative) triple, or a (name, stat, relative, per_slab)
    4-tuple; `relative` and `per_slab` given as arguments are the defaults for items that don't set them.
    norm ('median' or 'mean') is the per-step reference for relative panels; per_slab divides by the slabs a
    group owns (see group_heatmap).  title_size is the panel title font size in points.  Returns
    {title: values}.  ylabels and cbar_label control the y-axis label and the colorbar label (see
    group_heatmap)."""
    import matplotlib.pyplot as plt
    items = []
    for s in specs:
        s = (s,) if isinstance(s, str) else tuple(s)
        items.append((s[0], s[1] if len(s) > 1 else 'max', s[2] if len(s) > 2 else relative,
                      s[3] if len(s) > 3 else per_slab))
    nrows = -(-len(items) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel[0] * ncols, panel[1] * nrows), squeeze=False)
    out = {}
    for a, (name, stat, rel, ps) in zip(axes.ravel(), items):
        out[f'{name}[{stat}]'] = group_heatmap(fd, name, stat, axis, relative=rel, log=log, ax=a,
                                                checkpoints=checkpoints, title_size=title_size,
                                                norm=norm, per_slab=ps, ylabels=ylabels,
                                                cbar_label=cbar_label)[1]
    for a in axes.ravel()[len(items):]:
        a.axis('off')
    plt.tight_layout(); plt.show()
    return out

