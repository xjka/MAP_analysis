#!/usr/bin/env python
"""Cut a non-periodic cubic box of final-timeslice MAPs out of a huge maplog set.

For simulations like AbacusAurora_demiflag_c200_ph410 (CPD 4375 split over 35 z-ranks x
62 x-ranks, 2170 maplog files, each a 142.9 x 5000 x ~81 Mpc/h column: all of y, one z
slab, one x segment), read only the files that touch a box and keep the MAPs inside it.

The box is a cube of side ``S = box_nzslices * L / NumZRanks`` (exactly that many
z-slices wide). z is the coarsest split, so it sets the resolution of the cuts:

* z: by default the box origin is snapped to a z-slice boundary (a random one), so
  exactly ``box_nzslices`` columns are read and no z cut is needed. A user-given
  origin with z off a boundary reads one more column and cuts in z.
* x: the x ranks (70-71 slabs each) are picked by overlap with the box; the first/last
  one straddles the box edge and is cut in x by position.
* y: not decomposed, so every file holds all of y; MAPs are cut by position.

The origin is the box's lower corner in simulation coordinates ([-L/2, L/2)^3); it is
random (``--seed``, default fresh entropy) unless ``--origin X Y Z`` is given, and is
always printed and saved, so a run can be repeated with ``--origin``. The box may wrap
around the periodic simulation edge; the returned positions are in *box* coordinates
[0, S)^3 (unwrapped), so a correlation function on them must be non-periodic.

x segmentation: slabs rotate through the x ranks during the run (by a step-dependent
amount), so rank j's final range is slabs ``[(S0 + CPD*j//NX) % CPD, ...)``. The maplog
header is the *start*-of-final-step state, so its FirstSlabRank0 is stale: S0 is instead
read from the final step's state file in the run's ``log.zip`` (``log/step<N>/step<N>.state``,
N = the header's FullStepNumber); for ph410 that is 1473 vs the header's 1463, matching the
MAP positions in the files. ``--x_slab_offset`` overrides this (added to the header value).
Every file read is also checked against its predicted x and z range, and the run aborts if
any MAP lies outside.

    mpiexec -n <nnodes> --ppn 1 python ~/MAP_analysis/utilities/box_maps.py SIMDIR --box_nzslices 3 --mult_min 1000
    python ~/MAP_analysis/utilities/box_maps.py SIMDIR --box_nzslices 3 --dry_run       # plan only, reads no data
"""
import argparse
import re
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import numpy as np
import asdf
from abacusnbody.data.read_abacus import read_asdf

_EPS = 1e-6          # Mpc/h, tolerance for "touching" intervals
_POS_TOL = 0.05      # Mpc/h, tolerance for the per-file domain check (positions are quantized to 1/128)


@dataclass
class Decomposition:
    """The (z-rank, x-rank) tiling of the simulation by maplog file."""
    L: float
    cpd: int
    nz: int
    nx: int
    s0: int                    # first x-slab of x-rank 0 at the final step
    files: dict                # (iz, ix) -> path

    @property
    def w(self):
        return self.L / self.cpd

    def z_range(self, iz):
        """(start, width) in Mpc/h of z-rank ``iz`` (start in [-L/2, L/2))."""
        a, b = self.cpd * iz // self.nz, self.cpd * (iz + 1) // self.nz
        return -self.L / 2 + a * self.w, (b - a) * self.w

    def x_range(self, ix):
        """(start, width) in Mpc/h of x-rank ``ix``; start in [-L/2, L/2), the range may wrap past L/2."""
        a, b = self.cpd * ix // self.nx, self.cpd * (ix + 1) // self.nx
        return -self.L / 2 + ((self.s0 + a) % self.cpd) * self.w, (b - a) * self.w


def _final_first_slab_rank0(simdir, step):
    """FirstSlabRank0 at the END of ``step`` (the final step), from the run's ``log.zip``; None if unavailable."""
    zp = Path(simdir) / 'log.zip'
    if not zp.is_file():
        return None
    with zipfile.ZipFile(zp) as z:
        try:
            txt = z.read(f'log/step{step:04d}/step{step:04d}.state').decode()
        except KeyError:
            return None
    m = re.search(r'FirstSlabRank0\s*=\s*(\d+)', txt)
    return int(m.group(1)) if m else None


def read_decomposition(simdir, x_slab_offset=None):
    """``x_slab_offset=None`` takes the final x segmentation from ``log.zip`` (see the module doc);
    an int is instead added to the maplog header's FirstSlabRank0."""
    files = sorted(glob(str(Path(simdir) / 'maplogs/*/maplog_z*_*.asdf')))
    if not files:
        raise FileNotFoundError(f'no maplog files under {simdir}')
    table = {}
    for f in files:
        m = re.search(r'maplog_z[\d.]+_(\d+)_(\d+)\.asdf$', f)
        table[(int(m.group(1)), int(m.group(2)))] = f
    asdf.get_config().validate_on_read = False
    with asdf.open(files[0], lazy_load=True, validate_checksums=False) as af:
        h = af.tree['header']
        if x_slab_offset is None:
            s0 = _final_first_slab_rank0(simdir, int(h['FullStepNumber']))
            if s0 is None:
                raise RuntimeError(f'no final-step state in {simdir}/log.zip; pass --x_slab_offset '
                                   '(slabs added to the header FirstSlabRank0)')
        else:
            s0 = int(h['FirstSlabRank0']) + x_slab_offset
        d = Decomposition(L=float(h['BoxSize']), cpd=int(h['CPD']), nz=int(h['NodeSizeZ']),
                          nx=int(h['NodeSizeX']), s0=s0, files=table)
    expect = {(iz, ix) for iz in range(d.nz) for ix in range(d.nx)}
    if set(table) != expect:
        raise RuntimeError(f'maplog files do not tile {d.nz} x {d.nx}: {len(table)} found')
    return d


@dataclass
class Box:
    origin: np.ndarray         # lower corner, simulation coords, in [-L/2, L/2)
    size: float                # cube side [Mpc/h]
    L: float
    nzslices: int
    keys: list                 # sorted (iz, ix) of the files that touch the box


def _overlaps(start, width, o, S, L):
    """Does the periodic interval [start, start+width) intersect [o, o+S)?"""
    rel = (start - o + _EPS) % L - _EPS
    return (rel < S - _EPS and rel + width > _EPS) or (rel + width - L > _EPS)


def choose_box(d, nzslices, origin=None, seed=None):
    if not 1 <= nzslices <= d.nz:
        raise ValueError(f'box_nzslices must be in [1, {d.nz}], got {nzslices}')
    S = nzslices * d.L / d.nz
    if origin is None:
        rng = np.random.default_rng(seed)
        xy = rng.uniform(-d.L / 2, d.L / 2, 2)
        z = d.z_range(int(rng.integers(d.nz)))[0]      # snapped to a z-slice boundary
        origin = np.array([xy[0], xy[1], z])
    else:
        origin = np.asarray(origin, dtype=float)
        origin = (origin + d.L / 2) % d.L - d.L / 2
    zs = [iz for iz in range(d.nz) if _overlaps(*d.z_range(iz), origin[2], S, d.L)]
    xs = [ix for ix in range(d.nx) if _overlaps(*d.x_range(ix), origin[0], S, d.L)]
    return Box(origin=origin, size=S, L=d.L, nzslices=nzslices, keys=sorted((iz, ix) for iz in zs for ix in xs))


def describe_box(d, box):
    zs = sorted({k[0] for k in box.keys})
    xs = sorted({k[1] for k in box.keys})
    o, S = box.origin, box.size
    lines = [f'box side {S:.3f} Mpc/h ({box.nzslices} z-slices); origin (x,y,z) = '
             f'({o[0]:.4f}, {o[1]:.4f}, {o[2]:.4f}); wraps in: '
             + (','.join(a for a, oa in zip('xyz', o) if oa + S > box.L / 2) or 'none'),
             f'{len(box.keys)} files = {len(zs)} z-columns {zs} x {len(xs)} x-ranks {xs}']
    z_on_edge = any(abs(((o[2] - d.z_range(iz)[0] + d.L / 2) % d.L) - d.L / 2) < 1e-3 for iz in range(d.nz))
    lines.append(f'z cut needed: {"no (origin on a z-slice boundary)" if z_on_edge and len(zs) == box.nzslices else "yes"}; '
                 f'x cut at both box edges (ranks {xs[0]} and {xs[-1]} straddle them); y cut by position '
                 f'(each file holds all of y: {100 * S / d.L:.0f}% of it is kept)')
    return '\n'.join(lines)


def final_timeslice_step(f):
    """Timestep of a maplog file's final TimeSlice nodes: the run's last step, ``FullStepNumber`` in the header."""
    with asdf.open(f, lazy_load=True, validate_checksums=False) as af:
        return int(af.tree['header']['FullStepNumber'])


def _read_box_file(key, f, d, box, mult_min, top_density=None, top_safety=1.25):
    """Final-timeslice MAPs with mult >= mult_min of one file that lie in the box, in box coords.

    ``top_density`` [(h/Mpc)^3], if given, additionally keeps only this file's
    highest-multiplicity MAPs: ``top_safety * top_density * V_file`` of them, with ``V_file``
    the volume of the file's whole domain (before the box cut). That's a local pre-selection
    for an abundance-matched sample, loose enough (``top_safety`` > 1) that the box-wide
    threshold ends up above every file's local cut; the caller checks that with the returned
    ``local_cut``."""
    asdf.get_config().validate_on_read = False
    final = final_timeslice_step(f)
    tbl = read_asdf(f, load=['pos', 'mult'], select={'node_type': 'timeslice', 'timestep': final})
    if len(tbl) == 0:
        raise RuntimeError(f'{f}: no TimeSlice nodes at the final step {final}')
    mult_all = np.asarray(tbl['mult'])
    m = mult_all >= mult_min
    local_cut = mult_min
    if top_density is not None:
        v_file = d.x_range(key[1])[1] * d.L * d.z_range(key[0])[1]
        k = int(np.ceil(top_safety * top_density * v_file))
        mm = mult_all[m]
        if k < len(mm):
            local_cut = max(local_cut, int(np.partition(mm, len(mm) - k)[len(mm) - k]))
            m &= mult_all >= local_cut
    pos = np.asarray(tbl['pos'])[m].astype(np.float64)
    mult = np.asarray(tbl['mult'])[m].astype(np.int32)

    # the file must really cover the domain we predicted for it, or the file selection is wrong
    L = d.L
    for ax, (start, width) in ((0, d.x_range(key[1])), (2, d.z_range(key[0]))):
        dd = (pos[:, ax] - start) % L
        bad = ~((dd <= width + _POS_TOL) | (dd >= L - _POS_TOL))
        if bad.any():
            raise RuntimeError(f'{f}: {bad.sum()} of {len(pos)} MAPs lie outside the predicted '
                               f'{"xyz"[ax]} range [{start:.3f}, {start + width:.3f}) (mod {L}); '
                               f'check --x_slab_offset')

    rel = (pos - box.origin) % L                       # box coordinates, unwrapped
    keep = np.all(rel < box.size, axis=1)
    return {'pos': rel[keep].astype(np.float32), 'mult': mult[keep], 'n_final': int(m.sum()), 'local_cut': local_cut,
            'redshift': float(np.asarray(tbl.meta['Step_to_Z'])[final - 1])}


def read_box_maps(d, box, mult_min=1, nworkers=16, comm=None, verbose=True, top_density=None, top_safety=1.25):
    """Read the box's MAPs. Files are dealt round-robin over MPI ranks (``comm``, one rank
    per node), then over ``nworkers`` processes within each rank (one file per task; each
    worker holds one decompressed file, so mind node memory). Returns a dict on rank 0
    (``pos`` (N,3) float32 in box coords, ``mult``, ``redshift``, ``origin``, ``size``,
    ``nfiles``, ``local_cut`` = the largest per-file multiplicity cut applied), None on other
    ranks. ``top_density``/``top_safety``: see ``_read_box_file``; with them set, any MAP with
    ``mult >= local_cut`` in the box is guaranteed present."""
    rank, size = (comm.Get_rank(), comm.Get_size()) if comm is not None else (0, 1)
    mine = box.keys[rank::size]
    t0 = time.time()
    out = []
    if mine:
        with ProcessPoolExecutor(max_workers=max(1, min(nworkers, len(mine)))) as ex:
            futs = [(k, ex.submit(_read_box_file, k, d.files[k], d, box, mult_min, top_density, top_safety)) for k in mine]
            for k, fut in futs:
                r = fut.result()
                out.append(r)
                if verbose:
                    print(f'[rank {rank}] file z{k[0]:03d}_x{k[1]:03d}: {len(r["mult"]):,} of '
                          f'{r["n_final"]:,} MAPs in box  ({time.time() - t0:.0f} s)', flush=True)
    gathered = comm.gather(out, root=0) if comm is not None else [out]
    if rank != 0:
        return None
    allr = [r for g in gathered for r in g]
    return {'pos': np.concatenate([r['pos'] for r in allr]), 'mult': np.concatenate([r['mult'] for r in allr]),
            'redshift': allr[0]['redshift'], 'origin': box.origin, 'size': box.size,
            'nfiles': len(box.keys), 'mult_min': mult_min,
            'local_cut': max(r['local_cut'] for r in allr)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('simdir')
    ap.add_argument('--box_nzslices', type=int, required=True, help='box side in z-slices (142.86 Mpc/h each for ph410)')
    ap.add_argument('--origin', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                    help='box lower corner in simulation coords [-L/2, L/2); default: random')
    ap.add_argument('--seed', type=int, default=None, help='seed for the random origin (default: fresh entropy)')
    ap.add_argument('--mult_min', type=int, default=1)
    ap.add_argument('--nworkers', type=int, default=16)
    ap.add_argument('--x_slab_offset', type=int, default=None,
                    help='override: slabs added to the header FirstSlabRank0 (default: use the final-step state in log.zip)')
    ap.add_argument('--dry_run', action='store_true', help='print the plan and exit without reading any data')
    ap.add_argument('--out', default=None, help='optional .npz to save the box MAPs to (default: nothing is saved; use /tmp or /dev/shm)')
    args = ap.parse_args()

    comm = None
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    except ImportError:
        pass
    rank = comm.Get_rank() if comm is not None else 0

    d = read_decomposition(args.simdir, args.x_slab_offset)
    if comm is not None:                                     # one origin for all ranks
        seed = comm.bcast(np.random.SeedSequence().entropy if (args.seed is None and rank == 0) else args.seed, root=0)
    else:
        seed = args.seed
    box = choose_box(d, args.box_nzslices, origin=args.origin, seed=seed)
    if rank == 0:
        print(describe_box(d, box), flush=True)
        if args.origin is None:
            print(f'random origin from seed {seed}  (repeat with --origin {" ".join(f"{v:.6f}" for v in box.origin)})', flush=True)
    if args.dry_run:
        return

    res = read_box_maps(d, box, args.mult_min, args.nworkers, comm)
    if rank != 0:
        return
    print(f'{len(res["mult"]):,} MAPs (mult >= {args.mult_min}) in box from {res["nfiles"]} files', flush=True)
    if args.out:        # nothing is written unless asked: the box lives in memory for the caller
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **res)
        print(f'saved -> {out}', flush=True)


if __name__ == '__main__':
    main()
