"""Periodic friends-of-friends labels for the HBT snapshots written by abacus_to_hbt.py.

HBT-HERONS needs, per snapshot, which particles belong to which FOF group; it does not find FOF
groups itself. The Abacus timeslice files carry no FOF labels, so this script makes them. It
reads only PartType1/Coordinates of the *converted* snapshots (so the labels are aligned with the
particle order by construction; the Abacus sim is never touched) and writes either

  * labels_<epoch>.npy (int32, -1 = in no group)   with --labels-dir, and/or
  * the GroupNumber column into the snapshot files    with --inplace.

Algorithm (one epoch per process, whole box in memory, single thread, numba):
  1. linking length  l = b * L / NP^(1/3)   (b = 0.2, the HBT/standard convention)
  2. cell grid of side >= l (~4 l, at most 1290^3 cells); particles counting-sorted into cells
  3. union-find over each cell and its 13 forward neighbour cells, periodic minimum image
  4. groups smaller than --min-part (20) get -1; the rest are numbered 0, 1, ... by decreasing
     size (ties: lowest original particle index), so the labelling does not depend on file order
     beyond the particle order itself.

Peak memory is ~50-60 bytes per particle (~28-30 GB for 512M); epochs run in parallel processes
with the same retry-on-OOM logic as the converter.

    python abacus_fof.py OUT --epochs all --inplace --skip-existing [--workers 16]
    python abacus_fof.py OUT --epochs z0.000 --labels-dir DIR            # labels file only
"""
import argparse
import multiprocessing as mp
import os
import resource
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from abacus_to_hbt import (DM, PROTECTED_ROOTS, _snapshot_files, add_labels, available_ram,  # noqa: E402
                           has_labels, log)

BYTES_PER_PARTICLE = 60          # conservative peak estimate; refine from the reported peak GB
DEFAULT_MAX_WORKERS = 16
CELL_FACTOR = 4.0                # cell side in linking lengths
MAX_NC = 1290                    # nc^3 must stay below 2^31 (int32 cell ids / offsets)


# ----------------------------------------------------------------------------------------
# kernel
# ----------------------------------------------------------------------------------------
@njit(cache=True)
def _find(p, x):
    while p[x] != x:
        p[x] = p[p[x]]
        x = p[x]
    return x


@njit(cache=True)
def _cell_ids(pos, inv_c, nc):
    n = pos.shape[0]
    cid = np.empty(n, np.int32)
    for i in range(n):
        ix = min(max(int(pos[i, 0] * inv_c), 0), nc - 1)
        iy = min(max(int(pos[i, 1] * inv_c), 0), nc - 1)
        iz = min(max(int(pos[i, 2] * inv_c), 0), nc - 1)
        cid[i] = (ix * nc + iy) * nc + iz
    return cid


@njit(cache=True)
def _counting_sort(cid, ncell):
    """order[k] = original index of the k-th particle in cell order (stable); begin[c] .. begin[c+1]
    is cell c's slice of the sorted arrays."""
    n = cid.shape[0]
    begin = np.zeros(ncell + 1, np.int32)
    for i in range(n):
        begin[cid[i]] += 1
    s = 0
    for c in range(ncell):
        s += begin[c]
        begin[c] = s           # now the END of cell c
    begin[ncell] = n
    order = np.empty(n, np.int32)
    for i in range(n - 1, -1, -1):
        c = cid[i]
        begin[c] -= 1
        order[begin[c]] = i    # reversed fill keeps ascending original index inside a cell
    return order, begin


@njit(cache=True)
def _gather(pos, order):
    out = np.empty_like(pos)
    for k in range(order.shape[0]):
        o = order[k]
        out[k, 0] = pos[o, 0]
        out[k, 1] = pos[o, 1]
        out[k, 2] = pos[o, 2]
    return out


@njit(cache=True)
def _link(sp, begin, nc, box, ell):
    """Union-find over particles in cell order. Roots are always the lowest sorted index."""
    n = sp.shape[0]
    parent = np.arange(n).astype(np.int32)
    l2 = ell * ell
    half = 0.5 * box
    for ix in range(nc):
        for iy in range(nc):
            for iz in range(nc):
                c = (ix * nc + iy) * nc + iz
                a0, a1 = begin[c], begin[c + 1]
                if a1 == a0:
                    continue
                # pairs inside the cell
                for i in range(a0, a1):
                    xi, yi, zi = sp[i, 0], sp[i, 1], sp[i, 2]
                    for j in range(i + 1, a1):
                        dx = abs(xi - sp[j, 0]); dx = box - dx if dx > half else dx
                        dy = abs(yi - sp[j, 1]); dy = box - dy if dy > half else dy
                        dz = abs(zi - sp[j, 2]); dz = box - dz if dz > half else dz
                        if dx * dx + dy * dy + dz * dz < l2:
                            ri = _find(parent, i); rj = _find(parent, j)
                            if ri != rj:
                                if ri < rj:
                                    parent[rj] = ri
                                else:
                                    parent[ri] = rj
                # 13 forward neighbour cells
                for ox in range(-1, 2):
                    for oy in range(-1, 2):
                        for oz in range(-1, 2):
                            if not (ox > 0 or (ox == 0 and (oy > 0 or (oy == 0 and oz > 0)))):
                                continue
                            jx = (ix + ox) % nc
                            jy = (iy + oy) % nc
                            jz = (iz + oz) % nc
                            d = (jx * nc + jy) * nc + jz
                            b0, b1 = begin[d], begin[d + 1]
                            if b1 == b0:
                                continue
                            for i in range(a0, a1):
                                xi, yi, zi = sp[i, 0], sp[i, 1], sp[i, 2]
                                for j in range(b0, b1):
                                    dx = abs(xi - sp[j, 0]); dx = box - dx if dx > half else dx
                                    dy = abs(yi - sp[j, 1]); dy = box - dy if dy > half else dy
                                    dz = abs(zi - sp[j, 2]); dz = box - dz if dz > half else dz
                                    if dx * dx + dy * dy + dz * dz < l2:
                                        ri = _find(parent, i); rj = _find(parent, j)
                                        if ri != rj:
                                            if ri < rj:
                                                parent[rj] = ri
                                            else:
                                                parent[ri] = rj
    for i in range(n):         # full compression: parent[i] is now the root
        parent[i] = _find(parent, i)
    return parent


@njit(cache=True)
def _group_stats(parent, order):
    """Per root: member count and the lowest ORIGINAL index among members."""
    n = parent.shape[0]
    size = np.zeros(n, np.int32)
    first = np.full(n, np.iinfo(np.int32).max, np.int32)
    for i in range(n):
        r = parent[i]
        size[r] += 1
        if order[i] < first[r]:
            first[r] = order[i]
    return size, first


@njit(cache=True)
def _scatter_labels(parent, order, rank):
    labels = np.empty(parent.shape[0], np.int32)
    for i in range(parent.shape[0]):
        labels[order[i]] = rank[parent[i]]
    return labels


def fof_labels(pos, box, b=0.2, min_part=20, ell=None):
    """Periodic FOF. pos: (N, 3) float32 in [0, box). Linking length ell (default b * box / N^(1/3)).
    Returns (labels int32 (N,), -1 = in no group of >= min_part, others 0.. by decreasing size, stats)."""
    pos = np.ascontiguousarray(pos, dtype=np.float32)
    n = len(pos)
    if n >= 2**31 - 1:
        raise ValueError("more than 2^31 particles: int32 indices overflow")
    if ell is None:
        ell = b * box / n ** (1.0 / 3.0)
    nc = int(min(box / (CELL_FACTOR * ell), MAX_NC))
    if nc < 3:
        nc = int(box / ell)
        if nc < 3:
            raise ValueError(f"box {box} is less than 3 linking lengths ({ell})")
    cell = box / nc            # >= ell by construction
    cid = _cell_ids(pos, 1.0 / cell, nc)
    order, begin = _counting_sort(cid, nc ** 3)
    del cid
    sp = _gather(pos, order)
    parent = _link(sp, begin, nc, np.float32(box), np.float32(ell))
    del sp, begin
    size, first = _group_stats(parent, order)
    roots = np.nonzero((size >= min_part) & (parent == np.arange(n, dtype=np.int32)))[0]
    # decreasing size, then lowest original index: independent of the cell ordering
    sel = np.lexsort((first[roots], -size[roots].astype(np.int64)))
    rank = first                                  # reuse the buffer: rank[root] -> group number
    rank[:] = -1
    rank[roots[sel]] = np.arange(len(roots), dtype=np.int32)
    sizes = size[roots[sel]]
    labels = _scatter_labels(parent, order, rank)
    stats = dict(ell=float(ell), nc=nc, n=n, ngroups=int(len(roots)), n_in_groups=int(sizes.sum()),
                 largest=int(sizes[0]) if len(sizes) else 0)
    return labels, stats


# ----------------------------------------------------------------------------------------
# snapshots in / labels out
# ----------------------------------------------------------------------------------------
def read_positions(out, snapid, threads=4):
    """PartType1 coordinates of a converted snapshot in file order: (pos float32 (N,3), BoxSize)."""
    import h5py
    files = _snapshot_files(out, snapid)
    counts, box = [], None
    for f in files:
        with h5py.File(f, "r") as h:
            counts.append(int(h["Header"].attrs["NumPart_ThisFile"][DM]))
            box = float(h["Header"].attrs["BoxSize"])
    edges = np.concatenate([[0], np.cumsum(counts)])
    pos = np.empty((int(edges[-1]), 3), np.float32)

    def one(i):
        with h5py.File(files[i], "r") as h:
            h["PartType1/Coordinates"].read_direct(pos, np.s_[:], np.s_[edges[i]:edges[i + 1]])

    with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
        for fut in [ex.submit(one, i) for i in range(len(files))]:
            fut.result()
    return pos, box


def read_epoch_ids(out):
    """{epoch name: snapshot id} from OUT/snapshots.txt (written by abacus_to_hbt.py)."""
    p = Path(out) / "snapshots.txt"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run the converter first")
    return {e: int(i) for i, e in (ln.split() for ln in p.read_text().splitlines() if ln.strip())}


def process_epoch(job):
    out, epoch, snapid, b, min_part, labels_dir, inplace, threads = job
    t0 = time.time()
    try:
        pos, box = read_positions(out, snapid, threads)
        t_read = time.time() - t0
        t1 = time.time()
        labels, st = fof_labels(pos, box, b, min_part)
        del pos
        t_fof = time.time() - t1
        if labels_dir:
            tmp = Path(labels_dir) / f"labels_{epoch}.npy.tmp.npy"
            np.save(tmp, labels)
            tmp.rename(Path(labels_dir) / f"labels_{epoch}.npy")
        if inplace:
            add_labels(out, snapid, labels, threads)
        return dict(epoch=epoch, id=snapid, ok=True, t_read=t_read, t_fof=t_fof, t_total=time.time() - t0,
                    peak_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, **st)
    except Exception:
        return dict(epoch=epoch, id=snapid, ok=False, err=traceback.format_exc(), t_total=time.time() - t0)


def run_pool(jobs, nworkers, mp_context="spawn"):
    """As in abacus_to_hbt: if a worker is killed (OOM) rebuild the pool at half size and retry."""
    failures, pending = [], list(jobs)
    total, done_n = len(jobs), 0
    while pending:
        ex = ProcessPoolExecutor(max_workers=min(nworkers, len(pending)), mp_context=mp.get_context(mp_context))
        futs = {ex.submit(process_epoch, j): j for j in pending}
        broken = False
        try:
            for fut in as_completed(futs):
                try:
                    r = fut.result()
                except BrokenProcessPool:
                    broken = True
                    break
                pending.remove(futs[fut])
                done_n += 1
                if r["ok"]:
                    log(f"[{done_n}/{total}] {r['epoch']} id {r['id']}: {r['ngroups']} groups, "
                        f"{r['n_in_groups'] / r['n']:.1%} of particles grouped, largest {r['largest']} "
                        f"(read {r['t_read']:.0f}s, fof {r['t_fof']:.0f}s, total {r['t_total']:.0f}s, "
                        f"peak {r['peak_gb']:.1f} GB)")
                else:
                    failures.append(r)
                    log(f"[{done_n}/{total}] FAILED {r['epoch']} id {r['id']}:\n{r['err']}")
        finally:
            if broken:
                for fut, j in futs.items():
                    if j in pending and fut.done() and not fut.cancelled() and fut.exception() is None:
                        r = fut.result()
                        pending.remove(j)
                        done_n += 1
                        (log(f"[{done_n}/{total}] {r['epoch']}: done") if r["ok"] else failures.append(r))
            ex.shutdown(wait=False, cancel_futures=True)
        if broken:
            if nworkers <= 1:
                bad = pending.pop(0)
                log(f"a single worker was killed on {bad[1]}; skipping it (out of memory?)")
                failures.append(dict(epoch=bad[1], id=bad[2], ok=False, err="worker killed even when alone"))
            else:
                nworkers = max(1, nworkers // 2)
                log(f"a worker was killed (most likely out of memory); retrying {len(pending)} epoch(s) "
                    f"with {nworkers} worker(s)")
    return failures


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="HBT root written by abacus_to_hbt.py (has snapshots/ and snapshots.txt)")
    ap.add_argument("--epochs", nargs="+", default=["all"], help="epoch names (z0.000 ...) or 'all'")
    ap.add_argument("--b", type=float, default=0.2, help="linking length / mean interparticle spacing")
    ap.add_argument("--min-part", type=int, default=20)
    ap.add_argument("--labels-dir", help="also save labels_<epoch>.npy here (not under the sim tree)")
    ap.add_argument("--inplace", action="store_true", help="write GroupNumber into the snapshot files")
    ap.add_argument("--workers", type=int, default=0, help="epochs in parallel (0 = from cores and RAM)")
    ap.add_argument("--threads", type=int, default=4, help="I/O threads per epoch")
    ap.add_argument("--mp-context", default="spawn", choices=["spawn", "forkserver", "fork"])
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--nranks", type=int, default=1)
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip epochs whose snapshots already have GroupNumber (with --inplace) or "
                         "whose labels file exists")
    a = ap.parse_args(argv)
    if not (a.inplace or a.labels_dir):
        sys.exit("nothing to write: give --inplace and/or --labels-dir")
    out = Path(a.out)
    for root in PROTECTED_ROOTS:
        root_r = Path(root).resolve()
        for p in [out, Path(a.labels_dir)] if a.labels_dir else [out]:
            pr = p.resolve()
            if pr == root_r or root_r in pr.parents:
                sys.exit(f"refusing: {pr} is under the protected tree {root_r}")
    ids = read_epoch_ids(out)
    wanted = list(ids) if a.epochs == ["all"] else a.epochs
    for e in wanted:
        if e not in ids:
            sys.exit(f"unknown epoch {e}")
    if a.labels_dir:
        Path(a.labels_dir).mkdir(parents=True, exist_ok=True)
    todo = []
    for e in wanted[a.rank::a.nranks]:
        sid = ids[e]
        try:
            _snapshot_files(out, sid)
        except FileNotFoundError:
            log(f"skip {e}: not converted yet")
            continue
        done = (has_labels(out, sid) if a.inplace else True) and \
               (Path(a.labels_dir, f"labels_{e}.npy").exists() if a.labels_dir else True)
        if a.skip_existing and done:
            log(f"skip {e} (id {sid})")
        else:
            todo.append((e, sid))
    n_part = 0
    if todo:
        import h5py
        with h5py.File(_snapshot_files(out, todo[0][1])[0], "r") as h:
            n_part = int(h["Header"].attrs["NumPart_Total"][DM])
    cores = os.cpu_count() or 1
    nworkers = a.workers or int(max(1, min(len(todo), cores, DEFAULT_MAX_WORKERS,
                                           0.8 * available_ram() // max(n_part * BYTES_PER_PARTICLE, 1))))
    log(f"{len(todo)} epochs to label (b={a.b}, min_part={a.min_part}) with {nworkers} worker processes "
        f"(N={n_part})")
    failures = []
    if todo:
        jobs = [(str(out), e, sid, a.b, a.min_part, a.labels_dir, a.inplace, a.threads) for e, sid in todo]
        failures = run_pool(jobs, nworkers, a.mp_context)
    if failures:
        log(f"{len(failures)} epoch(s) failed: {[f['epoch'] for f in failures]}; rerun with --skip-existing")
        sys.exit(1)
    log("done")


if __name__ == "__main__":
    main()
