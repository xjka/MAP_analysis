#!/usr/bin/env python3
"""Rewrite Abacus Aurora timeslice particle files as Apostle/EAGLE-style HDF5 for HBT-HERONS.

HBT-HERONS has no Abacus reader. Its `apostle` reader (src/io/apostle_io/apostle_io.cpp) takes the
particles AND their FOF membership from the snapshot files themselves: each snapshot file holds
PartType1/{Coordinates, Velocity, ParticleIDs, GroupNumber}, GroupNumber being the particle's FOF
group (negative = in no group). Particle order is irrelevant: HBT sorts by group itself. So the
snapshots stay in Abacus read order, and FOF labels are just one more column.

  OUT/snapshots/snapshot_NNN/snap_NNN.<i>.hdf5     SnapshotPath = OUT/snapshots
  OUT/subcat/                                      SubhaloPath (HBT's catalogues, written by HBT)
  OUT/hbt.conf, OUT/snapshots.txt                  config and id -> epoch table

The sim directory is only ever READ. Everything is written under OUT (the script refuses an OUT
that is inside, or contains, the sim, and any OUT under /flare/Abacus/AbacusAurora).

Abacus timeslice files (`subsamples/z*/col*/particles_all_*.asdf`, layout output_particle_24b)
hold the dark matter AND the MAPs (`is_map`); only the dark matter (is_map == False) is written.
They carry no FOF groups. FOF labels come from --labels-dir: per epoch an int64 .npy file
`labels_<epoch>.npy` aligned to this script's read order (DM rows only, input files in sorted
order, rows in file order), -1 = in no group. They can be given at conversion time, or added
afterwards in place with --add-labels (appends the GroupNumber column to the snapshots already
written; nothing is reordered or rewritten). HBT needs the GroupNumber column to run.

Units written: Mpc/h, km/s, 1e10 Msun/h. HBT multiplies the stored Velocity by sqrt(a), so the
physical peculiar velocities Abacus stores are written divided by sqrt(a). Snapshot id = index of
the epoch in decreasing-z order (id 0 = earliest).

Parallelism (one node): epochs are independent, so a pool of --workers processes each converts
one whole epoch (read its files, write). Inside an epoch the output files are written by
--threads threads (HDF5/numpy release the GIL). Peak memory per epoch is estimated at NP * 64 bytes
(NP * 56 without labels, ~33 GB for 512M particles): the default --workers divides the available
RAM (physical or cgroup limit) by that, capped at the core count and at 16. If a worker is killed
(usually out of memory) the pool is rebuilt with half the workers and only the unfinished epochs
are retried. Workers start with `spawn` (clean interpreters; --mp-context to change). Output is written to *.tmp directories and renamed on completion, so a killed job
leaves no half-written snapshot and `--skip-existing` makes resubmission cheap. Across nodes,
shard with --rank/--nranks (epochs[rank::nranks]).

    python abacus_to_hbt.py SIM OUT --epochs all [--labels-dir DIR] [--workers 16] [--threads 4]
    python abacus_to_hbt.py SIM OUT --epochs all --add-labels --labels-dir DIR   # labels afterwards
"""
import argparse
import faulthandler
import multiprocessing as mp
import os
import re
import resource
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from glob import glob
from pathlib import Path

import numpy as np

MASS_UNIT_HMSUN = 1e10        # internal mass unit (HBT default MassInMsunh)
NTYPES = 6
DM = 1                        # PartType1
NULL_GROUP = -1               # GroupNumber of a particle in no group (HBT: any negative)
MAX_GROUP_NUMBER = 1 << 30    # HBT asserts HostId <= 1<<30 (its NullGroupId)
# Nothing is ever written under these, whatever OUT is (the parent dir is group-writable)
PROTECTED_ROOTS = ("/flare/Abacus/AbacusAurora",)
# Peak bytes per particle in one epoch's worker: pos+vel+id (32) + the concatenation copy and one
# decoded file while reading (~24); labels add ~8. A conservative estimate, not a measurement.
BYTES_PER_PARTICLE_LABELED = 64
BYTES_PER_PARTICLE_PLAIN = 56
DEFAULT_MAX_WORKERS = 16      # raise with --workers once you have seen the real memory use
SUBCHUNK = 4_000_000          # particles converted/written at a time inside one output file


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------------------
# reading Abacus
# ----------------------------------------------------------------------------------------
def list_epochs(sim):
    """Epoch directory names (z0.000, ...) in DECREASING z, i.e. increasing time."""
    eps = [p.name for p in (Path(sim) / "subsamples").iterdir() if re.fullmatch(r"z\d+\.\d+", p.name)]
    return sorted(eps, key=lambda e: -float(e[1:]))


def epoch_files(sim, epoch):
    files = sorted(glob(str(Path(sim) / "subsamples" / epoch / "col*" / "particles_all_*.asdf")))
    if not files:
        raise FileNotFoundError(f"no particles_all_* files for {epoch} under {sim}")
    return files


def read_header(f):
    """The ASDF header of one file without decoding any particles (cheap)."""
    import asdf
    asdf.get_config().validate_on_read = False
    with asdf.open(f, lazy_load=True) as af:
        return dict(af["header"])


def _read_dm_one_file(f):
    """DM rows of one file: (pos float32 (n,3) [-L/2, L/2), vel float32 (n,3), id int64 (n,), header)."""
    import asdf
    from abacusnbody.data.read_abacus import read_asdf
    asdf.get_config().validate_on_read = False
    t = read_asdf(f, load=["pos", "vel", "pid", "is_map"])
    dm = ~np.asarray(t["is_map"])
    pid = np.asarray(t["pid"])[dm].astype(np.int64)
    # pid is the lattice triple (i, j, k) of the particle's initial position: unique and constant
    ids = pid[:, 0] + (pid[:, 1] << 16) + (pid[:, 2] << 32)
    return (np.asarray(t["pos"])[dm].astype(np.float32), np.asarray(t["vel"])[dm].astype(np.float32),
            ids, dict(t.meta))


def read_epoch(sim, epoch):
    """All DM particles of an epoch in sorted-file/row order, plus the header. Files are decoded
    one at a time and each array is concatenated and its parts freed before the next, to keep the
    peak near one extra array's worth."""
    P, V, I, hdr = [], [], [], None
    for f in epoch_files(sim, epoch):
        p, v, i, h = _read_dm_one_file(f)
        P.append(p); V.append(v); I.append(i)
        hdr = hdr or h
    out = []
    for parts in (P, V, I):
        out.append(np.concatenate(parts))
        parts.clear()
    return out[0], out[1], out[2], hdr


# ----------------------------------------------------------------------------------------
# writing Apostle-style HDF5
# ----------------------------------------------------------------------------------------
def _split(n, nfiles):
    """Contiguous [start, stop) chunks of n items over nfiles files."""
    edges = (np.arange(nfiles + 1) * n) // nfiles
    return list(zip(edges[:-1], edges[1:]))


def _fresh(path):
    """An empty directory at `path`, removing any stale one."""
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


def group_numbers(labels):
    """FOF labels -> int32 GroupNumber: negative labels become -1; labels HBT cannot hold
    (>= 2^30) are renumbered densely (0..G-1, order of first sorted label)."""
    labels = np.asarray(labels)
    if len(labels) and int(labels.max()) >= MAX_GROUP_NUMBER:
        out = np.full(len(labels), NULL_GROUP, np.int32)
        g = labels >= 0
        out[g] = np.unique(labels[g], return_inverse=True)[1].astype(np.int32)
        return out
    return np.where(labels >= 0, labels, NULL_GROUP).astype(np.int32)


def snapshot_file_name(snapid, i):
    return f"snap_{snapid:03d}.{i}.hdf5"


def snapshot_dir(out, snapid):
    return Path(out) / "snapshots" / f"snapshot_{snapid:03d}"


def write_snapshot(out, snapid, pos, vel, ids, hdr, nfiles, labels=None, threads=4):
    """Write PartType1 (Coordinates in [0, L), Velocity/sqrt(a), ParticleIDs, and GroupNumber if
    `labels`) contiguously over `nfiles` files, by `threads` threads. Input positions are Abacus's
    [-L/2, L/2). `labels`: raw FOF labels aligned to the particles (-1 = none)."""
    import h5py
    box = float(hdr["BoxSize"])
    a_scale = float(hdr["ScaleFactor"])
    n = len(ids)
    omega_m = float(hdr["Omega_M"])
    omega_l = float(hdr["Omega_DE"]) if "Omega_DE" in hdr else 1.0 - omega_m
    mass = float(hdr["ParticleMassHMsun"]) / MASS_UNIT_HMSUN
    gn = group_numbers(labels) if labels is not None else None
    if gn is not None and len(gn) != n:
        raise ValueError(f"{len(gn)} labels for {n} particles")
    final = snapshot_dir(out, snapid)
    tmp = _fresh(final.with_name(final.name + ".tmp"))
    if n >= 2**32:
        raise ValueError("more than 2^32 particles: NumPart_Total_HighWord not implemented")
    total = np.zeros(NTYPES, np.uint32)
    total[DM] = n

    def one(i, a, b):
        with h5py.File(tmp / snapshot_file_name(snapid, i), "w") as f:
            h = f.create_group("Header")
            h.attrs["NumFilesPerSnapshot"] = np.int32(nfiles)
            h.attrs["BoxSize"] = box
            h.attrs["Time"] = a_scale
            h.attrs["Redshift"] = float(hdr["Redshift"])
            h.attrs["Omega0"] = omega_m
            h.attrs["OmegaLambda"] = omega_l
            h.attrs["HubbleParam"] = float(hdr["H0"]) / 100.0
            mt = np.zeros(NTYPES); mt[DM] = mass
            h.attrs["MassTable"] = mt
            nt = np.zeros(NTYPES, np.int32); nt[DM] = b - a
            h.attrs["NumPart_ThisFile"] = nt
            h.attrs["NumPart_Total"] = total
            h.attrs["NumPart_Total_HighWord"] = np.zeros(NTYPES, np.uint32)
            g = f.create_group("PartType1")
            m = int(b - a)
            d_x = g.create_dataset("Coordinates", shape=(m, 3), dtype="f4")
            d_v = g.create_dataset("Velocity", shape=(m, 3), dtype="f4")
            d_i = g.create_dataset("ParticleIDs", shape=(m,), dtype="i8")
            d_g = g.create_dataset("GroupNumber", shape=(m,), dtype="i4") if gn is not None else None
            sqrt_a = np.float32(np.sqrt(a_scale))
            for s0 in range(0, m, SUBCHUNK):       # bounded temporaries, whatever the file size
                s1, lo, hi = min(s0 + SUBCHUNK, m), a + s0, a + min(s0 + SUBCHUNK, m)
                x = np.mod(pos[lo:hi].astype(np.float64) + box / 2, box).astype(np.float32)
                x[x >= box] = 0.0                   # float32 rounding can land exactly on box
                d_x[s0:s1] = x
                d_v[s0:s1] = (vel[lo:hi] / sqrt_a).astype(np.float32)
                d_i[s0:s1] = ids[lo:hi]
                if d_g is not None:
                    d_g[s0:s1] = gn[lo:hi]

    with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
        for fut in [ex.submit(one, i, a, b) for i, (a, b) in enumerate(_split(n, nfiles))]:
            fut.result()
    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)


def _snapshot_files(out, snapid):
    d = snapshot_dir(out, snapid)
    files = sorted(d.glob(f"snap_{snapid:03d}.*.hdf5"), key=lambda p: int(p.name.split(".")[-2]))
    if not files:
        raise FileNotFoundError(f"no snapshot files in {d}")
    return files


def add_labels(out, snapid, labels, threads=4):
    """Append (or replace) the GroupNumber column in a snapshot already written by this script.
    `labels` must be aligned to the snapshot's particle order (= this script's read order).
    Only touches our own output, in place; nothing is reordered. Returns #particles labelled."""
    import h5py
    files = _snapshot_files(out, snapid)
    counts = []
    for f in files:
        with h5py.File(f, "r") as h:
            counts.append(int(h["Header"].attrs["NumPart_ThisFile"][DM]))
    gn = group_numbers(labels)
    if len(gn) != sum(counts):
        raise ValueError(f"{len(gn)} labels for {sum(counts)} particles in snapshot {snapid}")
    edges = np.concatenate([[0], np.cumsum(counts)])

    def one(i):
        with h5py.File(files[i], "r+") as h:
            g = h["PartType1"]
            if "GroupNumber" in g:
                del g["GroupNumber"]
            g.create_dataset("GroupNumber", data=gn[edges[i]:edges[i + 1]])

    with ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
        for fut in [ex.submit(one, i) for i in range(len(files))]:
            fut.result()
    return len(gn)


def has_labels(out, snapid):
    """True if every file of the snapshot already has a GroupNumber column."""
    import h5py
    try:
        for f in _snapshot_files(out, snapid):
            with h5py.File(f, "r") as h:
                if "GroupNumber" not in h["PartType1"]:
                    return False
        return True
    except (FileNotFoundError, OSError, KeyError):
        return False


# ----------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------
def write_config(out, hdr, n_epochs, min_part=20):
    """hbt.conf. The apostle reader takes units and softening from here, not from the files.
    Abacus softening is proper (`ProperSoftening`) up to a comoving cap (`SofteningMax`), which is
    HBT's SofteningHalo (comoving) with MaxPhysicalSofteningHalo (physical)."""
    out = Path(out)
    soft = float(hdr.get("SofteningLength", 0.0))
    if int(hdr.get("ProperSoftening", 0)) and "SofteningMax" in hdr:
        soft_comoving, soft_physical = float(hdr["SofteningMax"]), soft
    else:
        soft_comoving, soft_physical = soft, -1
    (out / "hbt.conf").write_text(f"""# HBT-HERONS configuration for {out.name} (written by abacus_to_hbt.py)
[Compulsary Params]
SnapshotPath {out}/snapshots
HaloPath {out}/snapshots
SubhaloPath {out}/subcat
SnapshotFileBase snap
MaxSnapshotIndex {n_epochs - 1}
BoxSize {float(hdr['BoxSize'])}
SofteningHalo {soft_comoving}
MaxPhysicalSofteningHalo {soft_physical}

[Reader]
SnapshotFormat apostle
GroupFileFormat apostle_particle_index

[Units]
MassInMsunh {MASS_UNIT_HMSUN:g}
LengthInMpch 1
VelInKmS 1

MinNumPartOfSub {min_part}
""")


def _epoch_done(out, snapid, nfiles, with_labels):
    last = snapshot_dir(out, snapid) / snapshot_file_name(snapid, nfiles - 1)
    return last.exists() and (has_labels(out, snapid) or not with_labels)


def process_epoch(job):
    """One epoch end to end (runs in a worker process). Returns a summary dict."""
    t0 = time.time()
    sim, out, epoch, snapid, labels_dir, nfiles, threads, check, blosc_threads, add_only = job
    faulthandler.enable()          # a hard crash (segfault) prints the Python stack to stderr
    try:
        if add_only:
            labels = np.asarray(np.load(Path(labels_dir) / f"labels_{epoch}.npy", mmap_mode="r"))
            n = add_labels(out, snapid, labels, threads)
            return dict(epoch=epoch, id=snapid, ok=True, n=n, t_read=0.0, t_total=time.time() - t0)
        try:
            import blosc2
            blosc2.set_nthreads(blosc_threads)
        except ImportError:
            pass
        pos, vel, ids, hdr = read_epoch(sim, epoch)
        t1 = time.time()
        if check:
            assert len(ids) == int(hdr["NP"]), f"{epoch}: {len(ids)} DM rows but NP={hdr['NP']}"
            assert np.unique(ids).size == len(ids), f"{epoch}: duplicate particle ids"
        labels = None
        if labels_dir:
            labels = np.asarray(np.load(Path(labels_dir) / f"labels_{epoch}.npy", mmap_mode="r"))
        write_snapshot(out, snapid, pos, vel, ids, hdr, nfiles, labels, threads)
        return dict(epoch=epoch, id=snapid, ok=True, n=len(ids), t_read=t1 - t0, t_total=time.time() - t0,
                    peak_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
    except Exception:
        return dict(epoch=epoch, id=snapid, ok=False, err=traceback.format_exc(), t_total=time.time() - t0)


def available_ram():
    """Bytes of RAM this process may use: the smaller of physical RAM and any cgroup limit."""
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        ram = 256 * 2**30
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = Path(p).read_text().strip()
            if v.isdigit():
                ram = min(ram, int(v))
        except OSError:
            pass
    return ram


def default_workers(n_todo, np_particles, labeled):
    cores = os.cpu_count() or 1
    per = np_particles * (BYTES_PER_PARTICLE_LABELED if labeled else BYTES_PER_PARTICLE_PLAIN)
    return int(max(1, min(n_todo, cores, DEFAULT_MAX_WORKERS, 0.8 * available_ram() // max(per, 1))))


def run_pool(jobs, nworkers, mp_context):
    """Run process_epoch over jobs; if a worker dies abruptly (usually the OOM killer) the pool is
    rebuilt with half the workers and only the unfinished epochs are retried. Returns the list of
    failed-epoch result dicts."""
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
                    pk = f", peak {r['peak_gb']:.1f} GB" if "peak_gb" in r else ""
                    log(f"[{done_n}/{total}] {r['epoch']} id {r['id']}: {r['n']} DM particles "
                        f"(read {r['t_read']:.0f}s, total {r['t_total']:.0f}s{pk})")
                else:
                    failures.append(r)
                    log(f"[{done_n}/{total}] FAILED {r['epoch']} id {r['id']}:\n{r['err']}")
        finally:
            if broken:   # keep whatever finished before the pool broke
                for fut, j in futs.items():
                    if j in pending and fut.done() and not fut.cancelled() and fut.exception() is None:
                        r = fut.result()
                        pending.remove(j)
                        done_n += 1
                        (log(f"[{done_n}/{total}] {r['epoch']} id {r['id']}: done") if r["ok"]
                         else failures.append(r))
            ex.shutdown(wait=False, cancel_futures=True)
        if broken:
            for j in pending:   # a killed worker can leave a half-written *.tmp directory; ours only
                shutil.rmtree(snapshot_dir(j[1], j[3]).with_name(snapshot_dir(j[1], j[3]).name + ".tmp"),
                              ignore_errors=True)
            if nworkers <= 1:
                # one worker runs the queue in order, so the first pending epoch is the one that died
                bad = pending.pop(0)
                log(f"a single worker was killed converting {bad[2]}; skipping it (out of memory? check "
                    f"`free -g` / the job's memory limit) and continuing with {len(pending)} more")
                failures.append(dict(epoch=bad[2], id=bad[3], ok=False, err="worker killed even when running alone"))
            else:
                nworkers = max(1, nworkers // 2)
                log(f"a worker process was killed (most likely out of memory); retrying "
                    f"{len(pending)} epoch(s) with {nworkers} worker(s)")
    return failures


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sim", help="Abacus sim directory (has subsamples/z*/); only ever read")
    ap.add_argument("out", help="output root for HBT (use /flare/Abacus/ajohnson/..., not $HOME)")
    ap.add_argument("--epochs", nargs="+", default=["all"], help="epoch dir names (z0.000 ...) or 'all'")
    ap.add_argument("--labels-dir", help="directory of labels_<epoch>.npy FOF labels (see module doc)")
    ap.add_argument("--add-labels", action="store_true",
                    help="only append the GroupNumber column (from --labels-dir) to snapshots already "
                         "written by this script; does not read the Abacus particle files")
    ap.add_argument("--nfiles-out", type=int, default=8, help="output files per snapshot")
    ap.add_argument("--workers", type=int, default=0,
                    help="epochs converted concurrently (processes); 0 = from cores and RAM")
    ap.add_argument("--threads", type=int, default=4, help="writer threads per epoch")
    ap.add_argument("--mp-context", default="spawn", choices=["spawn", "forkserver", "fork"],
                    help="how worker processes start; spawn gives each a clean interpreter (default)")
    ap.add_argument("--blosc-threads", type=int, default=1, help="blosc2 decompression threads per worker")
    ap.add_argument("--rank", type=int, default=0, help="shard epochs over nodes: epochs[rank::nranks]")
    ap.add_argument("--nranks", type=int, default=1)
    ap.add_argument("--check", action="store_true", help="assert DM count == NP and ids unique (slow)")
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args(argv)

    all_eps = list_epochs(a.sim)
    wanted = all_eps if a.epochs == ["all"] else a.epochs
    for e in wanted:
        if e not in all_eps:
            sys.exit(f"unknown epoch {e}; have {all_eps[0]} ... {all_eps[-1]}")
    if a.add_labels and not a.labels_dir:
        sys.exit("--add-labels needs --labels-dir")
    out = Path(a.out)
    sim_r, out_r = Path(a.sim).resolve(), out.resolve()
    for root in PROTECTED_ROOTS:
        root_r = Path(root).resolve()
        if out_r == root_r or root_r in out_r.parents:
            sys.exit(f"refusing: output {out_r} is under the protected tree {root_r}; "
                     f"write somewhere else (e.g. /flare/Abacus/ajohnson/...)")
    if out_r == sim_r or sim_r in out_r.parents or out_r in sim_r.parents:
        sys.exit(f"refusing: output {out_r} and sim {sim_r} must be separate directories "
                 f"(the sim is only ever read)")
    out.mkdir(parents=True, exist_ok=True)
    with_labels = bool(a.labels_dir)
    todo = []
    for e in wanted[a.rank::a.nranks]:
        snapid = all_eps.index(e)
        done = has_labels(out, snapid) if a.add_labels else _epoch_done(out, snapid, a.nfiles_out, with_labels)
        if a.skip_existing and done:
            log(f"skip {e} (id {snapid})")
        else:
            todo.append((e, snapid))

    hdr = read_header(epoch_files(a.sim, all_eps[0])[0])
    nworkers = a.workers or default_workers(len(todo), int(hdr["NP"]), with_labels)
    log(f"{len(todo)} epochs to {'label' if a.add_labels else 'convert'} on {os.cpu_count()} cores with "
        f"{nworkers} worker processes x {a.threads} writer threads (NP={hdr['NP']})")

    failures = []
    if todo:
        jobs = [(a.sim, str(out), e, sid, a.labels_dir, a.nfiles_out, a.threads, a.check, a.blosc_threads,
                 a.add_labels) for e, sid in todo]
        failures = run_pool(jobs, nworkers, a.mp_context)

    if a.rank == 0:
        write_config(out, hdr, len(all_eps))
        with open(out / "snapshots.txt", "w") as f:
            for i, e in enumerate(all_eps):
                f.write(f"{i} {e}\n")
    if failures:
        log(f"{len(failures)} epoch(s) failed: {[f['epoch'] for f in failures]}; rerun with --skip-existing")
        sys.exit(1)
    log("done")


if __name__ == "__main__":
    main()
