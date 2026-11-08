#!/usr/bin/env python
"""Recompute optimize_maps_history.json losses under the current loss_config().

You normally don't need this: optimize_maps.py tags every history record with
the loss_config() (R_EDGES, TARGET_DATASET + source, COMPARE_REDSHIFTS,
ABUNDANCE, LOSS_VERSION) it was computed under, and at startup recomputes any
record whose config differs from the current one (History.refresh_stale).

Like optimize_maps.py, the loss settings come from the environment at import
time (R_SPACING, R_MIN, R_MAX, N_R_BINS, TARGET_DATASET, TNG_BASEPATH, ...),
so export the same variables as run.pbs before running this.

Uses:
  * refresh without starting a tuning run (in place):
        python recompute_losses.py /flare/Abacus/$USER/Tuning_png
  * recompute a backup snapshot into a new file, leaving the original intact:
        python recompute_losses.py --input .../optimize_maps_history.json.bak \\
            --output .../optimize_maps_history_recomputed.json
  * --force: recompute every record even if its loss_config already matches
    (e.g. after editing compute_loss() without bumping LOSS_VERSION).

With no positional dirs or --input, defaults to $OPTIMIZE_MAPS_DIR.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimize_maps import History, reference_target, TARGET_DATASET  # noqa: E402


def recompute(history_path: Path, output_dir: Path, out_path: Path | None = None,
              force: bool = False) -> None:
    """Refresh stale (or, with ``force``, all) losses in ``history_path``.

    ``output_dir`` holds the <TARGET_DATASET>_target.npz cache (reused only if
    its loss_config matches) and anchors run dirs at ``output_dir.parent /
    sim_name``. Writes to ``out_path`` (default: in place).
    """
    if not history_path.exists():
        print(f"[recompute] no history at {history_path}, skipping")
        return
    target = reference_target(cache_path=output_dir / f"{TARGET_DATASET}_target.npz")
    history = History(history_path)
    # refresh_stale resolves run dirs against history.path.parent.parent; point
    # it at output_dir in case the history file lives elsewhere.
    history.path = output_dir / history_path.name
    history.refresh_stale(target, force=force, out_path=out_path or history_path)


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="*",
                   help="OUTPUT_DIR(s) whose optimize_maps_history.json to refresh in place "
                        "(ignored if --input is given)")
    p.add_argument("--input", default=None,
                   help="refresh this specific history JSON file instead")
    p.add_argument("--output-dir", default=None,
                   help="OUTPUT_DIR for the target cache and sim_name run dirs, "
                        "when --input is given (default: --input's parent directory)")
    p.add_argument("--output", default=None,
                   help="write the result here instead of overwriting --input")
    p.add_argument("--force", action="store_true",
                   help="recompute every record, not just ones with a stale loss_config")
    args = p.parse_args()

    if args.input:
        history_path = Path(args.input)
        output_dir = Path(args.output_dir) if args.output_dir else history_path.parent
        recompute(history_path, output_dir,
                  Path(args.output) if args.output else None, force=args.force)
        return

    dirs = args.dirs or [os.environ.get(
        "OPTIMIZE_MAPS_DIR", f"/flare/Abacus/{os.environ.get('USER', '')}/Tuning")]
    for d in dirs:
        d = Path(d)
        recompute(d / "optimize_maps_history.json", d, force=args.force)


if __name__ == "__main__":
    main()
