#!/usr/bin/env python
"""Recompute history losses under the current loss_config().

You normally don't need this: optimize_maps.py tags every history record with
the loss_config() (bin edges, target dataset + source, compare_redshifts,
abundance, LOSS_VERSION) it was computed under, and at startup recomputes any
record whose config differs from the current one (History.refresh_stale).

The loss settings come from the same TOML config optimize_maps.py uses, so
pass the campaign's config:

  * refresh the config's current-spec history in place, without starting a
    tuning run:
        python recompute_losses.py --config tuning.toml
  * recompute some other history file (another spec's, a pre-spec history,
    a backup) into a new file, leaving the original intact:
        python recompute_losses.py --config tuning.toml \\
            --history .../optimize_maps_history.json.bak \\
            --output .../optimize_maps_history_recomputed.json
  * --force: recompute every scored record even if its loss_config already
    matches (e.g. after editing compute_loss() without bumping LOSS_VERSION).

Run dirs are found at $ABACUS_OUTPUT_ROOT/<sim_name>, so any history file can
be recomputed regardless of where it lives.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimize_maps import History, configure, load_config, reference_target  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True,
                   help="tuning TOML config defining the loss (and the default history)")
    p.add_argument("--history", default=None,
                   help="history JSON to recompute (default: the config's current-spec history)")
    p.add_argument("--output", default=None,
                   help="write the result here instead of overwriting the input history")
    p.add_argument("--force", action="store_true",
                   help="recompute every scored record, not just ones with a stale loss_config")
    args = p.parse_args()

    cfg = configure(load_config(args.config))
    history_path = Path(args.history) if args.history else cfg.history_path
    if not history_path.exists():
        sys.exit(f"[recompute] no history at {history_path}")

    target = reference_target()
    history = History(history_path)
    history.refresh_stale(target, force=args.force,
                          out_path=Path(args.output) if args.output else history_path)


if __name__ == "__main__":
    main()
