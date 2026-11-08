# MAP_analysis

Analysis, tuning, and job scripts for MAPs (the on-the-fly halo/galaxy tracers)
in [Abacus](https://github.com/abacusorg/abacus) simulations, mainly the
AbacusAurora runs on Aurora. This repo used to live in the Abacus repo as
`acj/` and was moved out so these scripts can change independently of Abacus.

## Layout

```
MAP_analysis/
├── utilities/          analysis modules and command-line scripts
├── parameter_tuning/   Bayesian optimization of MAP parameters
├── pbs/                PBS job scripts for the utilities
├── build_scripts/      notes/scripts for building Abacus and dependencies
├── notebooks/          Jupyter notebooks (not tracked: see .gitignore)
├── analysis_data -> /flare/Abacus/$USER/analysis_data/   (untracked symlink)
└── Images        -> /flare/Abacus/$USER/saves/           (untracked symlink)
```

`analysis_data/` (caches, `.npz` results) and `Images/` (figures) are symlinks
to `/flare` and are deliberately not committed. Many scripts write to them
with paths relative to the working directory (see [Running](#running)).

### `utilities/`

| File | Purpose |
|---|---|
| `utilities.py` | Core library: maplog reading (`read_maplog_dir`, cached variants), correlation functions (`get_xi`, `get_map_xi`), TNG/FLAMINGO loaders, and the main comparison plots. Most other files import from it. |
| `match_maps_to_tng.py` | Match MAPs to TNG halos by position at one or more redshifts; HOD and velocity comparisons. |
| `map_by_map_comparison.py` | Match MAPs between two runs (tiered by multiplicity) and plot unmatched fractions/residuals. |
| `group_finding.py` | Friends-of-friends grouping of MAPs from maplogs, merger-forest reading, and a MAP-population HOD. |
| `run_group_scan.py` | Time-resolved central/infalling/orbiting classification of every MAP in a simulation. |
| `box_maps.py` | Cut a non-periodic cube of final-timeslice MAPs out of a very large maplog set. |
| `box_xi.py` | Real-space MAP xi(r) in random non-periodic sub-boxes of a large run. |
| `merger_rate_subset.py` | Merger rates of final-timeslice MAPs from a random subset of maplog files. |
| `plot_map_variations.py` | xi(r) of the `varMAP` MAP-parameter variations vs. the fiducial. |
| `plot_cosmo_variations.py`, `cache_cosmo_variations_xi.py` | xi(r) of the `emulatorslim0p5` cosmology variations; the second fills the xi cache the first reads. |
| `run_sim_diagnostics.py` | Batch MAP diagnostics (xi evolution, multiplicity, velocities, fractions) over many simulations via MPI. |
| `flagship_analysis.py` | Read-only analysis of the flagship run's per-step timing files. |
| `read_timebin.py` | Reader for Abacus `stepXXXX.time.bin` timing logs. |
| `abacus_to_hbt.py`, `abacus_fof.py` | Convert Abacus timeslices to Apostle/EAGLE-style HDF5 for HBT-HERONS, and add periodic FoF labels. |
| `illustris_utilities.py` | Older MAP-to-TNG matching helpers (particle IDs, positions). |
| `group_tasks.py`, `make_lc_task.py` | Write disBatch task files for group/lightcone post-processing. |
| `scratch.py` | Older plotting and exploratory code. |

### `parameter_tuning/`

- `optimize_maps.py`: tunes MAP parameters so a small Abacus run's MAP
  correlation function matches a reference (TNG, FLAMINGO, or a fiducial Abacus
  MAP catalog). Each evaluation is a full Abacus run, and the optimizer is a GP
  surrogate (skopt).
  - **Settings:** everything comes from a TOML config passed with `--config`:
    the par2, the output directory, `[[params]]` (names and bounds), the loss
    binning, the target dataset, and the optimizer settings.
  - **History:** each distinct `[[params]]` spec gets its own history under
    `<output_dir>/spec_<hash>/`, so changing the spec starts fresh.
    `--import-history` carries over older points whose parameter names match
    and that lie inside the bounds.
  - **Changing loss settings** keeps the history; stale losses are recomputed
    from the stored maplogs.
- `recompute_losses.py`: recomputes stored losses under the current loss
  settings without starting a tuning run.

The configs and the PBS scripts that drive tuning live with the par2 files in
the Abacus checkout, under `prod/ParamTuning*/` (`tuning.toml` and `run.pbs`).

### `pbs/`

Job scripts for Aurora: `run_box_xi`, `run_merger_rate`, `run_group_scan`,
`run_match_maps`, `run_sim_diagnostics`, `run_map_variations`, and
`run_cosmo_variations` each run the matching utility. `jupyter_tunnel.pbs`
starts JupyterLab on a compute node, tunnelled back to a login node, with
`qsub -v LOGIN_HOST=$(hostname -s) jupyter_tunnel.pbs`. `debug.pbs` and
`delete.pbs` are utility jobs.

### `build_scripts/`

Records of how Abacus was built on different machines (`aurora_build.sh`,
`franklin_build.sh`, `hal_build.sh`), and `pfft_build.sh`, which patches and
builds pfft-python for MPICH-based systems.

## Running

The Python environment comes from the Abacus checkout (`~/abacus`):

```bash
cd ~/abacus && . env/aurora.sh      # modules + abacus/.venv
cd ~/MAP_analysis
python utilities/box_xi.py --help
```

- **Imports:** modules in `utilities/` import each other by plain name
  (`from utilities import ...`). Running a script as `python utilities/X.py`
  puts `utilities/` on `sys.path` automatically.
- **Working directory:** run from the repo root so that relative `Images/` and
  `analysis_data/` paths resolve to the symlinks. The exceptions are
  `plot_map_variations.py` and the two cosmo-variations scripts, which are run
  from inside `utilities/` and write to `../Images`. The PBS scripts already
  `cd` to the right place.
- **Notebooks:** their first import cell adds `../utilities` to `sys.path`.
- **Abacus checkout:** a few scripts load `abacus.param` straight from
  `$ABACUS_REPO` (default `~/abacus`).
