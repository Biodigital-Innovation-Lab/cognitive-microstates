# Microstate sequence analysis (Go/NoGo EEG)

Analysis code for a study of EEG microstate sequences (30 participants x 2
conditions x 20 trials = 1,200 trials; 19 channels; 650 samples per trial at
500 Hz; K = 5 microstates labelled A-E).

Version 1.0.0. Licence: MIT (code, see `LICENSE`); the deposited data are
CC BY 4.0. How to cite: `CITATION.cff`. Software DOI: Zenodo DOI: added after
the first release (see the repository's Releases page and `CITATION.cff`).
Repository: https://github.com/Biodigital-Innovation-Lab/cognitive-microstates

## Purpose and status

This code was re-implemented from the described methods and from the logic of
the original analysis notebooks, and it was verified against the published
values. It is not the original set of scripts.

## Templates (provenance)

The analyses use group templates for K = 4-9 obtained in an October 2025 run
of the clustering procedure described in the paper. These templates are
deposited in Mendeley Data and are inputs to the pipeline. `fit_K5.py` and
`fit_all_K.py` document the procedure; re-running them yields a close but not
identical solution (min |r| per K between matched maps, Hungarian assignment
on |r|: K = 4 0.9999, K = 5 0.9978, K = 6 0.9987, K = 7 0.9785, K = 8 0.9678,
K = 9 0.9972). The original script of that run is not available; this code is
a re-implementation verified against the published values.

One template set is used for every result: the K = 5 solution
(`microstates_K5_templates_reference.npz`, identical to
`microstates_K5_templates.npz`) for back-fitting, label sequences, metrics,
GEE, motif analysis, consensus, Needleman-Wunsch and canonical
correspondence; K = 4-9 for the GEV curve (K selection) and the
within-solution correlation matrices. `fit_K5.py` and `fit_all_K.py` are not
called by `run_all.sh`.

## Windows

The primary analysis uses exact windows (early = samples 50-249, late =
samples 250-449 of the 650-sample string, collapsed after slicing). The
proportional-window variant of the original pipeline is available as a
sensitivity analysis (`--sensitivity`).

## Figures

`05_alignment/within_condition_nw_analysis.py` writes base plots (alignment
null distributions and forest plots); `fit_K5.py` and `fit_all_K.py` (not run
by `run_all.sh`) write topography and correlation plots. The data behind the
template figures (GEV per K, |r| matrices, T4 values) are written as CSV files
(see `RESULTS_MAP.md`). The published figures were edited after
generation and are not reproduced by this code alone.

## Entry points

All locations are given through environment variables (see `config.py`); no
path is hard-coded. Two entry points, plus one option:

```
./run_all.sh --from-eeg        [--sensitivity]
./run_all.sh --from-sequences  [--sensitivity]
```

### A. From raw EEG (`--from-eeg`)

Starts from the raw EEG and the deposited templates and produces every table
of the analysis: label sequences (back-fitting), GEV of the K = 4-9 templates,
template correlation matrices, per-trial metrics, sequence tables, consensus
sequences, motif analysis, split-half reliability, Needleman-Wunsch analysis,
GEE and the Supplementary Tables S1-S5. No clustering is run.

### B. From the deposited label sequences (`--from-sequences`)

Starts from the deposited sequence table (650-letter strings per trial) and
the deposited templates and reproduces everything except `all_K_gev.csv`
(which needs the EEG; the deposited `all_K_gev.csv` is its output): template
correlation matrices, metrics, sequence tables, consensus, motif analysis,
split-half, Needleman-Wunsch, GEE and the Supplementary Tables.

### Option `--sensitivity`

Adds `03_sequences/full_prepost.py` and the motif analysis with proportional
windows (`*_proportional` files). Without it the proportional files are not
written.

## Inputs

Deposited data: Mendeley Data, Version 2, doi:10.17632/6jcnrr9v58.2,
https://data.mendeley.com/datasets/6jcnrr9v58/2 (folders `inputs/` and
`supplementary/`; data dictionary in `README_data.md` there).

| input | used by | environment variable | where |
|---|---|---|---|
| raw EEG `.txt` files (60 files) | `--from-eeg` | `MICROSTATE_DATA_DIR` | available on reasonable request (not deposited) |
| `microstates_K5_templates_reference.npz` (keys `templates` (5, 19), `ch_names`, `map_names`) | `--from-eeg` (back-fitting); `canonical_similarity.py` | `MICROSTATE_REFERENCE_TEMPLATES` | Mendeley doi:10.17632/6jcnrr9v58.2, `inputs/` |
| `microstates_K4_templates.npz` ... `microstates_K9_templates.npz` (keys `templates` (K, 19), `ch_names`, `map_names`) | `gev_by_k.py` (`--from-eeg`); `k_template_correlations.py` (both modes) | `MICROSTATE_TEMPLATES_DIR` (folder) | Mendeley doi:10.17632/6jcnrr9v58.2, `inputs/` |
| `microstate_sequences_K5_reference.csv` | `--from-sequences` | `MICROSTATE_SEQUENCES_DIR` (folder) | Mendeley doi:10.17632/6jcnrr9v58.2, `inputs/` |
| folder for derived files | all | `MICROSTATE_OUT_DIR` (default `./outputs`) | not an input |
| canonical reference maps (optional, see below) | `canonical_similarity.py` | `MICROSTATE_REFERENCE_MAPS_DIR` | not distributed |
| sex label override (optional): `h=Male,m=Female` or `h=Female,m=Male` | GEE, `assemble_supplementary.py` | `MICROSTATE_SEX_MAP` | not an input file |

The first step of both entry points (`tools/verify_inputs.py`) compares the
SHA-256 of the input files with `checksums.sha256` in the deposited folder
(the list is found in the parent folder of the input files; the check is
skipped if there is none).

Raw file-name pattern expected by `01_reshape/reshape.py`:
`<h|m><number>_<code>_<GO|NG>.txt`. Each file has 13,000 rows x 19
tab-separated columns (20 trials x 650 samples). Only the prefix (`h` or `m`),
the number and the condition are read; the `<code>` part is never parsed or
stored.

One set of label sequences is deposited (`reference`: back-fitting to the
K = 5 templates of the October 2025 run). The deposited `inputs/` folder
holds all three input groups, so `MICROSTATE_SEQUENCES_DIR`,
`MICROSTATE_TEMPLATES_DIR` and the folder of `MICROSTATE_REFERENCE_TEMPLATES`
are normally the same folder.

## Outputs (under `MICROSTATE_OUT_DIR`)

- `eeg_trials.npz`, `all_K_gev.csv` (GEV of the K = 4-9 templates; `--from-eeg`
  only)
- `template_correlations/` : `K{4..9}_abs_r.csv` (|r| between the maps of each
  solution), `K6_T4.csv` (T4 value of each K = 6 map)
- `sequences/reference/` : per-sample sequences (CSV) and labels (NPZ)
- `tables/` : metrics (`microstate_metrics_K5_reference.csv`), per-trial
  sequence tables, consensus, k-mer, bigram,
  conditional-successor, split-half, GEE and Needleman-Wunsch tables (files
  of the motif analysis carry the suffix `_exact`, or `_proportional` with
  `--sensitivity`)
- `figures/` : Needleman-Wunsch base plots
- `canonical/` : canonical-correspondence tables (only with the reference maps)
- `supplementary/` : Supplementary Tables S1-S5 as `.xlsx` and one `.csv` per
  sheet
- `logs/` : one log per step

`RESULTS_MAP.md` maps each manuscript item to its script and output file.

## Installation

Python 3.11.14. Verified platform: macOS (Apple silicon, arm64), Python
3.11.14, the versions pinned in `uv.lock`. Other platforms are untested.

With [uv](https://docs.astral.sh/uv/) (recommended):

```
uv sync --frozen
source .venv/bin/activate
```

With pip:

```
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is exported from `uv.lock` (`uv export --no-hashes
--no-emit-project --no-annotate`). The list includes `setuptools`, which
`mne 1.3.1` imports (`pkg_resources`).

## Third-party canonical maps (not distributed)

`02_microstates/canonical_similarity.py` compares the K = 5 maps with
published canonical maps. These maps are not distributed with this code;
obtain them from the original sources:

- Koenig, T., Prichep, L., Lehmann, D., Sosa, P. V., Braeker, E.,
  Kleinlogel, H., Isenhart, R., & John, E. R. (2002). Millisecond by
  millisecond, year by year: Normative EEG microstates and developmental
  stages. NeuroImage, 16(1), 41-48. https://doi.org/10.1006/nimg.2002.1070
- Milz, P., Faber, P., Lehmann, D., Koenig, T., Kochi, K., & Pascual-Marqui,
  R. D. (2016). The functional significance of EEG microstates - Associations
  with modalities of thinking. NeuroImage, 125, 643-656.
  https://doi.org/10.1016/j.neuroimage.2015.08.023
- Artoni, F., Merk, T., Schneider, J., Tiber, V., Sheller, J., Blankertz, B.,
  & Konig, P. (2023). Microstate analysis of resting-state EEG: A standardized
  template for large-scale data. NeuroImage, 277, 120196.
  https://doi.org/10.1016/j.neuroimage.2023.120196
  (verify against the publisher record)

The expected file names are listed in the docstring of the script.

## Run order

`run_all.sh` runs the steps in this order and writes one log per step to
`$MICROSTATE_OUT_DIR/logs/<step>.log` (command with path arguments by file
name, start, end, duration).

`./run_all.sh --from-eeg`

1. `tools/verify_inputs.py` : checksums of the reference templates and the
   K = 4-9 templates
2. `01_reshape/reshape.py` : raw files -> `eeg_trials.npz`
3. `03_sequences/backfit.py` : per-sample labels (reference templates)
4. `02_microstates/gev_by_k.py` : GEV of the deposited K = 4-9 templates ->
   `all_K_gev.csv`
5. common steps

`./run_all.sh --from-sequences`

1. `tools/verify_inputs.py` : checksums of the sequence table and the K = 4-9
   templates
2. `03_sequences/import_sequences.py` : deposited sequences -> `sequences/`
3. common steps

Common steps

- `02_microstates/k_template_correlations.py` : |r| matrices (K = 4-9) and
  K = 6 T4 values
- `03_sequences/metrics.py` : coverage, occurrences, lifespan per trial
- `03_sequences/trial_table.py` : per-trial sequence table (exact windows)
- `05_alignment/consensus.py` : per-participant consensus sequences
- `04_statistics/motif_analysis.py --windows exact`
- `04_statistics/split_half.py` : split-half reliability
- `05_alignment/within_condition_nw_analysis.py --source eeg`
- `04_statistics/gee_sex_condition.py --source eeg`
- `02_microstates/canonical_similarity.py` : only if
  `MICROSTATE_REFERENCE_MAPS_DIR` and `MICROSTATE_REFERENCE_TEMPLATES` are set
- with `--sensitivity`: `03_sequences/full_prepost.py` and
  `04_statistics/motif_analysis.py --windows proportional --source eeg`
- `06_tables/assemble_supplementary.py` : Supplementary Tables S1-S5

## Verify your run

`expected_outputs/OUTPUTS.sha256` lists the SHA-256 of the 73 output files of
a reference `--from-eeg` run (code identical to this version, without
`--sensitivity`, with the canonical reference maps; macOS arm64, Python
3.11.14). Paths in the list are relative to the output folder, so run the
check from inside `MICROSTATE_OUT_DIR`. From the repository folder, after a
run:

```
CODE_DIR="$PWD"
cd "$MICROSTATE_OUT_DIR"
shasum -a 256 -c "$CODE_DIR/expected_outputs/OUTPUTS.sha256"   # macOS
sha256sum -c "$CODE_DIR/expected_outputs/OUTPUTS.sha256"       # Linux
```

Every produced file should be reported `OK`. Files that are not produced are
reported as missing (`FAILED open or read`):

- `--from-sequences`: `all_K_gev.csv` (it needs the EEG; it equals
  `inputs/all_K_gev.csv` of the data deposit). The other 72 listed files
  are produced (with the canonical reference maps).
- either mode without `MICROSTATE_REFERENCE_MAPS_DIR`: the 14 files in
  `canonical/`.

`logs/` and the `_proportional` files written with `--sensitivity` are not in
the list.

## Methods notes

- **Pre/post concatenated sequence (`full_prepost.py`, sensitivity only).**
  The 650-sample string of a trial is split at sample 150 (pre-stimulus /
  post-stimulus limit of the recording). Each part has consecutive repeated
  letters removed, and the two parts are joined without further collapsing, so
  a repeated letter can remain at the junction. The column
  `sequence_collapsed` is the collapse of the whole 650-sample string.
- **Consensus (`consensus.py`).** For each participant, condition and window
  (full, early, late), the 20 collapsed trial strings are aligned one by one
  to a reference string with a global pairwise alignment (match +1, mismatch
  -1, gap open -2, gap extend -0.5). The reference is the string at index
  n // 2 of the length-sorted strings. Each aligned column is decided by the
  most frequent non-gap letter (first one encountered wins ties); the support
  of a column is the winner count divided by the number of strings, gaps
  included. The consensus length equals the number of non-empty columns. The
  window strings come from the exact-window table (samples 50-249 and
  250-449, collapsed after slicing).
- **Effect size r.** Two formulas are used. For the 20 bigram rates of the
  bigram table (S3) Z is taken from the two-sided Mann-Whitney p
  (`r = |Z|/sqrt(N)`). For the E-A rate, E-A-D rate and D-frequency tests Z is
  computed from U without tie correction,
  `Z = (U - n1*n2/2) / sqrt(n1*n2*(N+1)/12)`, `r = |Z|/sqrt(N)`.
- **Windows.** `--windows exact` (primary) uses the 650-sample string sliced
  at samples 50-249 (100-500 ms) and 250-449 (500-900 ms) before collapsing.
  `--windows proportional` (sensitivity) slices the pre/post concatenated
  string by the proportional index rule of the original pipeline (positions in
  a string of about 84 letters, not milliseconds).

## Contributions

Original analysis code: Keith Guzmán-Díaz and Omar Cano-Garcia. This
re-implementation was reviewed, verified against the published values and
audited by Omar Paredes.

## Not included

- Code for the published figures: the code writes base plots and the data
  behind the figures only.
- The script of the October 2025 clustering run: not available; the
  templates are deposited as inputs, and `fit_K5.py` / `fit_all_K.py`
  document the procedure (see Templates).
- The raw EEG: not deposited (available on reasonable request).

## Sex coding

The sequence and metric tables store only the raw file-name prefix (`h` or
`m`) as `file_prefix`. The sex label is attached with `config.SEX_MAP`
(`h` = Male, `m` = Female; author decision of 2026-10-04 based on the raw
folders `Hombres` (h) and `Mujeres` (m)) in the GEE and in the supplementary
tables. `MICROSTATE_SEX_MAP` overrides it. The two possible mappings define
the same partition of participants with the labels swapped, so the Wald tests
and the AIC/QIC differences are the same under both; only the sign of the sex
coefficients changes.

## Changelog

See `CHANGELOG.md`.
