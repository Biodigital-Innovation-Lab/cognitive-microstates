# Microstate sequence analysis (Go/NoGo EEG)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23177216.svg)](https://doi.org/10.5281/zenodo.23177216)

Analysis code for "Sequential Syntax of EEG Microstates Captures the Temporal
Dynamics of Inhibitory Control" (iScience, 2026): back-fitting of K = 5
microstate templates (A-E), microstate metrics, motif and transition
analyses, consensus sequences, Needleman-Wunsch alignment, GEE models and
Supplementary Tables S1-S5 (30 participants x 2 conditions x 20 trials;
19 channels, 500 Hz).

The code is a re-implementation of the original analysis, verified against
the published values.

- Code DOI (version used in the paper): [10.5281/zenodo.23177217](https://doi.org/10.5281/zenodo.23177217)
- Data: Mendeley Data, [10.17632/6jcnrr9v58.2](https://doi.org/10.17632/6jcnrr9v58.2)
- How to cite: `CITATION.cff`
- Licence: MIT (code); the deposited data are CC BY 4.0

## Installation

Python 3.11. Tested on macOS (Apple silicon) with the versions pinned in
`uv.lock`.

```
uv sync --frozen && source .venv/bin/activate
```

or, with pip:

```
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Data

Download the Mendeley Data deposit. Its `inputs/` folder contains the label
sequences, the group templates (K = 4-9) and the GEV per K; `supplementary/`
contains Supplementary Tables S1-S5. The raw EEG is available from the Lead
Contact upon reasonable request.

## Run

Locations are set with environment variables (see `config.py`):

| variable | content |
|---|---|
| `MICROSTATE_SEQUENCES_DIR` | the `inputs/` folder of the deposit |
| `MICROSTATE_TEMPLATES_DIR` | the `inputs/` folder of the deposit |
| `MICROSTATE_REFERENCE_TEMPLATES` | `inputs/microstates_K5_templates_reference.npz` |
| `MICROSTATE_DATA_DIR` | raw EEG files (only for `--from-eeg`) |
| `MICROSTATE_REFERENCE_MAPS_DIR` | third-party canonical maps (optional, see below) |
| `MICROSTATE_OUT_DIR` | output folder (required by `run_all.sh`; use an empty folder per run) |

```
./run_all.sh --from-sequences    # from the deposited label sequences
./run_all.sh --from-eeg          # from the raw EEG
```

Add `--sensitivity` to also run the proportional-window variant of the motif
analysis. Each step writes a log to `$MICROSTATE_OUT_DIR/logs/`.
`RESULTS_MAP.md` lists the script and output file of each result in the
paper.

## Templates

All results use the group templates deposited in Mendeley Data, which are
inputs to the pipeline; no clustering is run. `fit_K5.py` and `fit_all_K.py`
implement the clustering procedure described in the paper but are not called
by `run_all.sh`: re-running them yields maps that are very similar but not
identical to the deposited ones (|r| > 0.99 for every K = 5 map), and back-fitting
with them changes some sequence-level results. Use the deposited templates to
reproduce the published values.

## Verify your run

`expected_outputs/OUTPUTS.sha256` lists the SHA-256 of the reference outputs.
From the repository folder, after a run:

```
CODE_DIR="$PWD"; cd "$MICROSTATE_OUT_DIR"
shasum -a 256 -c "$CODE_DIR/expected_outputs/OUTPUTS.sha256"   # macOS
sha256sum -c "$CODE_DIR/expected_outputs/OUTPUTS.sha256"       # Linux
```

Expected missing files: `all_K_gev.csv` with `--from-sequences` (it needs the
EEG; it equals the deposited file), and the 14 files in `canonical/` when the
third-party maps are not provided.

## Third-party canonical maps

`02_microstates/canonical_similarity.py` compares the K = 5 templates with
published canonical maps (Koenig et al., 2002; Milz et al., 2016; Artoni et
al., 2023). These maps are not distributed here; the expected file names are
listed in the script's docstring.

## Contributions

Original analysis code: Keith Guzmán-Díaz and Omar Cano-Garcia. This
re-implementation was reviewed, verified against the published values and
audited by Omar Paredes.
