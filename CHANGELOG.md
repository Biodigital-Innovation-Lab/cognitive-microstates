# Changelog

## 1.0.0

First public release. Code and outputs are unchanged from 0.2.0; only the
version number changes. Added: `expected_outputs/OUTPUTS.sha256` (SHA-256 of
the reference outputs) and a "Verify your run" section in `README.md`; the
Mendeley Data DOI (doi:10.17632/6jcnrr9v58.2, Version 2) and the repository
URL in `README.md` and `CITATION.cff`; the author list in `CITATION.cff`. The Zenodo DOI of the
software is added after this release. Contributions section in `README.md`.

## 0.2.0

- One template set for every result: the group templates of the October 2025
  clustering run, K = 4-9, deposited in Mendeley Data (K = 5 = the reference
  templates). The templates fitted by `fit_K5.py` / `fit_all_K.py` (and the
  label sequences derived from them) no longer support any reported result.
- `run_all.sh`: `fit_K5.py` and `fit_all_K.py` are no longer run (they stay as
  documentation of the clustering procedure); `--from-eeg` back-fits the
  reference templates and computes the GEV of the deposited K = 4-9 templates
  (`02_microstates/gev_by_k.py` -> `all_K_gev.csv`); `--from-sequences`
  imports the reference sequences only; step logs show path arguments by file
  name only.
- New common step `02_microstates/k_template_correlations.py`: |r| matrix of
  each K = 4-9 solution and the T4 value of each K = 6 map
  (`template_correlations/`).
- New environment variable `MICROSTATE_TEMPLATES_DIR` (folder with
  `microstates_K{4..9}_templates.npz`).
- The `pipeline` label-sequence set is removed: `backfit.py` and `metrics.py`
  have no `--templates` option; `import_sequences.py` reads the reference set
  only. S1 metrics and the GEE use the reference labels
  (`tables/microstate_metrics_K5_reference.csv`).
- Supplementary Table S3: new column `p (FDR)` after `p` in every sheet
  (Benjamini-Hochberg over the 20 bigrams of the sheet; already computed in
  `tables/S3_bigram_rates_exact.csv`).
- Data deposit: `microstates_K{4..9}_templates.npz` replaced by the October
  2025 templates (keys `templates` (K, 19), `ch_names`, `map_names`);
  `all_K_gev.csv` = GEV of these templates; `microstate_sequences_K5_pipeline.csv`
  removed.

## 0.1.0

First release. Entry points `--from-eeg` and `--from-sequences` (new
`03_sequences/import_sequences.py`); `--from-tables` removed (the deposited
metric, consensus and FULL tables are no longer inputs; the scripts keep their
`--source deposited` option, which `run_all.sh` does not call); exact windows
are the primary analysis, proportional windows only with `--sensitivity`; sex
coding `h` = Male, `m` = Female; labels sex Male/Female, condition GO/NG,
window full/early/late in all outputs; reference templates deposited as a
clean `.npz`; new `06_tables/assemble_supplementary.py`; uv environment.
