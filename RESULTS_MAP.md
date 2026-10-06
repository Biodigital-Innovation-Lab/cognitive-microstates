# Results map

Manuscript item -> script -> output file (under `$MICROSTATE_OUT_DIR`).
Exact windows unless stated. All items use the October 2025 templates (see
`README.md`, Templates). The published figures were edited after generation;
the files below hold the values behind them.

| manuscript item | script | output file |
|---|---|---|
| Figure 2A: GEV for K = 4-9 (K selection) | `02_microstates/gev_by_k.py` (`--from-eeg`) | `all_K_gev.csv` (equals `inputs/all_K_gev.csv` of the data deposit) |
| Figure 2B: within-solution \|r\|, K = 5 and K = 6 | `02_microstates/k_template_correlations.py` | `template_correlations/K5_abs_r.csv`, `template_correlations/K6_abs_r.csv` |
| Supplementary Figure S2: T4 value of each K = 6 map | `02_microstates/k_template_correlations.py` | `template_correlations/K6_T4.csv` |
| Supplementary Figure S3: within-solution \|r\|, K = 4-9 | `02_microstates/k_template_correlations.py` | `template_correlations/K{4..9}_abs_r.csv` |
| GEV of the K = 5 solution (text) | `02_microstates/gev_by_k.py` | `all_K_gev.csv`, row K = 5 |
| Table 1 and Data S1: canonical correspondence | `02_microstates/canonical_similarity.py` (needs the third-party maps) | `canonical/*.csv` |
| Table 2: GEE, sex and condition (coverage, lifespan, occurrences) | `04_statistics/gee_sex_condition.py --source eeg` | `tables/gee_sex_condition_results.csv`, `.txt` |
| Supplementary Table S1: metrics per trial and state | `03_sequences/metrics.py`, `06_tables/assemble_supplementary.py` | `tables/microstate_metrics_K5_reference.csv`; `supplementary/S1_Microstate_Metrics.xlsx` (+ CSV) |
| Supplementary Table S2: k-mer screen | `04_statistics/motif_analysis.py --windows exact` | `tables/S2_kmer_*_exact.csv`; `supplementary/S2_Collapsed_Sequences.xlsx` (+ CSVs) |
| Supplementary Table S3: bigram rates (with BH column `p (FDR)`) | `04_statistics/motif_analysis.py --windows exact` | `tables/S3_bigram_rates_exact.csv`; `supplementary/S3_Bigram_Transition_Rates.xlsx` (+ CSVs) |
| Supplementary Table S4: conditional successors | `04_statistics/motif_analysis.py --windows exact` | `tables/S4_conditional_successors_exact.csv`; `supplementary/S4_Conditional_Successor_Statistics.xlsx` (+ CSVs) |
| Table 3: significant conditional successors, early window | `04_statistics/motif_analysis.py` | `tables/Table3_significant_conditional_early_exact.csv` (`_proportional` with `--sensitivity`) |
| E-A-X Fisher tests | `04_statistics/motif_analysis.py` | `tables/EAX_fisher_exact.csv` |
| E-A rate, E-A-D rate and D-frequency tests | `04_statistics/motif_analysis.py` | `tables/EA_EAD_Dfreq_tests_exact.csv` |
| Split-half reliability (NoGo) | `04_statistics/split_half.py` | `tables/SplitHalf_Motif_Reliability_NoGo_exact.csv` |
| Supplementary Table S5: sequences by trial and consensus by subject | `03_sequences/trial_table.py`, `05_alignment/consensus.py`, `06_tables/assemble_supplementary.py` | `tables/S5_trial_sequences_reference.csv`, `tables/msa_consensus_SUBJECT.csv`; `supplementary/S5_MSA_Alignment_Scores.xlsx` (+ CSVs) |
| Figure 3: within-condition Needleman-Wunsch alignment (z, OR) | `05_alignment/within_condition_nw_analysis.py --source eeg` | `tables/within_condition_summary.csv`, `figures/fig_wc_*`, `figures/fig_or_forest_within_condition.*` |

Not produced by this code: the GEE coefficients plotted in Supplementary
Figure S4 (the GEE output holds the Wald tests and AIC/QIC), the global and
meta consensus of Supplementary Figure S1, and the published figure files.
