"""
split_half.py -- split-half reliability of the NoGo early-window motif rates.

Purpose : for the motifs ACD, ABE and EAD, split each participant's 20 NoGo
          trials into trials 1-10 and 11-20, take the mean motif rate of each
          half per participant, and correlate the halves across participants
          (Pearson r, Spearman-Brown). The counting, correlation and
          Spearman-Brown logic is copied from the original
          splithalf_motif_reliability.py without change. Only the source of
          early_collapsed differs: it is derived from the 650-sample
          sequences with exact windows (sequence_full[50:250], then consecutive
          repeats removed), as in S5 sheet "Microstate Sequences by Trial".
Usage   : python split_half.py
Inputs  : MICROSTATE_OUT_DIR/sequences/reference/microstate_sequences_K5.csv
          (from 03_sequences/backfit.py, reference templates)
Outputs : MICROSTATE_OUT_DIR/tables/SplitHalf_Motif_Reliability_NoGo_exact.csv
              motif, r, p, spearman_brown_rho, h1_mean, h1_sd, h2_mean,
              h2_sd, n_subjects
          No participant identifiers are written.
Environment : Python 3.11.14; numpy 1.25.2, pandas 2.0.3, scipy 1.11.4; see
              requirements.txt. No random numbers are used.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from motif_analysis import load_trials  # noqa: E402

MOTIFS = ["ACD", "ABE", "EAD"]
CONDITION_NOGO = "NG"
N_SUBJECTS_EXPECTED = 30
N_TRIALS_EXPECTED = 20


def motif_rate(seq: str, motif: str) -> float:
    """
    Number of occurrences of motif in seq (overlapping starts counted)
    divided by the number of possible start positions (len(seq) - len(motif)
    + 1). Returns 0.0 if seq is not a string or shorter than motif.
    """
    if not isinstance(seq, str) or len(seq) < len(motif):
        return 0.0
    count = sum(
        1 for i in range(len(seq) - len(motif) + 1)
        if seq[i: i + len(motif)] == motif
    )
    denom = len(seq) - len(motif) + 1
    return count / denom if denom > 0 else 0.0


def spearman_brown(r: float) -> float:
    """Full-test reliability estimate via the Spearman-Brown formula."""
    return (2 * r) / (1 + r)


def splithalf_analysis(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """One row per motif; halves are trials 1-10 and 11-20."""
    subset = df[df["condition"] == condition].copy()

    subjects = sorted(subset["subject"].unique())
    assert len(subjects) == N_SUBJECTS_EXPECTED, (
        f"Expected {N_SUBJECTS_EXPECTED} subjects, found {len(subjects)}"
    )

    records = []
    for motif in MOTIFS:
        h1_means, h2_means = [], []

        for subj in subjects:
            subj_df = subset[subset["subject"] == subj].sort_values("trial")

            h1_rows = subj_df[subj_df["trial"] <= 10]
            h2_rows = subj_df[subj_df["trial"] > 10]

            h1_rates = [motif_rate(row["early_collapsed"], motif)
                        for _, row in h1_rows.iterrows()]
            h2_rates = [motif_rate(row["early_collapsed"], motif)
                        for _, row in h2_rows.iterrows()]

            h1_means.append(np.mean(h1_rates))
            h2_means.append(np.mean(h2_rates))

        h1_arr = np.array(h1_means)
        h2_arr = np.array(h2_means)

        r, p = stats.pearsonr(h1_arr, h2_arr)
        rho = spearman_brown(r)

        records.append({
            "motif": motif,
            "r": round(r, 3),
            "p": round(p, 4),
            "spearman_brown_rho": round(rho, 3),
            "h1_mean": round(h1_arr.mean(), 4),
            "h1_sd": round(h1_arr.std(), 4),
            "h2_mean": round(h2_arr.mean(), 4),
            "h2_sd": round(h2_arr.std(), 4),
            "n_subjects": len(subjects),
        })

    return pd.DataFrame(records)


def main():
    df = load_trials("exact").rename(columns={"early": "early_collapsed"})
    assert df.groupby(["subject", "condition"]).size().eq(N_TRIALS_EXPECTED).all()

    print("NoGo, early window (exact: sequence_full[50:250], then collapsed)")
    results = splithalf_analysis(df, CONDITION_NOGO)
    print(results.to_string(index=False))

    tab_dir = config.get_out_dir() / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(tab_dir / "SplitHalf_Motif_Reliability_NoGo_exact.csv",
                   index=False)
    print("Saved to tables/ in MICROSTATE_OUT_DIR")


if __name__ == "__main__":
    main()
