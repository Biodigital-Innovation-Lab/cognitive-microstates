"""
full_prepost.py -- per-trial table with the pre/post concatenated sequence.

Purpose : build the per-trial table read by 04_statistics/motif_analysis.py
          --windows proportional --source eeg. The table has the 650-sample
          backfit string collapsed in two parts and the two parts joined:
            sequence_full       collapse(sf[0:150]) + collapse(sf[150:650])
                                (plain concatenation; when the last letter of
                                the first part equals the first letter of the
                                second part, the repeat is kept)
            sequence_collapsed  collapse(sf), the whole 650-sample string
            *_length            len() of the corresponding string
          where sf is sequence_full of the per-trial sequence table and
          collapse removes consecutive repeated letters. The boundary at
          sample 150 is the pre-stimulus / post-stimulus limit of the
          recording. "sequence_full" here is therefore NOT the 650-sample
          string; it is a short string (about 84 letters on average).
          Re-implemented from an earlier script and verified against the
          deposited table.
Usage   : python full_prepost.py
Inputs  : MICROSTATE_OUT_DIR/tables/S5_trial_sequences_reference.csv
              (from 03_sequences/trial_table.py; columns used: subject,
              file_prefix, condition, trial, sequence_full)
Outputs : MICROSTATE_OUT_DIR/tables/microstate_sequences_FULL_prepost.csv
              subject, file_prefix (h|m), condition (GO|NG), trial,
              sequence_full, sequence_length, sequence_collapsed,
              sequence_collapsed_length
          No sex column, no participant initials.
Environment : Python 3.11.14; pandas 2.0.3; see requirements.txt.
              No random numbers are used.
"""

import sys
from itertools import groupby
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

PRE_SAMPLES = 150     # samples before the stimulus; the rest is post


def collapse(sequence):
    """Remove consecutive repeated letters."""
    return "".join(key for key, _ in groupby(sequence))


def main():
    tab_dir = config.get_out_dir() / "tables"
    src = tab_dir / "S5_trial_sequences_reference.csv"
    if not src.is_file():
        raise FileNotFoundError(
            f"{src} not found; run 03_sequences/trial_table.py")
    df = pd.read_csv(src, usecols=["subject", "file_prefix", "condition",
                                   "trial", "sequence_full"],
                     dtype={"file_prefix": str, "condition": str})
    assert df["sequence_full"].str.len().eq(config.N_SAMPLES).all()

    sf = df["sequence_full"]
    out = df[["subject", "file_prefix", "condition", "trial"]].copy()
    out["sequence_full"] = [collapse(s[:PRE_SAMPLES]) + collapse(s[PRE_SAMPLES:])
                            for s in sf]
    out["sequence_length"] = out["sequence_full"].str.len()
    out["sequence_collapsed"] = sf.map(collapse)
    out["sequence_collapsed_length"] = out["sequence_collapsed"].str.len()

    out.to_csv(tab_dir / "microstate_sequences_FULL_prepost.csv", index=False)
    print(f"rows: {len(out)}; saved to tables/microstate_sequences_FULL_prepost.csv "
          "in MICROSTATE_OUT_DIR")


if __name__ == "__main__":
    main()
