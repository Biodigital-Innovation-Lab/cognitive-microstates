"""
trial_table.py -- per-trial sequence table (full, collapsed, early and late).

Purpose : build the per-trial table of microstate sequences from the 650-sample
          backfit strings. Definitions:
            collapse        remove consecutive repeated letters
            early_full      sequence_full[50:250]   (100-500 ms)
            late_full       sequence_full[250:450]  (500-900 ms)
            early/late_collapsed   collapse applied after slicing
            *_length        len() of the corresponding string
          Column order follows the deposited per-trial sequence table, with
          file_prefix (h|m) in place of a sex column.
Usage   : python trial_table.py
Inputs  : MICROSTATE_OUT_DIR/sequences/reference/microstate_sequences_K5.csv
              (from 03_sequences/backfit.py --templates reference; it already
              contains file_prefix)
Outputs : MICROSTATE_OUT_DIR/tables/S5_trial_sequences_reference.csv
              subject, file_prefix, condition (GO|NG), trial (1-based),
              sequence_full, sequence_length, sequence_collapsed,
              sequence_collapsed_length, early_full, early_collapsed,
              late_full, late_collapsed
Environment : Python 3.11.14; pandas 2.0.3; see requirements.txt.
              No random numbers are used.
"""

import sys
from itertools import groupby
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

EARLY = (50, 250)
LATE = (250, 450)


def collapse(sequence):
    """Remove consecutive repeated letters."""
    return "".join(key for key, _ in groupby(sequence))


def main():
    out_dir = config.get_out_dir()
    src = out_dir / "sequences" / "reference" / "microstate_sequences_K5.csv"
    if not src.is_file():
        raise FileNotFoundError(
            f"{src} not found; run 03_sequences/backfit.py --templates reference")
    df = pd.read_csv(src, dtype={"file_prefix": str, "condition": str})

    full = df["sequence_full"]
    out = pd.DataFrame({
        "subject": df["subject"],
        "file_prefix": df["file_prefix"],
        "condition": df["condition"],
        "trial": df["trial"],
        "sequence_full": full,
        "sequence_length": full.str.len(),
    })
    out["sequence_collapsed"] = full.map(collapse)
    out["sequence_collapsed_length"] = out["sequence_collapsed"].str.len()
    out["early_full"] = full.str[EARLY[0]:EARLY[1]]
    out["early_collapsed"] = out["early_full"].map(collapse)
    out["late_full"] = full.str[LATE[0]:LATE[1]]
    out["late_collapsed"] = out["late_full"].map(collapse)

    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(tables_dir / "S5_trial_sequences_reference.csv", index=False)
    print(f"rows: {len(out)}; saved to tables/S5_trial_sequences_reference.csv "
          "in MICROSTATE_OUT_DIR")


if __name__ == "__main__":
    main()
