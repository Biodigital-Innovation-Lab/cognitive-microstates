"""
import_sequences.py -- read the deposited per-sample label sequences.

Purpose : entry point of run_all.sh --from-sequences. Reads the deposited
          sequence table (650-letter strings, one per trial; labels from the
          K=5 reference templates, i.e. the K=5 solution of the October 2025
          clustering run) and writes it where 03_sequences/backfit.py writes
          its output, in the same format, so that every downstream script runs
          unchanged.
Usage   : python import_sequences.py
Inputs  : MICROSTATE_SEQUENCES_DIR/microstate_sequences_K5_reference.csv
              subject (1-30), file_prefix (h|m), condition (GO|NG),
              trial (1-20, 1-based), sequence_full (650 letters A-E)
Outputs : MICROSTATE_OUT_DIR/sequences/reference/microstate_sequences_K5.csv
              same columns as the input
          MICROSTATE_OUT_DIR/sequences/reference/microstate_labels_K5.npz
              labels int8 (1200, 650), subject, file_prefix, condition, trial
          No sex column, no participant initials.
Environment : Python 3.11.14; numpy 1.25.2, pandas 2.0.3; see requirements.txt.
              No random numbers are used.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

MAP_NAMES = ["A", "B", "C", "D", "E"]
COLUMNS = ["subject", "file_prefix", "condition", "trial", "sequence_full"]
N_SUBJECTS = 30
N_ROWS = N_SUBJECTS * len(config.CONDITIONS) * config.N_TRIALS   # 1,200


def letters_to_labels(sequence):
    """'ABC...' -> int array 0-4 (A-E)."""
    lookup = {m: i for i, m in enumerate(MAP_NAMES)}
    return np.array([lookup[c] for c in sequence], dtype=np.int8)


def check(df, name):
    """Structure checks on one deposited table."""
    if list(df.columns) != COLUMNS:
        raise ValueError(f"{name}: columns must be {COLUMNS}")
    if len(df) != N_ROWS:
        raise ValueError(f"{name}: expected {N_ROWS} rows, got {len(df)}")
    if df.duplicated(["subject", "condition", "trial"]).any():
        raise ValueError(f"{name}: duplicated (subject, condition, trial)")
    if not set(df["condition"]) == set(config.CONDITIONS):
        raise ValueError(f"{name}: condition must be GO or NG")
    if not set(df["file_prefix"]) <= {"h", "m"}:
        raise ValueError(f"{name}: file_prefix must be h or m")
    if not df["sequence_full"].str.fullmatch(f"[{''.join(MAP_NAMES)}]{{{config.N_SAMPLES}}}").all():
        raise ValueError(f"{name}: sequence_full must be {config.N_SAMPLES} letters A-E")


def main():
    in_dir = config.get_sequences_dir()
    out_dir = config.get_out_dir()
    for name in ("reference",):
        src = in_dir / f"microstate_sequences_K5_{name}.csv"
        if not src.is_file():
            raise FileNotFoundError(src)
        df = pd.read_csv(src, dtype={"file_prefix": str, "condition": str,
                                     "sequence_full": str})
        check(df, src.name)

        seq_dir = out_dir / "sequences" / name
        seq_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(seq_dir / "microstate_sequences_K5.csv", index=False)
        labels = np.stack([letters_to_labels(s) for s in df["sequence_full"]])
        np.savez(
            seq_dir / "microstate_labels_K5.npz",
            labels=labels.astype(np.int8),
            subject=df["subject"].to_numpy(dtype=np.int16),
            file_prefix=df["file_prefix"].to_numpy(dtype="<U1"),
            condition=df["condition"].to_numpy(dtype="<U2"),
            trial=df["trial"].to_numpy(dtype=np.int16),
        )
        counts = np.bincount(labels.ravel(), minlength=len(MAP_NAMES))
        print(f"{name}: {len(df)} trials; label frequencies (samples):",
              {m: int(c) for m, c in zip(MAP_NAMES, counts)})
    print("Saved to sequences/ in MICROSTATE_OUT_DIR")


if __name__ == "__main__":
    main()
