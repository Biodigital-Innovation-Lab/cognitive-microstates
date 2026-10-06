"""
metrics.py -- per-trial microstate metrics (coverage, occurrences, lifespan).

Purpose : from the per-sample labels written by backfit.py or
          import_sequences.py (reference templates, the only template set),
          compute for every
          (subject, condition, trial, state):
            coverage    = fraction of the 650 samples labelled with the state
            occurrences = number of runs (maximal blocks of identical
                          consecutive labels) of the state
            lifespan    = mean run length in samples * 1000 / SFREQ (ms);
                          0 when occurrences = 0
Usage   : python metrics.py
Inputs  : MICROSTATE_OUT_DIR/sequences/reference/microstate_labels_K5.npz
                         (from 03_sequences/backfit.py or import_sequences.py)
Outputs : MICROSTATE_OUT_DIR/tables/microstate_metrics_K5_reference.csv
              subject, file_prefix, condition, trial, state, coverage,
              lifespan, occurrences
          No sex column, no participant initials.
Environment : Python 3.11.14; numpy 1.25.2, pandas 2.0.3; see requirements.txt.
              No random numbers are used.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

MAP_NAMES = ["A", "B", "C", "D", "E"]


def trial_metrics(labels, n_states=len(MAP_NAMES)):
    """labels (n_samples,) ints -> list of (coverage, lifespan_ms, occurrences)."""
    n = len(labels)
    starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    lengths = np.diff(np.r_[starts, n])
    run_states = labels[starts]
    out = []
    for s in range(n_states):
        occ = int((run_states == s).sum())
        cov = float((labels == s).sum()) / n
        life = float(lengths[run_states == s].mean()) * 1000.0 / config.SFREQ if occ else 0.0
        out.append((cov, life, occ))
    return out


def main():
    argparse.ArgumentParser(description=__doc__.split("\n")[1]).parse_args()

    out_dir = config.get_out_dir()
    npz = np.load(out_dir / "sequences" / "reference" / "microstate_labels_K5.npz")
    labels = npz["labels"].astype(int)

    rows = []
    for i in range(labels.shape[0]):
        for s, (cov, life, occ) in enumerate(trial_metrics(labels[i])):
            rows.append((int(npz["subject"][i]), str(npz["file_prefix"][i]),
                         str(npz["condition"][i]), int(npz["trial"][i]),
                         MAP_NAMES[s], cov, life, occ))
    df = pd.DataFrame(rows, columns=["subject", "file_prefix", "condition", "trial",
                                     "state", "coverage", "lifespan", "occurrences"])
    tab_dir = out_dir / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)
    path = tab_dir / "microstate_metrics_K5_reference.csv"
    df.to_csv(path, index=False)
    print(f"{len(df)} rows -> tables/{path.name}")


if __name__ == "__main__":
    main()
