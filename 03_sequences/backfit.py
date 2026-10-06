"""
backfit.py -- per-sample microstate labels (backfitting) for every trial.

Purpose : assign each of the 650 samples of each trial to the K=5 template
          with the highest absolute spatial correlation (argmax). Same logic as
          the original Secuencias_K5 notebook: unfiltered signal, per-timepoint
          spatial mean-centering, unit-norm vectors (+1e-12 in the
          denominator), argmax of |correlation|. Labels 0-4 -> A-E.
          Templates: the deposited K=5 reference templates (the K=5 solution
          of the October 2025 clustering run), the only template set used.
          No polarity flip is applied (argmax of the absolute correlation is
          polarity-invariant).
Usage   : python backfit.py
Inputs  : MICROSTATE_OUT_DIR/eeg_trials.npz        (from 01_reshape/reshape.py)
          MICROSTATE_REFERENCE_TEMPLATES (.npz, key 'templates'; loaded with
          allow_pickle=False)
Outputs : MICROSTATE_OUT_DIR/sequences/reference/microstate_sequences_K5.csv
              subject (1-30), file_prefix (h|m), condition (GO|NG),
              trial (1-20, 1-based), sequence_full (650 letters A-E)
          MICROSTATE_OUT_DIR/sequences/reference/microstate_labels_K5.npz
              labels int8 (1200, 650), subject, file_prefix, condition, trial
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


def load_templates(path_npz):
    """Templates as (n_channels, n_clusters)."""
    templates = np.load(path_npz, allow_pickle=False)["templates"]
    if templates.shape[0] == 5:
        templates = templates.T
    return templates


def backfit_epoch(epoch_data, templates):
    """
    epoch_data (n_channels, n_samples), templates (n_channels, n_clusters).
    Returns integer labels (n_samples,) = argmax of the absolute correlation.
    """
    epoch_centered = epoch_data - epoch_data.mean(axis=0, keepdims=True)
    templates_centered = templates - templates.mean(axis=0, keepdims=True)
    epoch_norm = epoch_centered / (
        np.linalg.norm(epoch_centered, axis=0, keepdims=True) + 1e-12)
    templates_norm = templates_centered / (
        np.linalg.norm(templates_centered, axis=0, keepdims=True) + 1e-12)
    correlations = templates_norm.T @ epoch_norm            # (n_clusters, n_samples)
    return np.argmax(np.abs(correlations), axis=0).astype(int)


def labels_to_letters(labels):
    return "".join(MAP_NAMES[i] for i in labels)


def main():
    argparse.ArgumentParser(description=__doc__.split("\n")[1]).parse_args()

    out_dir = config.get_out_dir()
    seq_dir = out_dir / "sequences" / "reference"
    seq_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(out_dir / "eeg_trials.npz")
    data = npz["data"].astype(np.float64)                   # (1200, 19, 650)
    ref_path = config.get_reference_templates()
    if ref_path is None:
        raise EnvironmentError("Set MICROSTATE_REFERENCE_TEMPLATES.")
    templates = load_templates(ref_path)
    print(f"templates: reference; data {data.shape}, templates {templates.shape}")

    labels = np.stack([backfit_epoch(trial, templates) for trial in data])
    assert labels.shape == (1200, 650)

    df = pd.DataFrame({
        "subject": npz["subject"].astype(int),
        "file_prefix": npz["file_prefix"].astype(str),
        "condition": npz["condition"].astype(str),
        "trial": npz["trial"].astype(int),
        "sequence_full": [labels_to_letters(row) for row in labels],
    })
    df.to_csv(seq_dir / "microstate_sequences_K5.csv", index=False)
    np.savez(
        seq_dir / "microstate_labels_K5.npz",
        labels=labels.astype(np.int8),
        subject=npz["subject"], file_prefix=npz["file_prefix"],
        condition=npz["condition"], trial=npz["trial"],
    )

    counts = np.bincount(labels.ravel(), minlength=len(MAP_NAMES))
    print("label frequencies (samples):",
          {m: int(c) for m, c in zip(MAP_NAMES, counts)})
    print("Saved to sequences/reference/ in MICROSTATE_OUT_DIR")


if __name__ == "__main__":
    main()
