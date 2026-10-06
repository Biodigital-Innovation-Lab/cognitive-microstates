"""
reshape.py -- raw EEG .txt files -> one structured array file.

Purpose : read the 60 raw .txt files (30 subjects x GO/NG), reshape each into
          (20 trials, 19 channels, 650 samples) and stack them into one file.
          Same logic as the original load_reshape notebook.
Inputs  : *.txt files in MICROSTATE_DATA_DIR. Each file: 13,000 rows x 19
          tab-separated columns (20 trials x 650 samples; 500 Hz).
          Filename pattern: {h|m}{NN}_{participant code}_{GO|NG}.txt
          Only the prefix, the subject number and the condition are kept; the
          participant code part of the name is never parsed or stored.
Outputs : MICROSTATE_OUT_DIR/eeg_trials.npz
            data        float32 (1200, 19, 650)
            subject     int16   (1200,)   1-30
            file_prefix str     (1200,)   raw filename prefix, 'h' or 'm'
            condition   str     (1200,)   'GO' or 'NG'
            trial       int16   (1200,)   1-20 within subject x condition
            ch_names    str     (19,)
            sfreq       float64 scalar
          No sex column is created (coding unresolved; see config.SEX_MAP).
Environment : Python 3.11.14; see requirements.txt.
"""

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

CH_NAMES = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6",
    "Fz", "Cz", "Pz",
]

# The participant-code part of the filename is matched but not captured.
FILENAME_PATTERN = re.compile(r"([hm])(\d+)_[A-Z]+_(GO|NG)\.txt", re.IGNORECASE)


def parse_filename(fname: str) -> dict:
    """Return file_prefix, subject and condition from a raw filename."""
    match = FILENAME_PATTERN.match(fname)
    if not match:
        raise ValueError("A filename does not match the expected pattern")
    return {
        "file_prefix": match.group(1).lower(),
        "subject": int(match.group(2)),
        "condition": match.group(3).upper(),
    }


def load_txt(filepath: Path) -> np.ndarray:
    """Load one file and reshape to (n_trials, n_channels, n_samples)."""
    raw = np.loadtxt(filepath, delimiter="\t")  # (13000, 19)
    expected = (config.N_TRIALS * config.N_SAMPLES, config.N_CHANNELS)
    if raw.shape != expected:
        raise ValueError(f"Expected shape {expected}, got {raw.shape}")
    trials = raw.reshape(config.N_TRIALS, config.N_SAMPLES, config.N_CHANNELS)
    return trials.transpose(0, 2, 1).astype(np.float32)


def main():
    data_dir = config.get_data_dir()
    out_dir = config.get_out_dir()
    txt_files = sorted(data_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError("No .txt files found in MICROSTATE_DATA_DIR")
    print(f"Found {len(txt_files)} files")

    all_data, subjects, prefixes, conditions, trials = [], [], [], [], []
    for fpath in txt_files:
        meta = parse_filename(fpath.name)
        arr = load_txt(fpath)
        n = arr.shape[0]
        all_data.append(arr)
        subjects.extend([meta["subject"]] * n)
        prefixes.extend([meta["file_prefix"]] * n)
        conditions.extend([meta["condition"]] * n)
        trials.extend(range(1, n + 1))  # 1-based within subject x condition
        print(f"  subject={meta['subject']:02d} prefix={meta['file_prefix']} "
              f"condition={meta['condition']}")

    data = np.concatenate(all_data, axis=0)
    subject = np.array(subjects, dtype=np.int16)
    file_prefix = np.array(prefixes, dtype=str)
    condition = np.array(conditions, dtype=str)
    trial = np.array(trials, dtype=np.int16)

    # ── Checks ───────────────────────────────────────────────────────────
    assert data.shape == (1200, 19, 650), f"Unexpected shape {data.shape}"
    assert sorted(np.unique(subject).tolist()) == list(range(1, 31))
    for s in range(1, 31):
        for c in config.CONDITIONS:
            sel = (subject == s) & (condition == c)
            assert sel.sum() == config.N_TRIALS, f"subject {s} {c}: {sel.sum()}"
            assert sorted(trial[sel].tolist()) == list(range(1, 21))
        assert len(np.unique(file_prefix[subject == s])) == 1
    print(f"\nShape {data.shape}; 30 subjects x (20 GO + 20 NG) trials: OK")
    print(f"file_prefix counts (subjects): "
          f"{ {p: int(len(np.unique(subject[file_prefix == p]))) for p in np.unique(file_prefix)} }")

    out_file = out_dir / "eeg_trials.npz"
    np.savez(
        out_file,
        data=data,
        subject=subject,
        file_prefix=file_prefix,
        condition=condition,
        trial=trial,
        ch_names=np.array(CH_NAMES, dtype=str),
        sfreq=np.float64(config.SFREQ),
    )
    print(f"Saved {out_file.name} in MICROSTATE_OUT_DIR")


if __name__ == "__main__":
    main()
