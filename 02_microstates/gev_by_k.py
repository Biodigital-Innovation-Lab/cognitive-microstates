"""
gev_by_k.py -- global explained variance of fixed microstate templates.

Purpose : back-fit FIXED templates (no clustering) to the preprocessed EEG and
          compute the GEV with the same preprocessing and definition as
          fit_K5.py and fit_all_K.py (microstate_utils.preprocess: average
          reference, 4-30 Hz band-pass on the concatenated trials;
          microstate_utils.compute_gev: squared GFP times the squared largest
          absolute correlation with the templates, over all samples).
          run_all.sh --from-eeg calls it with the six deposited group
          templates (October 2025 run, K = 4..9) and --out all_K_gev.csv
          (Figure 2A data; equals the deposited all_K_gev.csv).
Usage   : python gev_by_k.py [--templates FILE [FILE ...]] [--out NAME]
Inputs  : MICROSTATE_OUT_DIR/eeg_trials.npz (from 01_reshape/reshape.py);
          --templates  one .npz per K with a 'templates' array, (K, 19) or
                       (19, K); K is taken from the array shape. Default: the
                       file in MICROSTATE_REFERENCE_TEMPLATES (K=5). If a file
                       stores 'ch_names' (readable without pickle), they must
                       equal the channel order of eeg_trials.npz; otherwise
                       the file's channel order is used as stored.
Outputs : MICROSTATE_OUT_DIR/<NAME> (default gev_by_k.csv): K, GEV (fraction,
          8 decimals), one row per template file, sorted by K.
Environment : Python 3.11.14; mne 1.3.1; see requirements.txt. No random
              numbers are used.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import microstate_utils as mu  # noqa: E402


def load_templates(path, ch_names):
    """Templates (n_channels, K) from an .npz file, loaded without pickle."""
    npz = np.load(path, allow_pickle=False)
    templates = npz["templates"].astype(np.float64)
    n_ch = len(ch_names)
    if templates.shape[0] != n_ch:
        templates = templates.T
    if templates.shape[0] != n_ch:
        raise ValueError(f"{Path(path).name}: no axis of length {n_ch}")
    try:
        stored = npz["ch_names"].tolist() if "ch_names" in npz.files else None
    except ValueError:                       # stored as a pickled object array
        stored = None
    if stored is None:
        print(f"  {Path(path).name}: ch_names not readable; stored channel order used")
    elif stored != ch_names:
        raise ValueError(f"{Path(path).name}: ch_names differ from eeg_trials.npz")
    return templates


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--templates", nargs="+", default=None)
    parser.add_argument("--out", default="gev_by_k.csv")
    args = parser.parse_args()

    files = args.templates
    if files is None:
        ref = config.get_reference_templates()
        if ref is None:
            raise EnvironmentError(
                "Give --templates or set MICROSTATE_REFERENCE_TEMPLATES.")
        files = [ref]

    out_dir = config.get_out_dir()
    data, ch_names, sfreq = mu.load_trials(out_dir / "eeg_trials.npz")
    print(f"data shape: {data.shape}; sfreq: {sfreq} Hz")
    info = mu.make_info(ch_names, sfreq)
    data_prep = mu.preprocess(data, info)

    gev_per_k = {}
    for path in files:
        templates = load_templates(path, ch_names)
        K = templates.shape[1]
        if K in gev_per_k:
            raise ValueError(f"two template files with K={K}")
        gev_per_k[K] = mu.compute_gev(data_prep, templates)
        print(f"  K={K}  {Path(path).name}  GEV={gev_per_k[K] * 100:.4f}%")

    with open(out_dir / args.out, "w") as fh:
        fh.write("K,GEV\n")
        for K in sorted(gev_per_k):
            fh.write(f"{K},{gev_per_k[K]:.8f}\n")


if __name__ == "__main__":
    main()
