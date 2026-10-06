"""
k_template_correlations.py -- within-solution |r| of the group templates, K = 4..9.

Purpose : for each deposited K = 4..9 solution (October 2025 run), the absolute
          Pearson correlation between every pair of template maps, with the
          same definition as fit_all_K.save_correlation
          (abs(scipy.stats.pearsonr) between the 19-channel maps), and the T4
          value of each K = 6 map as stored. No EEG is needed.
Usage   : python k_template_correlations.py
Inputs  : MICROSTATE_TEMPLATES_DIR/microstates_K{4..9}_templates.npz
              templates (K, 19) float64, ch_names (19,), map_names (K,);
              loaded with allow_pickle=False
Outputs : MICROSTATE_OUT_DIR/template_correlations/K{K}_abs_r.csv  (K = 4..9)
              K x K matrix; first column "map", one column per map; labels from
              map_names; 6 decimals (Figure 2B: K = 5, 6; Supplementary
              Figure S3: K = 4..9)
          MICROSTATE_OUT_DIR/template_correlations/K6_T4.csv
              map, T4: template value at channel T4 for each K = 6 map, sign as
              stored, 6 decimals (Supplementary Figure S2)
Environment : Python 3.11.14; numpy 1.25.2, pandas 2.0.3, scipy 1.11.4; see
              requirements.txt. No random numbers are used.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

K_RANGE = range(4, 10)   # K = 4, 5, 6, 7, 8, 9
T4_K = 6
T4_CHANNEL = "T4"


def load(path):
    """templates (K, 19), ch_names, map_names from a clean template file."""
    npz = np.load(path, allow_pickle=False)
    templates = npz["templates"].astype(np.float64)
    ch_names = npz["ch_names"].tolist()
    map_names = npz["map_names"].tolist()
    if templates.shape != (len(map_names), len(ch_names)):
        raise ValueError(f"{Path(path).name}: templates {templates.shape} do not "
                         f"match {len(map_names)} maps x {len(ch_names)} channels")
    return templates, ch_names, map_names


def abs_r_matrix(templates):
    """|r| between template maps (rows of templates)."""
    K = templates.shape[0]
    return np.array([[abs(pearsonr(templates[i], templates[j])[0])
                      for j in range(K)] for i in range(K)])


def main():
    argparse.ArgumentParser(description=__doc__.split("\n")[1]).parse_args()
    in_dir = config.get_templates_dir()
    out_dir = config.get_out_dir() / "template_correlations"
    out_dir.mkdir(parents=True, exist_ok=True)

    ch_ref = None
    for K in K_RANGE:
        templates, ch_names, map_names = load(in_dir / f"microstates_K{K}_templates.npz")
        if ch_ref is None:
            ch_ref = ch_names
        elif ch_names != ch_ref:
            raise ValueError(f"K={K}: channel order differs from K={K_RANGE[0]}")
        corr = abs_r_matrix(templates)
        df = pd.DataFrame(np.round(corr, 6), columns=map_names)
        df.insert(0, "map", map_names)
        df.to_csv(out_dir / f"K{K}_abs_r.csv", index=False)
        lower = [f"{corr[i, j]:.2f}" for i in range(K) for j in range(i)]
        print(f"K={K}: maps {''.join(map_names)}; lower triangle {' '.join(lower)}")

        if K == T4_K:
            idx = ch_names.index(T4_CHANNEL)
            t4 = pd.DataFrame({"map": map_names,
                               T4_CHANNEL: np.round(templates[:, idx], 6)})
            t4.to_csv(out_dir / f"K{K}_T4.csv", index=False)
            print(f"K={K} {T4_CHANNEL} (channel index {idx}): "
                  + ", ".join(f"{m} {v:+.4f}" for m, v in zip(map_names, templates[:, idx])))
    print("Saved to template_correlations/ in MICROSTATE_OUT_DIR")


if __name__ == "__main__":
    main()
