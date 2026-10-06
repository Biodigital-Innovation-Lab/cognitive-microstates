"""
fit_all_K.py -- group-level microstate templates for K = 4 to 9.

NOT CALLED BY run_all.sh. The reported analyses use the group templates of an
October 2025 run of this clustering procedure, deposited in Mendeley Data
(microstates_K{4..9}_templates.npz; K=5 = microstates_K5_templates_reference.npz)
and read as inputs. This script documents the procedure. Re-running it gives a
solution close to, but not identical with, the deposited templates: minimum
|r| per K between matched maps (Hungarian assignment on |r|) K=4 0.9999,
K=5 0.9978, K=6 0.9987, K=7 0.9785, K=8 0.9678, K=9 0.9972.

Purpose : same pipeline as fit_K5.py for K = 4..9 in one pass; preprocessing and
          GFP-peak extraction are done once and reused. Reordering to the
          reference and polarity flips (maps A and D) apply to K=5 only. Same
          logic as the last (superseding) cell of the original
          All_K_Microstates notebook.
Inputs  : MICROSTATE_OUT_DIR/eeg_trials.npz (from 01_reshape/reshape.py);
          optional MICROSTATE_REFERENCE_TEMPLATES (.npz with 'templates', K=5).
Outputs : MICROSTATE_OUT_DIR/microstates_K{K}_templates.npz   (K = 4..9)
              templates (19, K), ch_names, map_names, gev, sfreq
          MICROSTATE_OUT_DIR/microstates_K{K}_topomaps.svg
          MICROSTATE_OUT_DIR/microstates_K{K}_correlation.svg
          MICROSTATE_OUT_DIR/all_K_gev.csv   (K, GEV)
Environment : Python 3.11.14; mne 1.3.1, pycrostates 0.4.1; see requirements.txt.
              ModKMeans arguments are printed at run time. random_state = 42.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import seaborn as sns
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import microstate_utils as mu  # noqa: E402

K_RANGE = range(4, 10)   # K = 4, 5, 6, 7, 8, 9
LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]

K_SETTINGS = {
    K: {"map_names": LETTERS[:K], "use_reference": K == 5,
        "flip_maps": ["A", "D"] if K == 5 else []}
    for K in K_RANGE
}


def save_topomaps(templates, map_names, info, gev, out_path):
    K = templates.shape[1]
    fig, axes = plt.subplots(1, K, figsize=(3 * K, 3.5))
    for k in range(K):
        mne.viz.plot_topomap(
            templates[:, k], info, axes=axes[k], show=False,
            outlines="head", contours=0, sensors=False, extrapolate="head")
        axes[k].set_title(map_names[k], fontsize=13, fontweight="bold")
    fig.suptitle(f"K={K} Microstate Templates  |  GEV = {gev*100:.1f}%",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def save_correlation(templates, info, out_path):
    K = templates.shape[1]
    corr_abs = np.array([[abs(pearsonr(templates[:, i], templates[:, j])[0])
                          for j in range(K)] for i in range(K)])
    mask_upper = np.triu(np.ones_like(corr_abs, dtype=bool), k=1)

    THUMB, MARGIN, HMAP = 0.11, 0.20, 0.65
    FIG_SIZE = max(7, 5 + K * 0.4)

    fig = plt.figure(figsize=(FIG_SIZE, FIG_SIZE), facecolor="black")
    ax = fig.add_axes([MARGIN, MARGIN, HMAP, HMAP])
    ax.set_facecolor("black")
    sns.heatmap(
        corr_abs, mask=mask_upper, cmap="coolwarm", vmin=0, vmax=1,
        linewidths=1.2, linecolor="black", cbar=True,
        cbar_kws={"shrink": 0.5, "pad": 0.02}, ax=ax,
        xticklabels=False, yticklabels=False, square=True)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors="white", labelsize=9)
    cbar.outline.set_edgecolor("white")

    fontsize = max(7, 11 - (K - 5))
    for i in range(K):
        for j in range(i + 1):
            val = corr_abs[i, j]
            ax.text(j + 0.5, i + 0.5,
                    "1.00" if i == j else f"{val:.2f}",
                    ha="center", va="center", fontsize=fontsize,
                    fontweight="bold" if val >= 0.5 and i != j else "normal",
                    color="white" if val >= 0.6 or i == j else "black")
    ax.tick_params(left=False, bottom=False)

    for i in range(K):
        y0 = MARGIN + HMAP * (K - 1 - i) / K + HMAP / K * 0.05
        ax_t = fig.add_axes([MARGIN - THUMB - 0.01, y0, THUMB, THUMB * FIG_SIZE / FIG_SIZE],
                            facecolor="black")
        mne.viz.plot_topomap(templates[:, i], info, axes=ax_t, show=False,
                             outlines="head", contours=0, sensors=False,
                             extrapolate="head")
        ax_t.set_facecolor("black")
    for j in range(K):
        x0 = MARGIN + HMAP * j / K + HMAP / K * 0.05
        ax_t = fig.add_axes([x0, MARGIN - THUMB - 0.01, THUMB, THUMB],
                            facecolor="black")
        mne.viz.plot_topomap(templates[:, j], info, axes=ax_t, show=False,
                             outlines="head", contours=0, sensors=False,
                             extrapolate="head")
        ax_t.set_facecolor("black")
    fig.savefig(out_path, format="svg", bbox_inches="tight", facecolor="black")
    plt.close(fig)


def main():
    out_dir = config.get_out_dir()
    data, ch_names, sfreq = mu.load_trials(out_dir / "eeg_trials.npz")
    n_ch = len(ch_names)
    print(f"data shape: {data.shape}; sfreq: {sfreq} Hz")

    info = mu.make_info(ch_names, sfreq)
    data_prep = mu.preprocess(data, info)
    print(f"Preprocessed shape: {data_prep.shape}")
    print("Extracting GFP peaks once, reused for all K")
    gfp_peaks = mu.extract_peaks(data_prep, info)

    ref_T = mu.load_reference(n_ch)
    gev_per_k = {}

    for K in K_RANGE:
        cfg = K_SETTINGS[K]
        map_names = cfg["map_names"]
        print(f"\n{'=' * 60}\nK = {K}\n{'=' * 60}")
        print("Stage 1 clusters PER TRIAL (1,200 fits); stage 2 clusters the pooled centres.")
        for key, value in mu.describe_modkmeans(K).items():
            print(f"  ModKMeans {key}: {value}")

        templates = mu.fit_templates(gfp_peaks, K, info, n_ch)

        if cfg["use_reference"] and ref_T is not None:
            templates, order = mu.match_to_reference(templates, ref_T)
            print(f"  Reordered to reference; column order was {order}")

        for name in cfg["flip_maps"]:
            templates[:, map_names.index(name)] *= -1
            print(f"  Flipped polarity: {name}")

        gev = mu.compute_gev(data_prep, templates)
        gev_per_k[K] = gev
        print(f"  GEV (K={K}): {gev * 100:.4f}%")

        np.savez(
            out_dir / f"microstates_K{K}_templates.npz",
            templates=templates,
            ch_names=np.array(ch_names, dtype=str),
            map_names=np.array(map_names, dtype=str),
            gev=np.float64(gev),
            sfreq=np.float32(sfreq),
        )
        save_topomaps(templates, map_names, info, gev,
                      out_dir / f"microstates_K{K}_topomaps.svg")
        save_correlation(templates, info,
                         out_dir / f"microstates_K{K}_correlation.svg")
        print("  Saved templates and figures")

    print(f"\n{'=' * 60}\nGEV SUMMARY\n{'=' * 60}")
    with open(out_dir / "all_K_gev.csv", "w") as fh:
        fh.write("K,GEV\n")
        for K, gev in gev_per_k.items():
            print(f"  K={K}  GEV={gev * 100:.4f}%")
            fh.write(f"{K},{gev:.8f}\n")


if __name__ == "__main__":
    main()
