"""
fit_K5.py -- group-level K=5 microstate templates.

NOT CALLED BY run_all.sh. The reported analyses use the group templates of an
October 2025 run of this clustering procedure, deposited in Mendeley Data
(microstates_K{4..9}_templates.npz; K=5 = microstates_K5_templates_reference.npz)
and read as inputs. This script documents the procedure. Re-running it gives a
solution close to, but not identical with, the deposited templates: minimum
|r| per K between matched maps (Hungarian assignment on |r|) K=4 0.9999,
K=5 0.9978, K=6 0.9987, K=7 0.9785, K=8 0.9678, K=9 0.9972.

Purpose : average-reference + 4-30 Hz band-pass, GFP peaks, per-trial ModKMeans
          (stage 1), group-level ModKMeans on the pooled centres (stage 2),
          reorder to a reference (optional), flip polarity of maps A and D,
          compute GEV, save templates and figures. Same logic as the original
          K5 notebook.
Inputs  : MICROSTATE_OUT_DIR/eeg_trials.npz (from 01_reshape/reshape.py);
          optional MICROSTATE_REFERENCE_TEMPLATES (.npz with 'templates').
Outputs : MICROSTATE_OUT_DIR/microstates_K5_templates.npz
              templates (19, 5), ch_names, map_names, gev, sfreq
          MICROSTATE_OUT_DIR/microstates_K5_topomaps.svg
          MICROSTATE_OUT_DIR/microstates_K5_correlation.svg
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

N_CLUSTERS = 5
MAP_NAMES = ["A", "B", "C", "D", "E"]
FLIP_MAPS = ["A", "D"]


def save_topomaps(templates, info, gev, out_path):
    fig, axes = plt.subplots(1, N_CLUSTERS, figsize=(3 * N_CLUSTERS, 3.5))
    for k in range(N_CLUSTERS):
        mne.viz.plot_topomap(
            templates[:, k], info, axes=axes[k], show=False,
            outlines="head", contours=0, sensors=False, extrapolate="head",
        )
        axes[k].set_title(MAP_NAMES[k], fontsize=13, fontweight="bold")
    fig.suptitle(f"K={N_CLUSTERS} Microstate Templates  |  GEV = {gev*100:.1f}%",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def save_correlation(templates, info, out_path):
    corr_abs = np.zeros((N_CLUSTERS, N_CLUSTERS))
    for i in range(N_CLUSTERS):
        for j in range(N_CLUSTERS):
            corr_abs[i, j] = abs(pearsonr(templates[:, i], templates[:, j])[0])
    mask_upper = np.triu(np.ones_like(corr_abs, dtype=bool), k=1)

    THUMB, MARGIN, HMAP = 0.13, 0.22, 0.62
    fig = plt.figure(figsize=(7, 7), facecolor="black")
    ax_heat = fig.add_axes([MARGIN, MARGIN, HMAP, HMAP])
    ax_heat.set_facecolor("black")
    sns.heatmap(
        corr_abs, mask=mask_upper, cmap="coolwarm", vmin=0, vmax=1,
        linewidths=1.5, linecolor="black", cbar=True,
        cbar_kws={"shrink": 0.6, "pad": 0.02}, ax=ax_heat,
        xticklabels=False, yticklabels=False,
    )
    cbar = ax_heat.collections[0].colorbar
    cbar.ax.tick_params(colors="white", labelsize=9)
    cbar.outline.set_edgecolor("white")
    for i in range(N_CLUSTERS):
        for j in range(i + 1):
            val = corr_abs[i, j]
            txt = "1.00" if i == j else f"{val:.2f}"
            weight = "bold" if val >= 0.5 and i != j else "normal"
            color = "white" if val >= 0.6 or i == j else "black"
            ax_heat.text(j + 0.5, i + 0.5, txt, ha="center", va="center",
                         fontsize=11, fontweight=weight, color=color)
    ax_heat.tick_params(left=False, bottom=False)

    for i in range(N_CLUSTERS):
        y0 = MARGIN + HMAP * (N_CLUSTERS - 1 - i) / N_CLUSTERS + 0.01
        ax_t = fig.add_axes([MARGIN - THUMB - 0.02, y0, THUMB, THUMB],
                            facecolor="black")
        mne.viz.plot_topomap(templates[:, i], info, axes=ax_t, show=False,
                             outlines="head", contours=0, sensors=False,
                             extrapolate="head")
        ax_t.set_facecolor("black")
    for j in range(N_CLUSTERS):
        x0 = MARGIN + HMAP * j / N_CLUSTERS + 0.01
        ax_t = fig.add_axes([x0, MARGIN - THUMB - 0.02, THUMB, THUMB],
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

    print("Stage 1 clusters PER TRIAL (1,200 fits); stage 2 clusters the pooled centres.")
    for key, value in mu.describe_modkmeans(N_CLUSTERS).items():
        print(f"  ModKMeans {key}: {value}")

    info = mu.make_info(ch_names, sfreq)
    data_prep = mu.preprocess(data, info)
    print(f"Preprocessed shape: {data_prep.shape}")

    gfp_peaks = mu.extract_peaks(data_prep, info)
    templates = mu.fit_templates(gfp_peaks, N_CLUSTERS, info, n_ch)

    ref_T = mu.load_reference(n_ch)
    if ref_T is not None:
        templates, order = mu.match_to_reference(templates, ref_T)
        print(f"Reordered to reference; column order was {order}")
    else:
        print("No reference templates configured; keeping clustering order")

    for name in FLIP_MAPS:
        templates[:, MAP_NAMES.index(name)] *= -1
        print(f"Flipped polarity of map {name}")

    gev = mu.compute_gev(data_prep, templates)
    print(f"GEV (K={N_CLUSTERS}): {gev * 100:.4f}%")

    save_topomaps(templates, info, gev, out_dir / "microstates_K5_topomaps.svg")
    save_correlation(templates, info, out_dir / "microstates_K5_correlation.svg")
    np.savez(
        out_dir / "microstates_K5_templates.npz",
        templates=templates,
        ch_names=np.array(ch_names, dtype=str),
        map_names=np.array(MAP_NAMES, dtype=str),
        gev=np.float64(gev),
        sfreq=np.float32(sfreq),
    )
    print("Saved templates and figures to MICROSTATE_OUT_DIR")


if __name__ == "__main__":
    main()
