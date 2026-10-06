#!/usr/bin/env python3
"""
Within-Condition Needleman–Wunsch Pairwise Alignment Analysis
=============================================================

Tests whether meta-consensus EEG microstate sequences encode non-random
temporal structure within each condition × window cell (Go-early, Go-late,
NoGo-early, NoGo-late), using permutation-based null models and odds ratios.

Alignment parameters
--------------------
  match = +1, mismatch = 0, gap penalty = −0.5 (open = extend)
  Normalisation: NW_score / (min(len(seq₁), len(seq₂)) × match)

Null model
----------
  Group level : 1 000 permutations (character-shuffle preserving frequencies)
  Subject level: 500 shuffles per subject (empirical p < 0.05)

NOTE — The normalisation divides the raw Needleman–Wunsch score (which
accumulates +1 per match, 0 per mismatch, and −0.5 per gap position) by
the *minimum* sequence length, not the maximum.  The Methods text currently
reads "min(len(seq₁), len(seq₂))"; this matches what is computed here.

Environment
-----------
  Python       3.11.14
  biopython    1.81
  scipy        1.11.4
  numpy        1.25.2
  pandas       2.0.3
  matplotlib   3.11.2
  seaborn      0.13.2
  (project venv; exact pins in ../requirements.txt)

Usage
-----
  python within_condition_nw_analysis.py [--source eeg|deposited]

Input
-----
  msa_consensus_SUBJECT.csv
    --source deposited (default): folder given by MICROSTATE_INPUT_DIR
    --source eeg: tables/ in MICROSTATE_OUT_DIR (from 05_alignment/consensus.py)
  Columns used: subject, condition (GO / NG), window, consensus.
  No participant identifiers or sex columns are used.

Outputs  (created automatically under get_out_dir())
-------
  tables/within_condition_summary.csv
  figures/fig_wc_null_distributions.png
  figures/fig_wc_null_distributions.svg
  figures/fig_wc_or_forest.png
  figures/fig_wc_or_forest.svg
  figures/fig_wc_subject_significance.png
  figures/fig_wc_subject_significance.svg
  figures/fig_wc_summary_heatmap.png
  figures/fig_wc_summary_heatmap.svg
  figures/fig_or_forest_within_condition.png   (compact forest plot)
  figures/fig_or_forest_within_condition.svg

Reproducibility
---------------
  Fixed seed (42). SVGs are written with a fixed hash salt and no date
  metadata so that repeated runs give identical files.
"""

# ══════════════════════════════════════════════════════════════════════
# Imports
# ══════════════════════════════════════════════════════════════════════
import argparse
import sys
import pathlib
import pandas as pd
import numpy as np
from itertools import combinations
from Bio.Align import PairwiseAligner
from scipy import stats
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config  # noqa: E402

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"     # keep text editable in SVG
matplotlib.rcParams["svg.hashsalt"] = "microstate"
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════
_parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
_parser.add_argument("--source", choices=["eeg", "deposited"],
                     default="deposited",
                     help="consensus table: built from the EEG "
                          "(05_alignment/consensus.py) or deposited "
                          "(default: deposited)")
SOURCE = _parser.parse_args().source
if SOURCE == "eeg":
    INPUT_CSV = config.get_out_dir() / "tables" / "msa_consensus_SUBJECT.csv"
else:
    INPUT_CSV = config.get_input_dir() / "msa_consensus_SUBJECT.csv"
TABLE_DIR  = config.get_out_dir() / "tables"
FIG_DIR    = config.get_out_dir() / "figures"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════════
MATCH       = 1
MISMATCH    = 0
GAP         = -0.5
N_PERM      = 1_000      # group-level permutations
N_SUBJ_PERM = 500        # per-subject permutations
SEED        = 42

np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════════════
# Plot style
# ══════════════════════════════════════════════════════════════════════
sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.dpi": 150})

# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════
def _make_aligner():
    a = PairwiseAligner()
    a.mode = "global"
    a.match_score = MATCH
    a.mismatch_score = MISMATCH
    a.open_gap_score = GAP
    a.extend_gap_score = GAP
    return a

_aligner = _make_aligner()


def nw_score(seq1: str, seq2: str) -> float:
    """Normalised Needleman–Wunsch score."""
    raw = _aligner.score(seq1, seq2)
    max_possible = min(len(seq1), len(seq2)) * MATCH
    return raw / max_possible if max_possible > 0 else 0.0


def shuffle_seq(seq: str) -> str:
    """Shuffle characters in place (destroys temporal order, keeps freqs)."""
    chars = list(seq)
    np.random.shuffle(chars)
    return "".join(chars)


def pairwise_scores(seqs: list[str]) -> np.ndarray:
    """All (n choose 2) pairwise NW scores for a list of sequences."""
    scores = []
    for i, j in combinations(range(len(seqs)), 2):
        scores.append(nw_score(seqs[i], seqs[j]))
    return np.array(scores)


def _save(fig, stem: str):
    """Save figure as PNG + SVG and close."""
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.svg", format="svg", bbox_inches="tight",
                metadata={"Date": None})
    plt.close(fig)
    print(f"  → {stem}.png / .svg")


# ══════════════════════════════════════════════════════════════════════
# 1. Load & filter (early + late only, no "full")
# ══════════════════════════════════════════════════════════════════════
print("Loading data …")
df = pd.read_csv(INPUT_CSV)
df = df[df["window"].isin(["early", "late"])].reset_index(drop=True)
print(f"  {df.shape[0]} rows, {df['subject'].nunique()} subjects, "
      f"conditions={list(df['condition'].unique())}, "
      f"windows={list(df['window'].unique())}")

# ══════════════════════════════════════════════════════════════════════
# 2. Within-condition analysis
# ══════════════════════════════════════════════════════════════════════
CELLS = [("GO", "early"), ("GO", "late"), ("NG", "early"), ("NG", "late")]
cell_results: dict = {}

for cond, window in CELLS:
    key = f"{cond}_{window}"
    print(f"\n{'─'*60}")
    print(f"  {key}")
    print(f"{'─'*60}")

    subset = df[(df["condition"] == cond) & (df["window"] == window)]
    seqs = subset["consensus"].tolist()
    n_subj = len(seqs)

    # ── Observed pairwise ─────────────────────────────────────────
    obs_scores = pairwise_scores(seqs)
    obs_mean = np.mean(obs_scores)
    print(f"  N subjects = {n_subj},  N pairs = {len(obs_scores)}")
    print(f"  Observed mean = {obs_mean:.4f}  (sd {np.std(obs_scores):.4f})")

    # ── Group-level null (1 000 permutations) ─────────────────────
    null_means = np.empty(N_PERM)
    for p in range(N_PERM):
        shuffled = [shuffle_seq(s) for s in seqs]
        null_means[p] = np.mean(pairwise_scores(shuffled))

    p_value = np.mean(null_means >= obs_mean)
    z_score = ((obs_mean - null_means.mean()) / null_means.std()
               if null_means.std() > 0 else 0.0)
    print(f"  Null mean   = {null_means.mean():.4f}  (sd {null_means.std():.4f})")
    print(f"  z = {z_score:.2f},  p = {p_value}")

    # ── Group-level OR (pair scores vs null pair pool) ────────────
    null_pair_pool = np.concatenate(
        [pairwise_scores([shuffle_seq(s) for s in seqs]) for _ in range(20)]
    )
    pair_threshold = np.median(null_pair_pool)

    a = int(np.sum(obs_scores > pair_threshold))
    b = len(obs_scores) - a
    c = int(np.sum(null_pair_pool > pair_threshold))
    d = len(null_pair_pool) - c

    group_or = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
    log_or = np.log(group_or)
    se = np.sqrt(1/(a+0.5) + 1/(b+0.5) + 1/(c+0.5) + 1/(d+0.5))
    ci_low  = np.exp(log_or - 1.96 * se)
    ci_high = np.exp(log_or + 1.96 * se)
    fisher_or, fisher_p = stats.fisher_exact(
        np.array([[a, b], [c, d]]), alternative="greater"
    )
    print(f"  Pairs above null median: {a}/{len(obs_scores)} "
          f"({100*a/len(obs_scores):.1f}%)")
    print(f"  Group OR = {group_or:.3f}  "
          f"[{ci_low:.3f}–{ci_high:.3f}]  Fisher p = {fisher_p:.6f}")

    # ── Subject-level OR (500 shuffles per subject) ───────────────
    subj_mean_obs = np.empty(n_subj)
    for i in range(n_subj):
        subj_mean_obs[i] = np.mean(
            [nw_score(seqs[i], seqs[j]) for j in range(n_subj) if j != i]
        )

    n_sig = 0
    subj_pvals = []
    for i in range(n_subj):
        null_subj = np.empty(N_SUBJ_PERM)
        for t in range(N_SUBJ_PERM):
            shuffled = [shuffle_seq(s) for s in seqs]
            null_subj[t] = np.mean(
                [nw_score(shuffled[i], shuffled[j])
                 for j in range(n_subj) if j != i]
            )
        p_subj = float(np.mean(null_subj >= subj_mean_obs[i]))
        subj_pvals.append(p_subj)
        if p_subj < 0.05:
            n_sig += 1

    expected = n_subj * 0.05
    subj_or = ((n_sig + 0.5) * (n_subj - expected + 0.5)) / \
              ((n_subj - n_sig + 0.5) * (expected + 0.5))
    binom_p = stats.binomtest(n_sig, n_subj, 0.05,
                              alternative="greater").pvalue
    print(f"  Subject sig: {n_sig}/{n_subj} ({100*n_sig/n_subj:.1f}%) "
          f"vs expected {expected:.1f} (5%)")
    print(f"  Subject OR = {subj_or:.2f},  binomial p = {binom_p:.6f}")

    # ── Store ─────────────────────────────────────────────────────
    cell_results[key] = dict(
        obs_mean=obs_mean, obs_sd=np.std(obs_scores), obs_scores=obs_scores,
        null_mean=null_means.mean(), null_sd=null_means.std(),
        null_distribution=null_means,
        z_score=z_score, p_value=p_value,
        group_or=group_or, ci_95=(ci_low, ci_high), fisher_p=fisher_p,
        pair_threshold=pair_threshold,
        obs_above=a, obs_total=len(obs_scores),
        n_subj_sig=n_sig, n_subj=n_subj,
        subj_or=subj_or, binom_p=binom_p,
        subj_pvals=subj_pvals, subj_mean_obs=subj_mean_obs,
    )

# ══════════════════════════════════════════════════════════════════════
# 3. Summary table
# ══════════════════════════════════════════════════════════════════════
print("\nSaving summary table …")
rows = []
for key, r in cell_results.items():
    rows.append(dict(
        cell=key,
        observed_mean_NW=r["obs_mean"], observed_sd=r["obs_sd"],
        null_mean_NW=r["null_mean"], null_sd=r["null_sd"],
        z_score=r["z_score"], permutation_p=r["p_value"],
        group_OR=r["group_or"],
        group_OR_CI_low=r["ci_95"][0], group_OR_CI_high=r["ci_95"][1],
        fisher_p=r["fisher_p"],
        pct_pairs_above_null=100 * r["obs_above"] / r["obs_total"],
        n_subj_sig=r["n_subj_sig"], n_subj=r["n_subj"],
        pct_subj_sig=100 * r["n_subj_sig"] / r["n_subj"],
        subject_OR=r["subj_or"], binomial_p=r["binom_p"],
    ))
summary_df = pd.DataFrame(rows)
summary_df.to_csv(TABLE_DIR / "within_condition_summary.csv", index=False)
print("  → within_condition_summary.csv")

# ══════════════════════════════════════════════════════════════════════
# 4. Figures
# ══════════════════════════════════════════════════════════════════════
CELL_KEYS = ["GO_early", "GO_late", "NG_early", "NG_late"]
LABELS = {"GO_early": "GO Early", "GO_late": "GO Late",
          "NG_early": "NG Early", "NG_late": "NG Late"}
COLORS = {"GO_early": "#27ae60", "GO_late": "#2ecc71",
          "NG_early": "#8e44ad", "NG_late": "#9b59b6"}

# ── 4a  Null distributions (2 × 2) ───────────────────────────────────
print("\nGenerating figures …")
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(
    "Within-Condition Between-Subject Similarity vs Null (Shuffled)\n"
    "match=+1, mismatch=0, gap=−0.5",
    fontsize=14, fontweight="bold", y=1.0,
)
for ax, key in zip(axes.flat, CELL_KEYS):
    r = cell_results[key]
    ax.hist(r["null_distribution"], bins=40, alpha=0.6, color="#95a5a6",
            density=True, label="Null (shuffled)", edgecolor="white")
    ax.axvline(r["obs_mean"], color=COLORS[key], linewidth=2.5,
               label=f'Observed = {r["obs_mean"]:.3f}')
    ax.axvline(r["null_mean"], color="#7f8c8d", linewidth=1.5,
               linestyle="--", label=f'Null = {r["null_mean"]:.3f}')
    ax.set_title(LABELS[key], fontweight="bold")
    ax.set_xlabel("Mean Pairwise NW Score")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7, loc="upper right")
    p_txt = "p < 0.001" if r["p_value"] < 0.001 else f'p = {r["p_value"]:.3f}'
    ax.text(0.02, 0.95, f'z = {r["z_score"]:.2f}\n{p_txt}',
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
plt.tight_layout()
_save(fig, "fig_wc_null_distributions")

# ── 4b  Forest plot OR (full-width) ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4.5))
fig.suptitle("Odds Ratios: Within-Condition Pairwise Similarity vs Null",
             fontsize=13, fontweight="bold")
for i, key in enumerate(CELL_KEYS):
    r = cell_results[key]
    ci_lo, ci_hi = r["ci_95"]
    ax.errorbar(r["group_or"], i,
                xerr=[[r["group_or"]-ci_lo], [ci_hi-r["group_or"]]],
                fmt="o", color=COLORS[key], markersize=10, capsize=5,
                linewidth=2, markeredgecolor="white", markeredgewidth=1)
    p_txt = "p < 0.001" if r["fisher_p"] < 0.001 else f'p = {r["fisher_p"]:.4f}'
    ax.text(ci_hi + 0.1, i,
            f'OR = {r["group_or"]:.2f} [{ci_lo:.2f}–{ci_hi:.2f}]\n{p_txt}',
            va="center", ha="left", fontsize=9, color="#2c3e50")
ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7,
           label="OR = 1")
ax.set_yticks(range(len(CELL_KEYS)))
ax.set_yticklabels([LABELS[k] for k in CELL_KEYS])
ax.set_xlabel("Odds Ratio")
ax.set_xlim(0.5, 5.5)
ax.legend(loc="lower right", fontsize=9)
ax.invert_yaxis()
plt.tight_layout()
_save(fig, "fig_wc_or_forest")

# ── 4c  Subject-level bars (significance + OR) ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
bar_colors = [COLORS[k] for k in CELL_KEYS]
xlabels = [LABELS[k] for k in CELL_KEYS]

# left: % significant
ax = axes[0]
pcts = [100 * cell_results[k]["n_subj_sig"] / cell_results[k]["n_subj"]
        for k in CELL_KEYS]
bars = ax.bar(range(4), pcts, color=bar_colors, alpha=0.8, edgecolor="white")
ax.axhline(5, color="gray", linestyle="--", linewidth=1.5,
           label="Expected (5%)")
ax.set_xticks(range(4)); ax.set_xticklabels(xlabels, fontsize=9)
ax.set_ylabel("% Subjects p < 0.05")
ax.set_title("Subject-Level Significance\n(Within-Condition vs Shuffled)",
             fontweight="bold")
ax.legend(fontsize=8); ax.set_ylim(0, 115)
for bar, pct, key in zip(bars, pcts, CELL_KEYS):
    bp = cell_results[key]["binom_p"]
    star = "***" if bp < 0.001 else "**" if bp < 0.01 else "*" if bp < 0.05 else "ns"
    n = cell_results[key]["n_subj_sig"]
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{n}/30\n{star}", ha="center", fontsize=9, fontweight="bold")

# right: subject OR
ax = axes[1]
ors = [cell_results[k]["subj_or"] for k in CELL_KEYS]
bars2 = ax.bar(range(4), ors, color=bar_colors, alpha=0.8, edgecolor="white")
ax.axhline(1, color="gray", linestyle="--", linewidth=1.5, label="OR = 1")
ax.set_xticks(range(4)); ax.set_xticklabels(xlabels, fontsize=9)
ax.set_ylabel("Odds Ratio (log scale)"); ax.set_yscale("log")
ax.set_title("Subject-Level Odds Ratios\n(Within-Condition)", fontweight="bold")
ax.legend(fontsize=8)
for bar, ov, key in zip(bars2, ors, CELL_KEYS):
    bp = cell_results[key]["binom_p"]
    star = "***" if bp < 0.001 else "**" if bp < 0.01 else "*" if bp < 0.05 else "ns"
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.3,
            f"{ov:.0f}\n{star}", ha="center", fontsize=9, fontweight="bold")
plt.tight_layout()
_save(fig, "fig_wc_subject_significance")

# ── 4d  Summary heatmap ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 4))
metrics = ["Obs. NW", "Null NW", "Z", "Perm. p", "Group OR",
           "% Pairs Above", "% Subj. Sig.", "Subj. OR", "Binom. p"]
matrix = np.zeros((4, len(metrics)))
for i, key in enumerate(CELL_KEYS):
    r = summary_df[summary_df["cell"] == key].iloc[0]
    matrix[i] = [r["observed_mean_NW"], r["null_mean_NW"], r["z_score"],
                 r["permutation_p"], r["group_OR"], r["pct_pairs_above_null"],
                 r["pct_subj_sig"], r["subject_OR"], r["binomial_p"]]
annot = []
for i in range(matrix.shape[0]):
    row = []
    for j in range(matrix.shape[1]):
        v = matrix[i, j]
        if j in (3, 8):    row.append("<.001" if v < 0.001 else f"{v:.4f}")
        elif j in (0, 1):  row.append(f"{v:.4f}")
        elif j == 2:       row.append(f"{v:.2f}")
        elif j in (5, 6):  row.append(f"{v:.1f}%")
        else:              row.append(f"{v:.2f}")
    annot.append(row)
matrix_n = matrix.copy()
for j in range(matrix.shape[1]):
    col = matrix_n[:, j]
    rng = col.max() - col.min()
    matrix_n[:, j] = (col - col.min()) / rng if rng else 0.5
sns.heatmap(matrix_n, annot=np.array(annot), fmt="", ax=ax,
            xticklabels=metrics,
            yticklabels=[LABELS[k] for k in CELL_KEYS],
            cmap="YlGn", cbar=False, linewidths=1, linecolor="white")
ax.set_title("Within-Condition Analysis Summary "
             "(Between-Subject Pairwise NW vs Shuffled Null)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
_save(fig, "fig_wc_summary_heatmap")

# ── 4e  Compact forest plot (half-width, as in Figure 3) ─────────────
fig, ax = plt.subplots(figsize=(6, 5))
ax.set_title("Odds Ratios: Within-Condition\nPairwise Similarity vs Null",
             fontsize=11, fontweight="bold")
y_pos = np.arange(len(CELL_KEYS))
for i, key in enumerate(CELL_KEYS):
    r = cell_results[key]
    ci_lo, ci_hi = r["ci_95"]
    ax.errorbar(r["group_or"], i,
                xerr=[[r["group_or"]-ci_lo], [ci_hi-r["group_or"]]],
                fmt="o", color=COLORS[key], markersize=9, capsize=4,
                linewidth=1.8, markeredgecolor="white", markeredgewidth=0.8)
    p_txt = "p < 0.001" if r["fisher_p"] < 0.001 else f'p = {r["fisher_p"]:.3f}'
    ax.text(ci_hi + 0.1, i,
            f'OR = {r["group_or"]:.2f} [{ci_lo:.2f}–{ci_hi:.2f}]\n{p_txt}',
            va="center", ha="left", fontsize=8, color="#2c3e50")
ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7,
           label="OR = 1")
ax.set_yticks(y_pos)
ax.set_yticklabels([LABELS[k] for k in CELL_KEYS], fontsize=10)
ax.set_xlabel("Odds Ratio", fontsize=10)
ax.set_xlim(0.5, 6.8)
ax.legend(loc="upper right", fontsize=8)
ax.invert_yaxis()
plt.tight_layout()
_save(fig, "fig_or_forest_within_condition")

# ══════════════════════════════════════════════════════════════════════
print("\n✓  All outputs written.")
