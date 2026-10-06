"""
motif_analysis.py -- k-mer screen, bigram rates, conditional successors and
the E-A / E-A-D / D-frequency tests for the Go/NoGo microstate sequences.

Purpose : regenerate Supplementary Tables S2 (k-mer screen), S3 (bigram
          rates) and S4 (conditional successors, plus the Table 3 filter),
          the Mann-Whitney tests of E-A rate, E-A-D rate and D frequency, and
          the E-A-X Fisher table. The counting functions, tests and FDR
          families are copied from the original motif_analysis_pipeline.py
          without change (trigram counts from the window string; BH within
          each bigram's 4 successors, self-transition excluded; BH per window
          and k for k-mers; minimum 5 combined occurrences). Only the window
          source differs.
Windows : --windows exact (default)
              sequence_full (650 characters, one per sample) is read from
              backfit.py output (reference templates). Derived as in S5:
              full = consecutive repeats removed from sequence_full;
              early = sequence_full[50:250]; late = sequence_full[250:450];
              each collapsed independently after slicing.
          --windows proportional
              reads microstate_sequences_FULL.csv (its sequence_full is not
              the 650-sample string: collapse of samples 0-149 joined to the
              collapse of samples 150-649). --source eeg uses the table built
              by 03_sequences/full_prepost.py; --source deposited (default)
              uses the deposited file. Full = its sequence_collapsed;
              early/late = proportional index mapping of its sequence_full
              (round(L * tp / 650)), then collapsed. This mode reproduces the
              deposited S3 and S4.
Usage   : python motif_analysis.py [--windows exact|proportional]
                                   [--source eeg|deposited]
          (--source applies to --windows proportional only)
Inputs  : exact        : MICROSTATE_OUT_DIR/sequences/reference/
                         microstate_sequences_K5.csv (from 03_sequences/backfit.py)
          proportional : --source deposited:
                         MICROSTATE_INPUT_DIR/microstate_sequences_FULL.csv
                         --source eeg: MICROSTATE_OUT_DIR/tables/
                         microstate_sequences_FULL_prepost.csv
                         (columns used: subject, condition, trial,
                         sequence_full, sequence_collapsed; no other column is
                         read)
Outputs : MICROSTATE_OUT_DIR/tables/  (suffix _exact or _proportional)
            S2_kmer_screen_k{3,4,5}_<mode>.csv   S2 layout, all windows
            S2_kmer_summary_<mode>.csv           motifs observed/tested/FDR hits
            S3_bigram_rates_<mode>.csv           S3 layout, all windows/levels;
                                                 extra columns "additional, not
                                                 in the original analysis"
            S4_conditional_successors_<mode>.csv S4 layout, all windows
            Table3_significant_conditional_early_<mode>.csv  S4 rows of the
                                                 early window with Sig = *
            EA_EAD_Dfreq_tests_<mode>.csv        Mann-Whitney (trial/subject)
            EAX_fisher_<mode>.csv                E-A-X Fisher, FDR with m = 4
                                                 and m = 5
          No participant identifiers are written.
Environment : Python 3.11.14; numpy 1.25.2, pandas 2.0.3, scipy 1.11.4,
              statsmodels 0.14.6; see requirements.txt. No random numbers
              are used; repeated runs are identical.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION (as in the original pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

STATES = list("ABCDE")
N_TP = 650                    # timepoints per trial at 500 Hz (1,300 ms)
ALPHA = 0.05                  # FDR threshold
MIN_KMER_COUNT = 5            # minimum combined occurrences for k-mer screen

# Temporal windows: (name, start_tp, end_tp)
WINDOWS = {
    "full":  (0,   650),
    "early": (50,  250),      # 100-500 ms
    "late":  (250, 450),      # 500-900 ms
}
WINDOW_LABELS = {
    "full": "Full epoch (0–1300 ms)",
    "early": "Early phase (100–500 ms)",
    "late": "Late phase (500–900 ms)",
}

# Directed bigrams (no self-transitions in collapsed sequences)
ALL_DUPLETS = [a + b for a in STATES for b in STATES if a != b]  # 20

ADDITIONAL = "additional, not in the original analysis"


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS (copied from the original pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

def collapse(seq: str) -> str:
    """Run-length encode: remove consecutive repeated characters."""
    if not seq:
        return ""
    out = [seq[0]]
    for c in seq[1:]:
        if c != out[-1]:
            out.append(c)
    return "".join(out)


def _extract_window(seq_full: str, tp_start: int, tp_end: int) -> str:
    """Proportional mapping: round(L * tp / 650), then collapse (original)."""
    L = len(seq_full)
    cs = max(0, min(int(round(L * tp_start / N_TP)), L))
    ce = max(cs, min(int(round(L * tp_end / N_TP)), L))
    return collapse(seq_full[cs:ce])


def count_kmers(seq: str, k: int) -> Counter:
    """Count all contiguous subsequences of length k."""
    return Counter(seq[i:i + k] for i in range(len(seq) - k + 1))


def cohen_d_pooled(go_vals: np.ndarray, ng_vals: np.ndarray) -> float:
    """Cohen's d, d = (mean_NG - mean_GO) / pooled SD."""
    n1, n2 = len(go_vals), len(ng_vals)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = go_vals.var(ddof=1), ng_vals.var(ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return (ng_vals.mean() - go_vals.mean()) / pooled


def mwu_effect_r(go: np.ndarray, ng: np.ndarray) -> tuple:
    """Mann-Whitney U (two-sided) with effect size r = |Z|/sqrt(N)."""
    U, p = mannwhitneyu(go, ng, alternative="two-sided")
    N = len(go) + len(ng)
    if p < 1.0 and p > 0.0:
        z = abs(stats.norm.isf(p / 2))
    else:
        z = 0.0
    return U, p, z / np.sqrt(N)


def mwu_effect_r_u(go: np.ndarray, ng: np.ndarray) -> tuple:
    """
    Mann-Whitney U (two-sided) with effect size r = |Z|/sqrt(N), Z computed
    from U without tie correction: Z = (U - n1*n2/2) / sqrt(n1*n2*(N+1)/12).
    Used for the E-A rate, E-A-D rate and D frequency tests.
    """
    U, p = mannwhitneyu(go, ng, alternative="two-sided")
    n1, n2 = len(go), len(ng)
    N = n1 + n2
    z = (U - n1 * n2 / 2) / np.sqrt(n1 * n2 * (N + 1) / 12)
    return U, p, abs(z) / np.sqrt(N)


# ═══════════════════════════════════════════════════════════════════════════════
# WINDOW SOURCE (the only part that differs from the original pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

def load_trials(mode: str, source: str = "deposited") -> pd.DataFrame:
    """
    One row per trial with columns subject, condition, trial and one collapsed
    window string each: full, early, late.
    """
    if mode == "exact":
        path = (config.get_out_dir() / "sequences" / "reference"
                / "microstate_sequences_K5.csv")
        df = pd.read_csv(path, usecols=["subject", "condition", "trial",
                                        "sequence_full"])
        assert df["sequence_full"].str.len().eq(N_TP).all()
        seq = df["sequence_full"]
        df["full"] = [collapse(s) for s in seq]
        df["early"] = [collapse(s[50:250]) for s in seq]
        df["late"] = [collapse(s[250:450]) for s in seq]
    else:
        if source == "eeg":
            path = (config.get_out_dir() / "tables"
                    / "microstate_sequences_FULL_prepost.csv")
        else:
            path = config.get_input_dir() / "microstate_sequences_FULL.csv"
        df = pd.read_csv(path, usecols=["subject", "condition", "trial",
                                        "sequence_full", "sequence_collapsed"])
        df["full"] = df["sequence_collapsed"]
        for wname in ("early", "late"):
            tp_s, tp_e = WINDOWS[wname]
            df[wname] = [_extract_window(s, tp_s, tp_e)
                         for s in df["sequence_full"]]
        df = df.drop(columns=["sequence_collapsed"])
    df = df.drop(columns=["sequence_full"])
    assert len(df) == 1200
    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 0 -- PER-TRIAL COUNT TABLE (original logic)
# ═══════════════════════════════════════════════════════════════════════════════

def build_trial_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per trial x window: bigram counts and rates, trigram counts,
    D count and frequency (count / collapsed length).
    """
    rows = []
    for _, trial in df.iterrows():
        for wname in WINDOWS:
            seq = trial[wname]
            L = len(seq)
            bigrams = Counter()
            trigrams = Counter()

            for i in range(L - 1):
                bigrams[seq[i] + seq[i + 1]] += 1
            for i in range(L - 2):
                trigrams[seq[i:i + 3]] += 1

            r = {
                "subject": trial["subject"],
                "condition": trial["condition"],
                "trial": trial["trial"],
                "window": wname,
                "seq_len": L,
            }

            for dup in ALL_DUPLETS:
                r[f"dup_{dup}_count"] = bigrams.get(dup, 0)
                r[f"dup_{dup}_rate"] = bigrams.get(dup, 0) / L if L > 0 else 0

            for dup in ALL_DUPLETS:
                for x in STATES:
                    trip = dup + x
                    r[f"trip_{trip}_count"] = trigrams.get(trip, 0)

            r["D_count"] = seq.count("D")
            r["D_freq"] = seq.count("D") / L if L > 0 else 0

            rows.append(r)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 1 -- EXHAUSTIVE k-MER SCREEN (original logic)
# ═══════════════════════════════════════════════════════════════════════════════

def level1_kmer_screen(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pool k-mer occurrences per condition; Fisher exact (one-sided, NoGo >
    Go) per motif with combined count >= MIN_KMER_COUNT; BH per window x k.
    """
    results = []

    for wname in WINDOWS:
        go_seqs = list(df.loc[df["condition"] == "GO", wname])
        ng_seqs = list(df.loc[df["condition"] != "GO", wname])

        for k in (3, 4, 5):
            # per-trial counters computed once (same values as recomputing)
            go_cts = [count_kmers(s, k) for s in go_seqs]
            ng_cts = [count_kmers(s, k) for s in ng_seqs]
            go_tot_t = [max(sum(c.values()), 1) for c in go_cts]
            ng_tot_t = [max(sum(c.values()), 1) for c in ng_cts]

            go_pool = Counter()
            ng_pool = Counter()
            for c in go_cts:
                go_pool += c
            for c in ng_cts:
                ng_pool += c

            go_total = sum(go_pool.values())
            ng_total = sum(ng_pool.values())

            all_motifs = set(go_pool.keys()) | set(ng_pool.keys())
            tested_motifs = sorted(
                m for m in all_motifs
                if go_pool.get(m, 0) + ng_pool.get(m, 0) >= MIN_KMER_COUNT
            )

            pvals = []
            motif_rows = []

            for motif in tested_motifs:
                go_n = go_pool.get(motif, 0)
                ng_n = ng_pool.get(motif, 0)
                go_other = go_total - go_n
                ng_other = ng_total - ng_n

                if go_n > 0 and ng_other > 0:
                    OR = (ng_n * go_other) / (go_n * ng_other)
                else:
                    OR = np.nan

                table = np.array([[ng_n, ng_other], [go_n, go_other]])
                _, p = fisher_exact(table, alternative="greater")

                go_rates = np.array([c.get(motif, 0) / t
                                     for c, t in zip(go_cts, go_tot_t)])
                ng_rates = np.array([c.get(motif, 0) / t
                                     for c, t in zip(ng_cts, ng_tot_t)])
                d = cohen_d_pooled(go_rates, ng_rates)

                go_pct = go_n / go_total * 100 if go_total > 0 else 0
                ng_pct = ng_n / ng_total * 100 if ng_total > 0 else 0

                pvals.append(p)
                motif_rows.append({
                    "Window": wname, "k": k, "Motif": motif,
                    "GO_n": go_n, "NG_n": ng_n,
                    "GO_%": round(go_pct, 2), "NG_%": round(ng_pct, 2),
                    "Diff_%": round(ng_pct - go_pct, 2),
                    "OR": round(OR, 4) if not np.isnan(OR) else np.nan,
                    "Cohen_d": round(d, 4),
                    "p_raw": round(p, 6),
                })

            if pvals:
                _, pfdr, _, _ = multipletests(pvals, method="fdr_bh")
                for i, mr in enumerate(motif_rows):
                    mr["p_FDR"] = round(pfdr[i], 6)
                    mr["Sig"] = "*" if pfdr[i] < ALPHA else "ns"

            results.extend(motif_rows)

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 2 -- BIGRAM RATES (original logic) + additional BH column
# ═══════════════════════════════════════════════════════════════════════════════

def level2_bigram_rates(tdf: pd.DataFrame) -> pd.DataFrame:
    """
    For each of the 20 directed bigrams, rate (count / collapsed length)
    compared Go vs NoGo: Mann-Whitney U two-sided, trial level (n = 600 per
    condition) and subject level (n = 30). The raw p is not changed. The
    columns labelled "additional, not in the original analysis" hold BH over
    the 20 bigrams within each window and level.
    """
    results = []

    for wname in WINDOWS:
        wd = tdf[tdf["window"] == wname]

        for level in ("trial", "subject"):
            block = []
            for dup in ALL_DUPLETS:
                col = f"dup_{dup}_rate"

                if level == "trial":
                    go_vals = wd.loc[wd["condition"] == "GO", col].values
                    ng_vals = wd.loc[wd["condition"] == "NG", col].values
                else:
                    go_vals = (wd[wd["condition"] == "GO"]
                               .groupby("subject")[col].mean()
                               .sort_index().values)
                    ng_vals = (wd[wd["condition"] == "NG"]
                               .groupby("subject")[col].mean()
                               .sort_index().values)

                U, p, r = mwu_effect_r(go_vals, ng_vals)

                block.append({
                    "Window": wname, "Level": level, "Duplet": dup,
                    "N_GO": len(go_vals), "N_NG": len(ng_vals),
                    "GO_mean": round(go_vals.mean(), 6),
                    "GO_sd":   round(go_vals.std(),  6),
                    "NG_mean": round(ng_vals.mean(), 6),
                    "NG_sd":   round(ng_vals.std(),  6),
                    "U": round(U, 1),
                    "p": round(p, 4),
                    "r": round(r, 3),
                    "Sig": "*" if p < ALPHA else "ns",
                    "_p_unrounded": p,
                })

            _, p_bh, _, _ = multipletests([b["_p_unrounded"] for b in block],
                                          method="fdr_bh")
            for b, pb in zip(block, p_bh):
                b[f"p_BH ({ADDITIONAL})"] = round(pb, 6)
                b[f"Sig_BH ({ADDITIONAL})"] = "*" if pb < ALPHA else "ns"
                del b["_p_unrounded"]
            results.extend(block)

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 3 -- CONDITIONAL SUCCESSORS (original logic)
# ═══════════════════════════════════════════════════════════════════════════════

def level3_conditional(tdf: pd.DataFrame) -> pd.DataFrame:
    """
    For each directed bigram M1M2, Fisher exact (one-sided, NoGo > Go) per
    successor X != M2 on counts pooled over trials; BH within the bigram's 4
    successors.
    """
    results = []

    for wname in WINDOWS:
        wd = tdf[tdf["window"] == wname]
        go_t = wd[wd["condition"] == "GO"]
        ng_t = wd[wd["condition"] == "NG"]

        for dup in ALL_DUPLETS:
            m2 = dup[1]
            successors = [x for x in STATES if x != m2]  # 4 successors

            go_counts, ng_counts = {}, {}
            for x in successors:
                trip = dup + x
                go_counts[x] = int(go_t[f"trip_{trip}_count"].sum())
                ng_counts[x] = int(ng_t[f"trip_{trip}_count"].sum())

            go_total = sum(go_counts.values())
            ng_total = sum(ng_counts.values())

            if go_total == 0 and ng_total == 0:
                continue

            pvals, test_rows = [], []
            for x in successors:
                trip = dup + x
                go_n = go_counts[x]
                ng_n = ng_counts[x]
                go_other = go_total - go_n
                ng_other = ng_total - ng_n

                if go_n > 0 and ng_other > 0:
                    OR = (ng_n * go_other) / (go_n * ng_other)
                else:
                    OR = np.nan

                table = np.array([[ng_n, ng_other], [go_n, go_other]])
                _, p_one = fisher_exact(table, alternative="greater")

                go_pct = go_n / go_total * 100 if go_total > 0 else 0
                ng_pct = ng_n / ng_total * 100 if ng_total > 0 else 0

                pvals.append(p_one)
                test_rows.append({
                    "Window": wname, "Duplet": dup, "Triplet": trip,
                    "GO_n": go_n, "GO_total": go_total,
                    "GO_%": round(go_pct, 1),
                    "NG_n": ng_n, "NG_total": ng_total,
                    "NG_%": round(ng_pct, 1),
                    "Diff_%": round(ng_pct - go_pct, 1),
                    "OR": round(OR, 3) if not np.isnan(OR) else np.nan,
                    "p_raw": round(p_one, 6),
                })

            valid_p = [p for p in pvals if not np.isnan(p)]
            if len(valid_p) > 1:
                _, pfdr, _, _ = multipletests(valid_p, method="fdr_bh")
                j = 0
                for tr in test_rows:
                    if not np.isnan(tr["p_raw"]):
                        tr["p_FDR"] = round(pfdr[j], 6)
                        j += 1
                    else:
                        tr["p_FDR"] = np.nan
            elif len(valid_p) == 1:
                test_rows[0]["p_FDR"] = test_rows[0]["p_raw"]

            for tr in test_rows:
                tr["Sig"] = ("*" if tr.get("p_FDR", 1) < ALPHA else "ns")
                tr["N_tests"] = len(valid_p)

            results.extend(test_rows)

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
# E-A, E-A-D, D FREQUENCY (Mann-Whitney; r from U, see mwu_effect_r_u)
# ═══════════════════════════════════════════════════════════════════════════════

def ea_ead_dfreq_tests(tdf: pd.DataFrame) -> pd.DataFrame:
    """
    E-A rate = EA count / collapsed length (the bigram rate of the original
    code); E-A-D rate = EAD count / collapsed length (same denominator);
    D frequency = D count / collapsed length (D_freq of the original code).
    Mann-Whitney U two-sided, trial level and subject level (mean over the 20
    trials of each participant). The effect size r uses Z from U without tie
    correction (mwu_effect_r_u); the bigram rates of S3 use Z from the
    two-sided p (mwu_effect_r).
    """
    tdf = tdf.copy()
    L = tdf["seq_len"].where(tdf["seq_len"] > 0, np.nan)
    tdf["EA_rate"] = tdf["dup_EA_rate"]
    tdf["EAD_rate"] = (tdf["trip_EAD_count"] / L).fillna(0)
    measures = [("E-A rate", "EA_rate"), ("E-A-D rate", "EAD_rate"),
                ("D frequency", "D_freq")]

    rows = []
    for wname in WINDOWS:
        wd = tdf[tdf["window"] == wname]
        for label, col in measures:
            for level in ("trial", "subject"):
                if level == "trial":
                    go_vals = wd.loc[wd["condition"] == "GO", col].values
                    ng_vals = wd.loc[wd["condition"] == "NG", col].values
                else:
                    go_vals = (wd[wd["condition"] == "GO"]
                               .groupby("subject")[col].mean()
                               .sort_index().values)
                    ng_vals = (wd[wd["condition"] == "NG"]
                               .groupby("subject")[col].mean()
                               .sort_index().values)
                U, p, r = mwu_effect_r_u(go_vals, ng_vals)
                rows.append({
                    "Window": wname, "Measure": label, "Level": level,
                    "N_GO": len(go_vals), "N_NG": len(ng_vals),
                    "GO_mean": round(go_vals.mean(), 6),
                    "GO_sd": round(go_vals.std(), 6),
                    "NG_mean": round(ng_vals.mean(), 6),
                    "NG_sd": round(ng_vals.std(), 6),
                    "U": round(U, 1), "p": round(p, 4), "r": round(r, 3),
                })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# E-A-X FISHER TABLE (FDR with m = 4 as in the Methods, and m = 5)
# ═══════════════════════════════════════════════════════════════════════════════

def eax_fisher(tdf: pd.DataFrame) -> pd.DataFrame:
    """
    Same Fisher test as level 3 for the bigram EA and the five possible third
    states X (A..E). m = 4: BH over X != A (the family of level 3).
    m = 5: BH over all five X, EAA included (EAA cannot occur in collapsed
    sequences; its Fisher p is computed from its zero counts).
    """
    rows = []
    for wname in WINDOWS:
        wd = tdf[tdf["window"] == wname]
        go_t = wd[wd["condition"] == "GO"]
        ng_t = wd[wd["condition"] == "NG"]
        go_counts = {x: int(go_t[f"trip_EA{x}_count"].sum()) for x in STATES}
        ng_counts = {x: int(ng_t[f"trip_EA{x}_count"].sum()) for x in STATES}
        go_total = sum(go_counts.values())
        ng_total = sum(ng_counts.values())

        block = []
        for x in STATES:
            go_n, ng_n = go_counts[x], ng_counts[x]
            go_other, ng_other = go_total - go_n, ng_total - ng_n
            if go_n > 0 and ng_other > 0:
                OR = (ng_n * go_other) / (go_n * ng_other)
            else:
                OR = np.nan
            _, p = fisher_exact(np.array([[ng_n, ng_other], [go_n, go_other]]),
                                alternative="greater")
            block.append({
                "Window": wname, "Triplet": "EA" + x,
                "GO_n": go_n, "GO_total": go_total,
                "GO_%": round(go_n / go_total * 100, 1) if go_total else 0,
                "NG_n": ng_n, "NG_total": ng_total,
                "NG_%": round(ng_n / ng_total * 100, 1) if ng_total else 0,
                "OR": round(OR, 3) if not np.isnan(OR) else np.nan,
                "p_raw": round(p, 6), "_p": p,
            })
        p5 = [b["_p"] for b in block]
        _, f5, _, _ = multipletests(p5, method="fdr_bh")
        p4 = [b["_p"] for b in block if b["Triplet"] != "EAA"]
        _, f4, _, _ = multipletests(p4, method="fdr_bh")
        j = 0
        for b, v5 in zip(block, f5):
            b["p_FDR_m5"] = round(v5, 6)
            if b["Triplet"] != "EAA":
                b["p_FDR_m4"] = round(f4[j], 6)
                j += 1
            else:
                b["p_FDR_m4"] = np.nan
            del b["_p"]
        rows.extend(block)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT LAYOUTS
# ═══════════════════════════════════════════════════════════════════════════════

def s2_layout(l1: pd.DataFrame, k: int) -> pd.DataFrame:
    """S2 layout: one table per k, windows stacked, sorted by Diff (%) desc."""
    sub = l1[l1["k"] == k].copy()
    sub["_w"] = sub["Window"].map({w: i for i, w in enumerate(WINDOWS)})
    sub = sub.sort_values(["_w", "Diff_%"], ascending=[True, False],
                          kind="mergesort")
    out = pd.DataFrame({
        "Window": sub["Window"].map(WINDOW_LABELS),
        "Motif": sub["Motif"],
        "GO (n)": sub["GO_n"], "NoGo (n)": sub["NG_n"],
        "GO (%)": sub["GO_%"], "NoGo (%)": sub["NG_%"],
        "Diff (%)": sub["Diff_%"], "OR": sub["OR"],
        "p (raw)": sub["p_raw"], "p (FDR)": sub["p_FDR"], "Sig": sub["Sig"],
        "Cohen_d": sub["Cohen_d"],
    })
    return out.reset_index(drop=True)


def s2_summary(l1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in (3, 4, 5):
        for wname in WINDOWS:
            sub = l1[(l1["k"] == k) & (l1["Window"] == wname)]
            rows.append({"k": k, "Window": WINDOW_LABELS[wname],
                         "Motifs tested (>=5 combined occurrences)": len(sub),
                         "Significant (p_FDR<0.05)": int((sub["Sig"] == "*").sum()),
                         "Raw p<0.05": int((sub["p_raw"] < 0.05).sum())})
    return pd.DataFrame(rows)


def s3_layout(l2: pd.DataFrame) -> pd.DataFrame:
    out = l2.rename(columns={"GO_mean": "GO mean", "GO_sd": "GO sd",
                             "NG_mean": "NG mean", "NG_sd": "NG sd"})
    return out[["Window", "Duplet", "Level", "N_GO", "N_NG", "GO mean",
                "GO sd", "NG mean", "NG sd", "U", "p", "r", "Sig",
                f"p_BH ({ADDITIONAL})", f"Sig_BH ({ADDITIONAL})"]]


def s4_layout(l3: pd.DataFrame) -> pd.DataFrame:
    """S4 layout (percentages as text). Sorted by Duplet, p_raw per window."""
    sub = l3.copy()
    sub["_w"] = sub["Window"].map({w: i for i, w in enumerate(WINDOWS)})
    sub = sub.sort_values(["_w", "Duplet", "p_raw"], kind="mergesort")
    return pd.DataFrame({
        "Window": sub["Window"],
        "Duplet": sub["Duplet"], "Triplet": sub["Triplet"],
        "GO n": sub["GO_n"], "GO total": sub["GO_total"],
        "GO %": sub["GO_%"].map(lambda v: f"{v:.1f}%"),
        "NG n": sub["NG_n"], "NG total": sub["NG_total"],
        "NG %": sub["NG_%"].map(lambda v: f"{v:.1f}%"),
        "Diff %": sub["Diff_%"].map(lambda v: f"{v:+.1f}%"),
        "OR": sub["OR"], "p (raw)": sub["p_raw"], "p (FDR)": sub["p_FDR"],
        "Sig": sub["Sig"],
    }).reset_index(drop=True)


def table3(l3: pd.DataFrame) -> pd.DataFrame:
    """Table 3 filter of the original code: early window, Sig == *."""
    early_sig = (l3[(l3["Window"] == "early") & (l3["Sig"] == "*")]
                 .sort_values("p_raw", kind="mergesort").copy())
    cols = ["Duplet", "Triplet", "GO_n", "GO_total", "GO_%",
            "NG_n", "NG_total", "NG_%", "Diff_%", "OR",
            "p_raw", "p_FDR", "Sig"]
    return early_sig[cols].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--windows", choices=["exact", "proportional"],
                        default="exact",
                        help="window source (default: exact)")
    parser.add_argument("--source", choices=["eeg", "deposited"],
                        default="deposited",
                        help="proportional mode only: table built from the EEG "
                             "(03_sequences/full_prepost.py) or the deposited "
                             "file (default: deposited)")
    args = parser.parse_args()
    mode = args.windows

    tab_dir = config.get_out_dir() / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)

    df = load_trials(mode, args.source)
    print(f"windows: {mode}"
          + (f" (source: {args.source})" if mode == "proportional" else "")
          + f"; {len(df)} trials, "
          f"{df['subject'].nunique()} subjects")
    for w in WINDOWS:
        lens = df[w].str.len()
        print(f"  collapsed length {w}: min {lens.min()}, "
              f"median {int(lens.median())}, max {lens.max()}")

    tdf = build_trial_table(df)
    l1 = level1_kmer_screen(df)
    l2 = level2_bigram_rates(tdf)
    l3 = level3_conditional(tdf)

    for k in (3, 4, 5):
        s2_layout(l1, k).to_csv(tab_dir / f"S2_kmer_screen_k{k}_{mode}.csv",
                                index=False)
    s2_summary(l1).to_csv(tab_dir / f"S2_kmer_summary_{mode}.csv", index=False)
    s3_layout(l2).to_csv(tab_dir / f"S3_bigram_rates_{mode}.csv", index=False)
    s4_layout(l3).to_csv(tab_dir / f"S4_conditional_successors_{mode}.csv",
                         index=False)
    table3(l3).to_csv(
        tab_dir / f"Table3_significant_conditional_early_{mode}.csv",
        index=False)
    ea = ea_ead_dfreq_tests(tdf)
    ea.to_csv(tab_dir / f"EA_EAD_Dfreq_tests_{mode}.csv", index=False)
    eax = eax_fisher(tdf)
    eax.to_csv(tab_dir / f"EAX_fisher_{mode}.csv", index=False)

    print(s2_summary(l1).to_string(index=False))
    for w in WINDOWS:
        sub = l3[l3["Window"] == w]
        print(f"conditional {w}: {(sub['Sig'] == '*').sum()}/{len(sub)} "
              f"FDR significant")
    print(f"Saved to tables/ in MICROSTATE_OUT_DIR (suffix _{mode})")


if __name__ == "__main__":
    main()
