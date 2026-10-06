"""
consensus.py -- per-participant consensus sequences (30 x 2 conditions x 3
windows = 180 rows).

Purpose : for every participant, condition and window, build one consensus
          string from the participant's 20 collapsed trial strings.
          Procedure:
            1. strings = the 20 collapsed strings of the cell (empty strings
               are dropped).
            2. reference = the string at index len(strings) // 2 of the
               strings sorted by length (stable sort; the sorted string at
               that index, i.e. the 11th shortest of 20).
            3. every string (in table order) is aligned to the reference with
               a global pairwise alignment (Biopython PairwiseAligner;
               match +1, mismatch -1, gap open -2, gap extend -0.5); the first
               alignment returned is used; query letters are placed on the
               reference coordinates and gaps ("-") are inserted where the
               reference has letters the query lacks.
            4. each aligned column is voted over its non-gap letters
               (collections.Counter.most_common, first letter encountered wins
               ties). A column with only gaps is skipped. The support of a
               column is the winner's count divided by the number of strings
               (gaps included in the denominator).
            5. consensus = the winners in column order; length = len(consensus);
               mean_support = mean column support rounded to 6 decimals.
          Windows: full = sequence_collapsed; early = early_collapsed
          (samples 50-249); late = late_collapsed (samples 250-449); the
          window strings come from the exact-window per-trial table.
          Re-implemented from an earlier script and verified against the
          deposited table.
Usage   : python consensus.py
Inputs  : MICROSTATE_OUT_DIR/tables/S5_trial_sequences_reference.csv
              (from 03_sequences/trial_table.py; columns used: subject,
              condition, sequence_collapsed, early_collapsed, late_collapsed)
Outputs : MICROSTATE_OUT_DIR/tables/msa_consensus_SUBJECT.csv
              subject, condition (GO|NG), window (full|early|late), n_trials,
              length, mean_support, consensus. Row order: window (full, early,
              late), then condition (GO, NG), then subject.
          No sex column, no participant initials.
Environment : Python 3.11.14; biopython 1.81, pandas 2.0.3; see
              requirements.txt. No random numbers are used.
"""

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from Bio import Align

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

WINDOWS = {
    "full": "sequence_collapsed",
    "early": "early_collapsed",
    "late": "late_collapsed",
}

aligner = Align.PairwiseAligner()
aligner.mode = "global"
aligner.match_score = 1
aligner.mismatch_score = -1
aligner.open_gap_score = -2
aligner.extend_gap_score = -0.5


def align_to_reference(ref, query):
    """Query placed on the reference coordinates, gaps inserted as '-'."""
    alignment = next(iter(aligner.align(ref, query)))
    ref_blocks, query_blocks = alignment.aligned[0], alignment.aligned[1]
    out, ref_pos = [], 0
    for (r_start, r_end), (q_start, q_end) in zip(ref_blocks, query_blocks):
        if r_start > ref_pos:
            out.extend(["-"] * (r_start - ref_pos))
        out.extend(list(query[q_start:q_end]))
        ref_pos = r_end
    if ref_pos < len(ref):
        out.extend(["-"] * (len(ref) - ref_pos))
    return "".join(out)


def build_consensus(sequences):
    """Return (consensus string, mean support)."""
    seqs = [s for s in sequences if isinstance(s, str) and len(s) > 0]
    if not seqs:
        return "", 0.0
    reference = sorted(seqs, key=len)[len(seqs) // 2]
    aligned = []
    for seq in seqs:
        try:
            aligned.append(align_to_reference(reference, seq))
        except Exception:
            aligned.append(seq)

    consensus, supports = [], []
    for i in range(max(len(s) for s in aligned)):
        chars = [s[i] if i < len(s) else "-" for s in aligned]
        non_gap = [c for c in chars if c != "-"]
        if not non_gap:
            continue
        best, count = Counter(non_gap).most_common(1)[0]
        consensus.append(best)
        supports.append(count / len(chars))
    mean_support = sum(supports) / len(supports) if supports else 0.0
    return "".join(consensus), mean_support


def main():
    tab_dir = config.get_out_dir() / "tables"
    src = tab_dir / "S5_trial_sequences_reference.csv"
    if not src.is_file():
        raise FileNotFoundError(
            f"{src} not found; run 03_sequences/trial_table.py")
    df = pd.read_csv(src, usecols=["subject", "condition",
                                   *WINDOWS.values()],
                     dtype={"condition": str})

    rows = []
    for window, col in WINDOWS.items():
        for cond in config.CONDITIONS:
            for subj in sorted(df["subject"].unique()):
                seqs = df.loc[(df["subject"] == subj)
                              & (df["condition"] == cond), col].tolist()
                cons, support = build_consensus(seqs)
                rows.append({"subject": subj, "condition": cond,
                             "window": window, "n_trials": len(seqs),
                             "length": len(cons),
                             "mean_support": round(support, 6),
                             "consensus": cons})
    out = pd.DataFrame(rows)
    out.to_csv(tab_dir / "msa_consensus_SUBJECT.csv", index=False)
    print(f"rows: {len(out)}; saved to tables/msa_consensus_SUBJECT.csv "
          "in MICROSTATE_OUT_DIR")


if __name__ == "__main__":
    main()
