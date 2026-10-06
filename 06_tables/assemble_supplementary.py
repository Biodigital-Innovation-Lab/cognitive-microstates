"""
assemble_supplementary.py -- Supplementary Tables S1-S5 as xlsx and CSV.

Purpose : collect the tables written by the earlier steps into the five
          supplementary workbooks, with the sheet and column names of the
          deposited tables. Exact windows are used throughout (S2, S3, S4).
          All tables use the labels from the K=5 reference templates (the
          K=5 solution of the October 2025 clustering run; the only template
          set).
            S1  microstate metrics per trial and state (reference templates;
                sex from config.SEX_MAP)
            S2  k-mer screen: Summary, 3-mers, 4-mers, 5-mers
            S3  bigram rates: 6 sheets (whole, early, late epoch; subject and
                trial level); column "p (FDR)" after "p" = Benjamini-Hochberg
                adjusted p over the 20 bigrams of the sheet (window x level),
                as computed by motif_analysis.py (6 decimals)
            S4  conditional successors: 3 sheets (full, early, late epoch)
            S5  sequences by trial (reference templates; sex from
                config.SEX_MAP) and the consensus by subject: 3 sheets
          Labels: sex Male/Female; condition GO/NG; window full/early/late
          (long window names in S2 and S4 sheet titles as in the deposited
          tables). Each sheet is also written as one CSV (header in row 1, no
          title rows). Numbers are written as produced by the earlier steps
          (no further rounding). The workbook files carry no author metadata
          and are written byte-reproducibly.
Usage   : python assemble_supplementary.py
Inputs  : MICROSTATE_OUT_DIR/tables/
              microstate_metrics_K5_reference.csv      (metrics.py)
              S2_kmer_screen_k{3,4,5}_exact.csv, S2_kmer_summary_exact.csv,
              S3_bigram_rates_exact.csv, S4_conditional_successors_exact.csv
                                                       (motif_analysis.py)
              S5_trial_sequences_reference.csv         (trial_table.py)
              msa_consensus_SUBJECT.csv                (consensus.py)
Outputs : MICROSTATE_OUT_DIR/supplementary/
              S1_Microstate_Metrics.xlsx, S2_Collapsed_Sequences.xlsx,
              S3_Bigram_Transition_Rates.xlsx,
              S4_Conditional_Successor_Statistics.xlsx,
              S5_MSA_Alignment_Scores.xlsx
              <workbook>__<sheet>.csv   one per sheet
Environment : Python 3.11.14; pandas 2.0.3, openpyxl 3.1.5; see
              requirements.txt. No random numbers are used.
"""

import io
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

FIXED_DATE = datetime(2000, 1, 1)
ZIP_DATE = (1980, 1, 1, 0, 0, 0)

WINDOW_LONG = {"full": "Full epoch (0–1300 ms)",
               "early": "Early phase (100–500 ms)",
               "late": "Late phase (500–900 ms)"}

S2_NOTE = ("Note: Denominator = total k-mer occurrences per condition per "
           "window. Minimum occurrence threshold: ≥5 combined (GO+NoGo). FDR "
           "correction applied independently per window. OR > 1 = enriched in "
           "NoGo; OR < 1 = enriched in Go.")
S2_TITLE = ("Supplementary Table — {k}-mer Motif Analysis  |  EEG Go/NoGo "
            "Task  |  30 subjects × 20 trials/condition  |  Fisher's exact "
            "test (one-sided, NG enrichment)  +  FDR (Benjamini-Hochberg)")

S2_COLUMNS = ["Window", "Motif", "GO (n)", "NoGo (n)", "GO (%)", "NoGo (%)",
              "Diff (%)", "OR", "p (raw)", "p (FDR)", "Sig"]
S3_COLUMNS = ["Duplet", "Level", "GO mean", "GO sd", "NG mean", "NG sd",
              "U", "p", "p (FDR)", "r", "Sig"]
S3_BH_SOURCE = "p_BH (additional, not in the original analysis)"
S4_COLUMNS = ["Duplet", "Triplet", "GO n", "GO total", "GO %", "NG n",
              "NG total", "NG %", "Diff %", "OR", "p (raw)", "p (FDR)", "Sig"]
S5_TRIAL_COLUMNS = ["subject", "sex", "condition", "trial", "sequence_full",
                    "sequence_length", "sequence_collapsed",
                    "sequence_collapsed_length", "early_full",
                    "early_collapsed", "late_full", "late_collapsed"]
S5_SUBJECT_COLUMNS = ["subject", "condition", "window", "n_trials", "length",
                      "mean_support", "consensus"]
S1_COLUMNS = ["subject", "sex", "condition", "trial", "state", "coverage",
              "lifespan", "occurrences"]


def native(value):
    """numpy scalar -> python value; NaN -> None; whole-number floats stay float."""
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def write_workbook(path, sheets):
    """
    sheets: list of (name, preamble_rows, DataFrame). The workbook is written
    to memory and re-zipped with fixed timestamps so that repeated runs give
    identical files; creator and dates are fixed (no author metadata).
    """
    wb = Workbook()
    wb.remove(wb.active)
    wb.properties.creator = None
    wb.properties.lastModifiedBy = None
    wb.properties.created = FIXED_DATE
    wb.properties.modified = FIXED_DATE
    for name, preamble, df in sheets:
        ws = wb.create_sheet(name)
        for row in preamble:
            ws.append(row)
        ws.append(list(df.columns))
        for rec in df.itertuples(index=False, name=None):
            ws.append([native(v) for v in rec])
        ws.freeze_panes = ws.cell(row=len(preamble) + 2, column=1)
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    with zipfile.ZipFile(buf) as zin, \
            zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            info = zipfile.ZipInfo(item.filename, date_time=ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            data = zin.read(item.filename)
            if item.filename == "docProps/core.xml":
                # openpyxl stamps the save time into dcterms:modified
                data = re.sub(rb"(<dcterms:modified[^>]*>)[^<]*(<)",
                              rb"\g<1>2000-01-01T00:00:00Z\g<2>", data)
            zout.writestr(info, data)


def write_csvs(out_dir, stem, sheets):
    for name, _, df in sheets:
        fname = f"{stem}__{name.replace(' ', '_')}.csv"
        df.to_csv(out_dir / fname, index=False)


def read(tab_dir, name, **kwargs):
    path = tab_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run the earlier steps first")
    return pd.read_csv(path, **kwargs)


def build_s1(tab_dir, sex_map):
    df = read(tab_dir, "microstate_metrics_K5_reference.csv",
              dtype={"file_prefix": str, "condition": str})
    df["sex"] = df["file_prefix"].map(sex_map)
    assert df["sex"].notna().all()
    return [("microstate_metrics_K5", [], df[S1_COLUMNS])]


def build_s2(tab_dir):
    summary = read(tab_dir, "S2_kmer_summary_exact.csv")
    sheets, rows = [], []
    for k in (3, 4, 5):
        df = read(tab_dir, f"S2_kmer_screen_k{k}_exact.csv")[S2_COLUMNS]
        sheets.append((f"{k}-mers",
                       [[S2_TITLE.format(k=k)], [S2_NOTE]], df))
        for window in WINDOW_LONG.values():
            top = df[df["Window"] == window].iloc[0]   # sorted by Diff (%)
            tested = summary[(summary["k"] == k) & (summary["Window"] == window)]
            assert len(tested) == 1
            rows.append([f"{k}-mers", k, window, 5 ** k,
                         int(tested["Motifs tested (>=5 combined occurrences)"].iloc[0]),
                         int(tested["Significant (p_FDR<0.05)"].iloc[0]),
                         f"{top['Motif']} ({top['Diff (%)']:+.2f}%)",
                         round(float(top["p (FDR)"]), 4)])
    summary_df = pd.DataFrame(rows, columns=[
        "Sheet", "k", "Window", "Total motifs tested", "Motifs ≥5 occurrences",
        "Significant (p_FDR<0.05)", "Top motif (Diff %)", "p_FDR top"])
    return [("Summary",
             [["Supplementary Table — k-mer Motif Analysis Summary"]],
             summary_df)] + sheets


def build_s3(tab_dir):
    df = read(tab_dir, "S3_bigram_rates_exact.csv")
    df["U"] = df["U"].map(lambda u: int(u) if float(u).is_integer() else u)
    df["p (FDR)"] = df[S3_BH_SOURCE]
    sheets = []
    for window, label in (("full", "Whole Epoch"), ("early", "Early Epoch"),
                          ("late", "Late Epoch")):
        for level, suffix in (("subject", "Subject"), ("trial", "Trials")):
            sub = df[(df["Window"] == window) & (df["Level"] == level)]
            assert len(sub) == 20
            sheets.append((f"{label} Duplets {suffix}", [], sub[S3_COLUMNS]))
    return sheets


def build_s4(tab_dir):
    df = read(tab_dir, "S4_conditional_successors_exact.csv")
    sheets = []
    for window, label in (("full", "Full"), ("early", "Early"), ("late", "Late")):
        sub = df[df["Window"] == window]
        assert len(sub) == 80
        sheets.append((f"{label} Epoch Triplets", [], sub[S4_COLUMNS]))
    return sheets


def build_s5(tab_dir, sex_map):
    trials = read(tab_dir, "S5_trial_sequences_reference.csv",
                  dtype={"file_prefix": str, "condition": str})
    trials["sex"] = trials["file_prefix"].map(sex_map)
    assert trials["sex"].notna().all()
    cons = read(tab_dir, "msa_consensus_SUBJECT.csv")
    sheets = [("Microstate Sequences by Trial", [], trials[S5_TRIAL_COLUMNS])]
    for window, label in (("full", "Full"), ("early", "Early"), ("late", "Late")):
        sub = cons[cons["window"] == window][S5_SUBJECT_COLUMNS]
        assert len(sub) == 60
        sheets.append((f"{label} Epoch Sequence by Subject", [], sub))
    return sheets


def main():
    out_dir = config.get_out_dir()
    tab_dir = out_dir / "tables"
    sup_dir = out_dir / "supplementary"
    sup_dir.mkdir(parents=True, exist_ok=True)
    sex_map = config.get_sex_map()

    workbooks = {
        "S1_Microstate_Metrics": build_s1(tab_dir, sex_map),
        "S2_Collapsed_Sequences": build_s2(tab_dir),
        "S3_Bigram_Transition_Rates": build_s3(tab_dir),
        "S4_Conditional_Successor_Statistics": build_s4(tab_dir),
        "S5_MSA_Alignment_Scores": build_s5(tab_dir, sex_map),
    }
    for stem, sheets in workbooks.items():
        write_workbook(sup_dir / f"{stem}.xlsx", sheets)
        write_csvs(sup_dir, stem, sheets)
        print(f"{stem}.xlsx: {len(sheets)} sheets, "
              f"{sum(len(df) for _, _, df in sheets)} data rows")
    print("Saved to supplementary/ in MICROSTATE_OUT_DIR")


if __name__ == "__main__":
    main()
