"""
canonical_similarity.py -- spatial similarity of the K=5 microstate maps to
canonical reference maps.

Purpose : polarity-invariant spatial correlation (|r| x 100, with signed r kept)
          between the five empirical maps (A-E) and three canonical reference
          sets (four maps A-D, four maps A-D, five maps A-E), restricted to the
          common 19-channel montage and spatially mean-centred; best match per
          empirical map; one-to-one (Hungarian) assignment; unassigned maps.
          Adapted from a collaborator's canonical-similarity script and
          verified against the reported values. Plotting, PDF and ZIP parts of
          that script are not included.
Reference maps (NOT distributed; obtain them from the original sources):
          Koenig, T., Prichep, L., Lehmann, D., Sosa, P. V., Braeker, E.,
            Kleinlogel, H., Isenhart, R., & John, E. R. (2002). Millisecond by
            millisecond, year by year: Normative EEG microstates and
            developmental stages. NeuroImage, 16(1), 41-48.
            https://doi.org/10.1006/nimg.2002.1070
          Milz, P., Faber, P., Lehmann, D., Koenig, T., Kochi, K., &
            Pascual-Marqui, R. D. (2016). The functional significance of EEG
            microstates - Associations with modalities of thinking.
            NeuroImage, 125, 643-656.
            https://doi.org/10.1016/j.neuroimage.2015.08.023
          Artoni, F., Merk, T., Schneider, J., Tiber, V., Sheller, J.,
            Blankertz, B., & König, P. (2023). Microstate analysis of
            resting-state EEG: A standardized template for large-scale data.
            NeuroImage, 277, 120196. (verify against the publisher record)
            https://doi.org/10.1016/j.neuroimage.2023.120196
Usage   : python canonical_similarity.py
Inputs  : MICROSTATE_REFERENCE_MAPS_DIR  folder with the reference-map files
              mean_models_koenig_et_al_2002.asc
              mean_models_koenig_et_al_2002_chlist.asc
              mean_models_milz_et_al_2016.asc
              mean_models_milz_et_al_2016_chlist.asc
              Artoni 2023.set
          MICROSTATE_REFERENCE_TEMPLATES   .npz with key 'templates' (5, 19),
              rows A-E, and key 'ch_names' (19,) giving the channel order
              (read with allow_pickle=False). If 'ch_names' is absent, the
              channel names are read from MICROSTATE_OUT_DIR/eeg_trials.npz
              (key 'ch_names'; from 01_reshape/reshape.py).
Outputs : MICROSTATE_OUT_DIR/canonical/   (CSV only)
              <Set>_absolute_similarity_percent.csv   |r| x 100, 5 x n_maps
              <Set>_signed_spatial_correlations.csv   signed r
              <Set>_best_match_assignments.csv        best and second best
              <Set>_hungarian_assignments.csv         one-to-one assignment
              unassigned_maps.csv                     empirical maps left
                                                      unassigned, per set
              combined_best_match_summary.csv         best matches, all sets
              (<Set> = Koenig_2002, Milz_2016, Artoni_2023)
Environment : Python 3.11.14; numpy 1.25.2, pandas 2.0.3, scipy 1.11.4; see
              requirements.txt. No random numbers are used.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

TARGET_19_CHANNELS = [
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2",
]

KOENIG_MAPS = "mean_models_koenig_et_al_2002.asc"
KOENIG_CHANNELS = "mean_models_koenig_et_al_2002_chlist.asc"
MILZ_MAPS = "mean_models_milz_et_al_2016.asc"
MILZ_CHANNELS = "mean_models_milz_et_al_2016_chlist.asc"
ARTONI_SET = "Artoni 2023.set"


@dataclass
class TemplateConfig:
    """Configuration for one canonical/reference template set."""

    name: str
    short_name: str
    maps_file: Path
    channels_file: Optional[Path] = None
    file_type: str = "ascii"  # "ascii" or "eeglab_set"


# -----------------------------------------------------------------------------
# Channel-name handling
# -----------------------------------------------------------------------------


def clean_string_label(x: object) -> str:
    """Convert MATLAB/NumPy labels into clean Python strings."""
    if isinstance(x, bytes):
        s = x.decode("utf-8")
    else:
        s = str(x)
    return s.strip().strip("'").strip('"').strip()


def standardize_channel_name(ch: str, *, map_f1f2_to_fp: bool = False) -> str:
    """Standardize EEG channel names to MNE/10-20-style labels."""
    ch = clean_string_label(ch)
    ch_upper = ch.upper()

    mapping = {
        "FP1": "Fp1",
        "FP2": "Fp2",
        "FPZ": "Fpz",
        "FZ": "Fz",
        "CZ": "Cz",
        "PZ": "Pz",
        "OZ": "Oz",
        "AFZ": "AFz",
        "FCZ": "FCz",
        "CPZ": "CPz",
        "POZ": "POz",
        # Old 10-20 temporal labels -> modern equivalents
        "T3": "T7",
        "T4": "T8",
        "T5": "P7",
        "T6": "P8",
    }

    if map_f1f2_to_fp:
        mapping.update({"F1": "Fp1", "F2": "Fp2"})

    if ch_upper in mapping:
        return mapping[ch_upper]
    return ch_upper


# -----------------------------------------------------------------------------
# Core spatial-correlation functions
# -----------------------------------------------------------------------------


def signed_spatial_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Signed Pearson spatial correlation (maps spatially mean-centred)."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den == 0:
        return 0.0
    return float(np.dot(a, b) / den)


def abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Polarity-invariant spatial correlation |r|."""
    return abs(signed_spatial_corr(a, b))


def classify_similarity(value_percent: float) -> str:
    """Descriptive interpretation of |r| x 100."""
    if value_percent >= 80:
        return "high"
    if value_percent >= 60:
        return "moderate"
    if value_percent >= 40:
        return "low/ambiguous"
    return "low"


# -----------------------------------------------------------------------------
# Loading empirical and canonical maps
# -----------------------------------------------------------------------------


def load_empirical_microstates(templates_path, ch_names) -> pd.DataFrame:
    """
    Empirical maps as a DataFrame (rows MS1-MS5, columns standardized channel
    labels). Only the 'templates' key of the npz is read.
    """
    templates = np.asarray(
        np.load(templates_path, allow_pickle=False)["templates"], dtype=float)
    channels = [standardize_channel_name(c, map_f1f2_to_fp=True) for c in ch_names]

    # Orient as maps x channels.
    if templates.shape[0] == len(channels):
        maps = templates.T
    elif templates.shape[1] == len(channels):
        maps = templates
    else:
        raise ValueError(
            f"Cannot infer empirical map orientation: templates shape {templates.shape}, "
            f"number of channels {len(channels)}."
        )

    labels = [f"Empirical MS{i + 1}" for i in range(maps.shape[0])]
    return pd.DataFrame(maps, index=labels, columns=channels)


def load_ascii_template(cfg: TemplateConfig) -> pd.DataFrame:
    """Canonical template from .asc maps and .asc channel list."""
    maps_raw = np.loadtxt(cfg.maps_file)
    with open(cfg.channels_file, "r", encoding="utf-8") as f:
        channels_raw = f.read().strip().split()

    # For canonical templates, do NOT map true F1/F2 to Fp1/Fp2.
    channels = [standardize_channel_name(c, map_f1f2_to_fp=False) for c in channels_raw]

    if maps_raw.shape[1] == len(channels):
        maps = maps_raw
    elif maps_raw.shape[0] == len(channels):
        maps = maps_raw.T
    else:
        raise ValueError(
            f"Cannot infer orientation for {cfg.name}: maps shape {maps_raw.shape}, "
            f"number of channels {len(channels)}."
        )

    if maps.shape[0] in (4, 5):
        labels = [f"{cfg.short_name} {c}" for c in "ABCDE"[:maps.shape[0]]]
    else:
        labels = [f"{cfg.short_name} {i + 1}" for i in range(maps.shape[0])]
    return pd.DataFrame(maps, index=labels, columns=channels)


def _extract_matlab_chanloc_labels(chanlocs: object) -> List[str]:
    """Labels from an EEGLAB chanlocs structure loaded via scipy.io.loadmat."""
    labels: List[str] = []
    if isinstance(chanlocs, np.ndarray):
        iterable = chanlocs.ravel()
    else:
        iterable = [chanlocs]

    for item in iterable:
        if hasattr(item, "labels"):
            labels.append(clean_string_label(getattr(item, "labels")))
        elif isinstance(item, np.void) and "labels" in item.dtype.names:
            labels.append(clean_string_label(item["labels"]))
        else:
            labels.append(clean_string_label(item))
    return labels


def load_eeglab_set_template(cfg: TemplateConfig) -> pd.DataFrame:
    """Canonical template from an EEGLAB .set file (EEG.data, EEG.chanlocs)."""
    mat = loadmat(cfg.maps_file, squeeze_me=True, struct_as_record=False)
    if "EEG" not in mat:
        raise KeyError(f"{cfg.maps_file} does not contain an EEG structure.")

    eeg = mat["EEG"]
    if not hasattr(eeg, "data") or not hasattr(eeg, "chanlocs"):
        raise KeyError(f"{cfg.maps_file} EEG structure lacks data or chanlocs.")

    maps_raw = np.asarray(getattr(eeg, "data"), dtype=float)
    channels_raw = _extract_matlab_chanloc_labels(getattr(eeg, "chanlocs"))

    # Do NOT map F1/F2 to Fp1/Fp2: this set has true F1/F2 and separate Fp1/Fp2.
    channels = [standardize_channel_name(c, map_f1f2_to_fp=False) for c in channels_raw]

    if maps_raw.shape[0] == len(channels):
        maps = maps_raw.T
    elif maps_raw.shape[1] == len(channels):
        maps = maps_raw
    else:
        raise ValueError(
            f"Cannot infer orientation for {cfg.name}: maps shape {maps_raw.shape}, "
            f"number of channels {len(channels)}."
        )

    if maps.shape[0] in (4, 5):
        labels = [f"{cfg.short_name} {c}" for c in "ABCDE"[:maps.shape[0]]]
    else:
        labels = [f"{cfg.short_name} {i + 1}" for i in range(maps.shape[0])]
    return pd.DataFrame(maps, index=labels, columns=channels)


def load_template(cfg: TemplateConfig) -> pd.DataFrame:
    if cfg.file_type == "ascii":
        return load_ascii_template(cfg)
    if cfg.file_type == "eeglab_set":
        return load_eeglab_set_template(cfg)
    raise ValueError(f"Unsupported template file_type: {cfg.file_type}")


# -----------------------------------------------------------------------------
# Similarity analysis and assignments
# -----------------------------------------------------------------------------


def compute_empirical_vs_canonical(
    empirical_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
    target_channels: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Empirical x canonical matrices: |r|, signed r, |r| x 100."""
    missing_empirical = [ch for ch in target_channels if ch not in empirical_df.columns]
    missing_canonical = [ch for ch in target_channels if ch not in canonical_df.columns]

    if missing_empirical or missing_canonical:
        raise ValueError(
            "Channel mismatch for common 19-channel comparison.\n"
            f"Missing empirical: {missing_empirical}\n"
            f"Missing canonical: {missing_canonical}"
        )

    E = empirical_df.loc[:, target_channels].values
    C = canonical_df.loc[:, target_channels].values

    n_emp = E.shape[0]
    n_can = C.shape[0]
    matrix_abs = np.zeros((n_emp, n_can), dtype=float)
    matrix_signed = np.zeros((n_emp, n_can), dtype=float)

    for i in range(n_emp):
        for j in range(n_can):
            matrix_abs[i, j] = abs_corr(E[i], C[j])
            matrix_signed[i, j] = signed_spatial_corr(E[i], C[j])

    abs_df = pd.DataFrame(matrix_abs, index=empirical_df.index, columns=canonical_df.index)
    signed_df = pd.DataFrame(matrix_signed, index=empirical_df.index, columns=canonical_df.index)
    percent_df = abs_df * 100.0
    return abs_df, signed_df, percent_df


def make_best_match_table(percent_df: pd.DataFrame, signed_df: pd.DataFrame) -> pd.DataFrame:
    """Best-match table from an empirical x canonical similarity matrix."""
    rows: List[Dict[str, object]] = []
    for emp_label in percent_df.index:
        ordered = percent_df.loc[emp_label].sort_values(ascending=False)
        best_label = ordered.index[0]
        best_value = float(ordered.iloc[0])
        second_label = ordered.index[1] if len(ordered) > 1 else ""
        second_value = float(ordered.iloc[1]) if len(ordered) > 1 else np.nan
        rows.append(
            {
                "Empirical map": emp_label,
                "Best canonical match": best_label,
                "Similarity (%)": best_value,
                "Signed r": float(signed_df.loc[emp_label, best_label]),
                "Interpretation": classify_similarity(best_value),
                "Second best": second_label,
                "Second best (%)": second_value,
            }
        )
    return pd.DataFrame(rows)


def make_hungarian_assignment(
    percent_df: pd.DataFrame, signed_df: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str]]:
    """One-to-one assignment (Hungarian algorithm); returns table and unassigned maps."""
    cost = -(percent_df.values / 100.0)
    row_idx, col_idx = linear_sum_assignment(cost)

    rows: List[Dict[str, object]] = []
    for i, j in zip(row_idx, col_idx):
        emp = percent_df.index[i]
        can = percent_df.columns[j]
        sim = float(percent_df.loc[emp, can])
        rows.append(
            {
                "Empirical map": emp,
                "Assigned canonical map": can,
                "Similarity (%)": sim,
                "Signed r": float(signed_df.loc[emp, can]),
                "Interpretation": classify_similarity(sim),
            }
        )

    assigned_empirical = set(row_idx)
    unassigned = [percent_df.index[i] for i in range(percent_df.shape[0])
                  if i not in assigned_empirical]
    return pd.DataFrame(rows), unassigned


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    maps_dir = config.get_reference_maps_dir()
    templates_path = config.get_reference_templates()
    if templates_path is None:
        raise EnvironmentError("Set MICROSTATE_REFERENCE_TEMPLATES.")
    out_dir = config.get_out_dir()
    csv_dir = out_dir / "canonical"
    csv_dir.mkdir(parents=True, exist_ok=True)

    templates_npz = np.load(templates_path, allow_pickle=False)
    if "ch_names" in templates_npz.files:
        ch_names = templates_npz["ch_names"].tolist()
    else:
        ch_names = np.load(out_dir / "eeg_trials.npz")["ch_names"].tolist()
    empirical_df = load_empirical_microstates(templates_path, ch_names)
    missing_emp = [ch for ch in TARGET_19_CHANNELS if ch not in empirical_df.columns]
    if missing_emp:
        raise ValueError(
            "The empirical maps do not contain all target 19 channels after "
            f"standardization. Missing: {missing_emp}. "
            f"Channels found: {list(empirical_df.columns)}"
        )

    template_configs = [
        TemplateConfig("Koenig et al. 2002 canonical/reference maps", "Koenig 2002",
                       maps_dir / KOENIG_MAPS, maps_dir / KOENIG_CHANNELS, "ascii"),
        TemplateConfig("Milz et al. 2016 canonical/reference maps", "Milz 2016",
                       maps_dir / MILZ_MAPS, maps_dir / MILZ_CHANNELS, "ascii"),
        TemplateConfig("Artoni 2023 five-map reference solution", "Artoni 2023",
                       maps_dir / ARTONI_SET, None, "eeglab_set"),
    ]
    for cfg in template_configs:
        if not cfg.maps_file.is_file():
            raise FileNotFoundError(f"Template maps file not found: {cfg.maps_file.name}")
        if cfg.file_type == "ascii" and not cfg.channels_file.is_file():
            raise FileNotFoundError(
                f"Template channel file not found: {cfg.channels_file.name}")

    summary_rows: List[Dict[str, object]] = []
    unassigned_rows: List[Dict[str, object]] = []
    for cfg in template_configs:
        canonical_df = load_template(cfg)
        missing_can = [ch for ch in TARGET_19_CHANNELS if ch not in canonical_df.columns]
        if missing_can:
            raise ValueError(f"{cfg.short_name} is missing target channels: {missing_can}")

        _, signed_df, percent_df = compute_empirical_vs_canonical(
            empirical_df, canonical_df, TARGET_19_CHANNELS)
        best_df = make_best_match_table(percent_df, signed_df)
        hungarian_df, unassigned = make_hungarian_assignment(percent_df, signed_df)

        prefix = cfg.short_name.replace(" ", "_")
        percent_df.to_csv(csv_dir / f"{prefix}_absolute_similarity_percent.csv")
        signed_df.to_csv(csv_dir / f"{prefix}_signed_spatial_correlations.csv")
        best_df.to_csv(csv_dir / f"{prefix}_best_match_assignments.csv", index=False)
        hungarian_df.to_csv(csv_dir / f"{prefix}_hungarian_assignments.csv", index=False)

        for _, row in best_df.iterrows():
            summary_rows.append({
                "Template set": cfg.name,
                "Empirical map": row["Empirical map"],
                "Best canonical match": row["Best canonical match"],
                "Similarity (%)": row["Similarity (%)"],
                "Signed r": row["Signed r"],
                "Interpretation": row["Interpretation"],
            })
        for emp in unassigned:
            unassigned_rows.append({
                "Template set": cfg.name,
                "Unassigned empirical map": emp,
                "Best canonical match": best_df.set_index("Empirical map").loc[
                    emp, "Best canonical match"],
                "Best similarity (%)": best_df.set_index("Empirical map").loc[
                    emp, "Similarity (%)"],
            })

    pd.DataFrame(summary_rows).to_csv(
        csv_dir / "combined_best_match_summary.csv", index=False)
    pd.DataFrame(
        unassigned_rows,
        columns=["Template set", "Unassigned empirical map",
                 "Best canonical match", "Best similarity (%)"],
    ).to_csv(csv_dir / "unassigned_maps.csv", index=False)
    print("Saved to canonical/ in MICROSTATE_OUT_DIR")


if __name__ == "__main__":
    main()
