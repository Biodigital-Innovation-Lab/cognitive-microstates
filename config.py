"""
config.py -- central configuration for the microstate analysis code.

Purpose : locate raw data and derived outputs; hold constants shared by all
          scripts. Contains no data and no participant information.
Inputs  : environment variables
            MICROSTATE_DATA_DIR  folder with the raw .txt EEG files (required
                                 by the scripts that read raw data)
            MICROSTATE_OUT_DIR   folder for derived files
                                 (default: ./outputs next to this file)
            MICROSTATE_SEQUENCES_DIR
                                 folder with the deposited label sequences
                                 (microstate_sequences_K5_reference.csv), read
                                 by 03_sequences/import_sequences.py
                                 (run_all.sh --from-sequences)
            MICROSTATE_TEMPLATES_DIR
                                 folder with the deposited group templates
                                 microstates_K{4..9}_templates.npz (October
                                 2025 run), read by 02_microstates/gev_by_k.py
                                 (run_all.sh --from-eeg) and 02_microstates/
                                 k_template_correlations.py (both modes)
            MICROSTATE_INPUT_DIR folder with the input tables read by
                                 04_statistics and 05_alignment when
                                 --source deposited is used
                                 (microstate_metrics_K5.csv,
                                 msa_consensus_SUBJECT.csv,
                                 microstate_sequences_FULL.csv); not used by
                                 run_all.sh
            MICROSTATE_SEX_MAP   optional override of SEX_MAP below;
                                 "h=Male,m=Female" or "h=Female,m=Male"
                                 (the short forms F and M are also accepted).
                                 Used only by 04_statistics/
                                 gee_sex_condition.py and 06_tables/
                                 assemble_supplementary.py.
Outputs : none (importable module).
Environment : Python 3.11.14; see requirements.txt.
"""

import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent


def get_data_dir() -> Path:
    """Folder with the raw .txt files (MICROSTATE_DATA_DIR)."""
    value = os.environ.get("MICROSTATE_DATA_DIR")
    if not value:
        raise EnvironmentError(
            "Set MICROSTATE_DATA_DIR to the folder containing the raw .txt files."
        )
    path = Path(value).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"MICROSTATE_DATA_DIR is not a folder: {path}")
    return path


def get_sequences_dir() -> Path:
    """Folder with the deposited label sequences (MICROSTATE_SEQUENCES_DIR)."""
    value = os.environ.get("MICROSTATE_SEQUENCES_DIR")
    if not value:
        raise EnvironmentError(
            "Set MICROSTATE_SEQUENCES_DIR to the folder containing "
            "microstate_sequences_K5_reference.csv."
        )
    path = Path(value).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"MICROSTATE_SEQUENCES_DIR is not a folder: {path}")
    return path


def get_templates_dir() -> Path:
    """
    Folder with the deposited group templates microstates_K{4..9}_templates.npz
    (MICROSTATE_TEMPLATES_DIR).
    """
    value = os.environ.get("MICROSTATE_TEMPLATES_DIR")
    if not value:
        raise EnvironmentError(
            "Set MICROSTATE_TEMPLATES_DIR to the folder containing "
            "microstates_K4_templates.npz ... microstates_K9_templates.npz."
        )
    path = Path(value).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"MICROSTATE_TEMPLATES_DIR is not a folder: {path}")
    return path


def get_input_dir() -> Path:
    """Folder with the input tables (MICROSTATE_INPUT_DIR)."""
    value = os.environ.get("MICROSTATE_INPUT_DIR")
    if not value:
        raise EnvironmentError(
            "Set MICROSTATE_INPUT_DIR to the folder containing "
            "microstate_metrics_K5.csv and msa_consensus_SUBJECT.csv."
        )
    path = Path(value).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"MICROSTATE_INPUT_DIR is not a folder: {path}")
    return path


def get_out_dir() -> Path:
    """Folder for derived files (MICROSTATE_OUT_DIR, default ./outputs)."""
    value = os.environ.get("MICROSTATE_OUT_DIR")
    path = Path(value).expanduser() if value else CODE_DIR / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_reference_templates():
    """
    K=5 reference templates (.npz with a 'templates' array; the K=5 solution
    of the October 2025 run) used for back-fitting and canonical
    correspondence, and by fit_K5.py / fit_all_K.py to order the maps. Path
    from MICROSTATE_REFERENCE_TEMPLATES; returns None if the variable is unset.
    """
    value = os.environ.get("MICROSTATE_REFERENCE_TEMPLATES")
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"MICROSTATE_REFERENCE_TEMPLATES is not a file: {path}"
        )
    return path


def get_reference_maps_dir() -> Path:
    """
    Folder with the third-party canonical reference maps used by
    02_microstates/canonical_similarity.py (MICROSTATE_REFERENCE_MAPS_DIR).
    The maps are not distributed with this code.
    """
    value = os.environ.get("MICROSTATE_REFERENCE_MAPS_DIR")
    if not value:
        raise EnvironmentError(
            "Set MICROSTATE_REFERENCE_MAPS_DIR to the folder containing the "
            "canonical reference maps."
        )
    path = Path(value).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(
            f"MICROSTATE_REFERENCE_MAPS_DIR is not a folder: {path}"
        )
    return path


def get_sex_map() -> dict:
    """
    Mapping filename prefix -> sex label. Returns SEX_MAP unless the optional
    environment variable MICROSTATE_SEX_MAP overrides it ("h=Male,m=Female" or
    "h=Female,m=Male"; "F" and "M" are accepted as short forms).
    """
    value = os.environ.get("MICROSTATE_SEX_MAP")
    if not value:
        return dict(SEX_MAP)
    short = {"F": "Female", "M": "Male"}
    try:
        pairs = dict(item.strip().split("=") for item in value.split(","))
    except ValueError:
        pairs = {}
    pairs = {k: short.get(v, v) for k, v in pairs.items()}
    if set(pairs) != {"h", "m"} or set(pairs.values()) != {"Male", "Female"}:
        raise ValueError(
            'MICROSTATE_SEX_MAP must be "h=Male,m=Female" or "h=Female,m=Male"')
    return pairs


# ── Recording constants (from the data description) ──────────────────────────
N_CHANNELS = 19
N_SAMPLES = 650        # samples per trial: 1,300 ms at 500 Hz
N_TRIALS = 20          # trials per file
SFREQ = 500.0
CONDITIONS = ("GO", "NG")

# Fixed random seed (value found in the original code)
RANDOM_STATE = 42

# ── Sex coding ───────────────────────────────────────────────────────────────
# Only the raw filename prefix (h or m) is stored in the sequence and metric
# tables, as "file_prefix". The sex label is attached where a table or model
# needs it (GEE, supplementary tables) with this mapping.
# Author decision 2026-10-04: raw folders 'Hombres' (h) and 'Mujeres' (m)
SEX_MAP = {"h": "Male", "m": "Female"}
