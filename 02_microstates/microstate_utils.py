"""
microstate_utils.py -- helper functions shared by fit_K5.py and fit_all_K.py.

Purpose : preprocessing, GFP-peak extraction, two-stage ModKMeans clustering,
          reordering to a reference, and GEV; same logic as the original
          K5 / All_K_Microstates notebooks.
Inputs  : arrays passed in by the calling scripts (no files are read here
          except the optional reference-templates file).
Outputs : none (importable module).
Environment : Python 3.11.14; mne 1.3.1, pycrostates 0.4.1; see requirements.txt.

Clustering as implemented (same as the original notebooks):
    stage 1: one ModKMeans fit PER TRIAL (1,200 fits) on that trial's GFP peaks
    stage 2: one ModKMeans fit on the pooled stage-1 cluster centres
             (19 channels x K*1200)
"""

import numpy as np
import mne
from scipy.stats import pearsonr
from pycrostates.preprocessing import extract_gfp_peaks
from pycrostates.cluster import ModKMeans
from pycrostates.io import ChData

import config

mne.set_log_level("ERROR")

LOW_FREQ = 4.0
HIGH_FREQ = 30.0
NAME_MAP = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def load_trials(npz_path):
    """Load the array written by 01_reshape/reshape.py."""
    npz = np.load(npz_path)
    data = npz["data"].astype(np.float64)       # (1200, 19, 650)
    ch_names = npz["ch_names"].tolist()
    sfreq = float(npz["sfreq"])
    return data, ch_names, sfreq


def make_info(ch_names, sfreq):
    """MNE Info with standard_1020 montage (old T3/T4/T5/T6 renamed)."""
    ch_names_mne = [NAME_MAP.get(c, c) for c in ch_names]
    info = mne.create_info(ch_names=ch_names_mne, sfreq=sfreq, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"))
    return info


def preprocess(data, info):
    """
    Concatenate all trials into one continuous block (19, n_trials*650),
    apply average reference and 4-30 Hz band-pass, split back into trials.
    """
    n_trials, n_ch, n_samp = data.shape
    continuous = data.transpose(1, 0, 2).reshape(n_ch, -1)
    raw = mne.io.RawArray(continuous, info, verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    raw.filter(l_freq=LOW_FREQ, h_freq=HIGH_FREQ, verbose=False)
    filtered = raw.get_data()
    return filtered.reshape(n_ch, n_trials, n_samp).transpose(1, 0, 2)


def extract_peaks(data_prep, info):
    """GFP peaks (min_peak_distance=1) for each trial."""
    peaks = []
    for i, trial in enumerate(data_prep):
        if (i + 1) % 200 == 0:
            print(f"  GFP peaks: trial {i + 1}/{len(data_prep)}")
        temp_raw = mne.io.RawArray(trial, info, verbose=False)
        peaks.append(extract_gfp_peaks(temp_raw, min_peak_distance=1))
    return peaks


def describe_modkmeans(K):
    """ModKMeans arguments as used here (n_init, max_iter, tol are defaults)."""
    m = ModKMeans(n_clusters=K, random_state=config.RANDOM_STATE)
    return {
        "n_clusters": m._n_clusters,
        "n_init": m._n_init,
        "random_state": config.RANDOM_STATE,
        "max_iter": m._max_iter,
        "tol": m._tol,
        "polarity": "ModKMeans is polarity-invariant (no argument)",
    }


def fit_templates(gfp_peaks_list, K, info, n_channels):
    """Two-stage clustering; returns templates (n_channels, K)."""
    individual_centers = []
    for i, gfp_data in enumerate(gfp_peaks_list):
        if (i + 1) % 200 == 0:
            print(f"  K={K} stage 1: trial {i + 1}/{len(gfp_peaks_list)}")
        modk = ModKMeans(n_clusters=K, random_state=config.RANDOM_STATE)
        modk.fit(gfp_data, n_jobs=1, verbose="WARNING")
        individual_centers.append(modk.cluster_centers_)

    group_centers = np.vstack(individual_centers).T          # (19, K*n_trials)
    print(f"  K={K} group matrix shape: {group_centers.shape}")
    modk_group = ModKMeans(n_clusters=K, random_state=config.RANDOM_STATE)
    modk_group.fit(ChData(group_centers, info), n_jobs=1, verbose="WARNING")

    templates = modk_group.cluster_centers_
    if templates.shape[0] != n_channels:
        templates = templates.T                              # ensure (19, K)
    return templates


def match_to_reference(new_T, ref_T):
    """Reorder columns of new_T to match ref_T by absolute correlation."""
    K = new_T.shape[1]
    used, order = set(), []
    for ref_k in range(K):
        best_corr, best_j = -1, -1
        for j in range(K):
            if j in used:
                continue
            r = abs(pearsonr(ref_T[:, ref_k], new_T[:, j])[0])
            if r > best_corr:
                best_corr, best_j = r, j
        used.add(best_j)
        order.append(best_j)
    return new_T[:, order], order


def load_reference(n_channels):
    """Reference templates (n_channels, K) or None if none configured."""
    ref_path = config.get_reference_templates()
    if ref_path is None:
        return None
    ref_T = np.load(ref_path, allow_pickle=True)["templates"]
    if ref_T.shape[0] != n_channels:
        ref_T = ref_T.T
    return ref_T


def compute_gev(data_array, templates):
    """
    Global explained variance. For every sample: squared GFP (ddof=1) times the
    squared largest absolute Pearson correlation with the templates, summed over
    all samples and divided by total squared GFP. Correlations are computed in
    one matrix product per trial; the quantity is the same as the original
    per-sample np.corrcoef loop.
    """
    T = templates - templates.mean(axis=0, keepdims=True)
    T_norm = np.linalg.norm(T, axis=0)
    gev_total = gfp_total = 0.0
    for trial in data_array:
        gfp = np.std(trial, axis=0, ddof=1)
        gfp_total += np.sum(gfp ** 2)
        X = trial - trial.mean(axis=0, keepdims=True)
        corr = (T.T @ X) / (T_norm[:, None] * np.linalg.norm(X, axis=0)[None, :])
        gev_total += np.sum((gfp ** 2) * (np.abs(corr).max(axis=0) ** 2))
    return float(gev_total / gfp_total)
