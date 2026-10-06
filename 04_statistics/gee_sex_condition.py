"""
gee_sex_condition.py
====================
Assesses sex bias and condition effects (Go vs. NoGo) in EEG microstate
metrics (coverage, lifespan, occurrences) using Generalized Estimating
Equations (GEE).

Environment
-----------
Python      : 3.11.14
numpy       : 1.25.2
pandas      : 2.0.3
scipy       : 1.11.4
statsmodels : 0.14.6
(project venv; exact pins in ../requirements.txt)

Usage
-----
python gee_sex_condition.py [--source eeg|deposited]

Input
-----
--source eeg
tables/microstate_metrics_K5_reference.csv in MICROSTATE_OUT_DIR (from
03_sequences/metrics.py; labels from the K=5 reference templates, the only
template set). It has file_prefix (h|m) instead of sex; the sex
label is taken from config.get_sex_map() (config.SEX_MAP, h = Male and
m = Female; optional override with the environment variable
MICROSTATE_SEX_MAP). Conditions are labelled GO and NG.
--source deposited
microstate_metrics_K5.csv   (in the folder given by MICROSTATE_INPUT_DIR);
the sex and condition labels are used as stored in that file. Not used by
run_all.sh.
The two possible mappings define the same partition of participants with the
labels swapped, so the joint Wald tests and the AIC/QIC differences do not
depend on the choice.

    Columns used
    ------------
    subject     : int   — participant ID (1–30)
    sex         : str   — 'Male' or 'Female' (from config.SEX_MAP when the
                          source is eeg; as stored when deposited)
    condition   : str   — 'GO' or 'NG' ('NOGO' in a deposited table)
    trial       : int   — trial number (1–20)
    state       : str   — microstate label ('A'–'E')
    coverage    : float — proportion of epoch time dominated by this state (0–1)
    lifespan    : float — mean duration (ms) per microstate occurrence;
                          zero values excluded for Gamma family
    occurrences : int   — count of times this state appeared per trial

    Column 'initials' is present in the raw file but is NOT loaded or used.

    Unit of observation: subject × condition × trial × state
    Total rows: 6 000  (30 × 2 × 20 × 5)

    Generation pipeline for coverage, lifespan and occurrences is NOT
    reconstructed here; the file is treated as-is.

Sex codification
----------------
    Author decision 2026-10-04: raw folders 'Hombres' (file prefix h, Male,
    subjects 1–15) and 'Mujeres' (file prefix m, Female, subjects 16–30);
    config.SEX_MAP. The model has the first category alphabetically
    (Female) as reference level.

Output (written to get_out_dir()/tables/)
-----------------------------------------
    gee_sex_condition_results.csv   — one row per metric with all statistics
    gee_sex_condition_results.txt   — human-readable console summary
    (no figures)

Models
------
    Reduced : metric ~ condition + state
    Full    : metric ~ condition + state + sex + condition:sex

    Within-subject correlation: exchangeable structure, grouped by subject.
    Sex Wald p-value  : joint test across sex + condition:sex terms (from full model).
    Condition Wald p-value: joint test across condition terms (from reduced model).

Distribution families
---------------------
    Coverage    : Binomial / logit — fitted WITHOUT analytic weights.
                  The Methods section states "using epoch duration as analytic
                  weights"; statsmodels GEE does not expose a weights parameter
                  equivalent to GLM weights, so no weights were passed in the
                  code executed to produce the reported results. The manuscript
                  values are reproduced without weights. The Methods text
                  ("analytic weights") should be corrected to reflect this.
                  No weights are added here; the discrepancy is documented, not
                  corrected.
    Lifespan    : Gamma / log
    Occurrences : Negative Binomial / log; overdispersion α estimated via
                  method of moments: α = max((Var − μ) / μ², 1e-6)

Determinism
-----------
    No stochastic components. Identical outputs on repeated runs are guaranteed
    by statsmodels GEE deterministic convergence; no random seeds needed.
"""

# ── Imports ───────────────────────────────────────────────────────────────────

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────

_parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
_parser.add_argument("--source", choices=["eeg", "deposited"],
                     default="deposited",
                     help="metrics table: computed from the EEG or deposited "
                          "(default: deposited)")
SOURCE = _parser.parse_args().source

SEX_MAP_ENV = None
if SOURCE == "eeg":
    SEX_MAP_ENV = config.get_sex_map()
    INPUT_FILE = (config.get_out_dir() / "tables"
                  / "microstate_metrics_K5_reference.csv")
else:
    INPUT_FILE = config.get_input_dir() / "microstate_metrics_K5.csv"
OUTPUT_DIR = config.get_out_dir() / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUT = OUTPUT_DIR / "gee_sex_condition_results.csv"
TXT_OUT = OUTPUT_DIR / "gee_sex_condition_results.txt"

# ── Load — 'initials' excluded ────────────────────────────────────────────────

USECOLS = ["subject", "sex", "condition", "trial", "state",
           "coverage", "lifespan", "occurrences"]

if SOURCE == "eeg":
    df = pd.read_csv(INPUT_FILE, usecols=[c if c != "sex" else "file_prefix"
                                          for c in USECOLS],
                     dtype={"file_prefix": str})
    df["sex"] = df.pop("file_prefix").map(SEX_MAP_ENV)
    df = df[USECOLS]
else:
    df = pd.read_csv(INPUT_FILE, usecols=USECOLS)

# ── Assertions ────────────────────────────────────────────────────────────────

assert df.shape[0] == 6000, (
    f"Expected 6 000 rows, got {df.shape[0]}."
)
assert df["subject"].nunique() == 30, (
    f"Expected 30 subjects, got {df['subject'].nunique()}."
)
assert df["state"].nunique() == 5, (
    f"Expected 5 microstate labels, got {df['state'].nunique()}."
)
trials_per_subj_cond = (
    df.groupby(["subject", "condition"])["trial"].nunique()
)
assert (trials_per_subj_cond == 20).all(), (
    f"Not all subject × condition cells have exactly 20 trials:\n"
    f"{trials_per_subj_cond[trials_per_subj_cond != 20]}"
)
assert df.isnull().sum().sum() == 0, (
    f"Missing values found:\n{df.isnull().sum()[df.isnull().sum() > 0]}"
)

# ── Encode categoricals ───────────────────────────────────────────────────────

df["sex"]       = df["sex"].astype("category")
df["condition"] = df["condition"].astype("category")
df["state"]     = df["state"].astype("category")
df["subject"]   = df["subject"].astype(str)

# ── Sex codification report ───────────────────────────────────────────────────

sex_counts = df.groupby("sex")["subject"].nunique()
if SOURCE == "eeg":
    _ids = (df.assign(subject=df["subject"].astype(int))
            .groupby("sex")["subject"].agg(["nunique", "min", "max"]))
    sex_report = (
        "Sex codification (config.SEX_MAP): "
        + ", ".join(f"{k} = {r['nunique']} subjects (IDs {r['min']}-{r['max']})"
                    for k, r in _ids.iterrows())
        + ".")
else:
    sex_report = (
        f"Sex codification (as stored in {INPUT_FILE.name}): "
        + ", ".join(f"{k} = {v} subjects" for k, v in sex_counts.items())
        + "."
    )

# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_qic(gee_model, gee_result):
    """
    QIC (Pan, 2001): -2Q + 2 * trace(Ω_I⁻¹ · Ω_R)
    Q = quasi-likelihood under independence evaluated at GEE estimates.
    Returns NaN on failure.
    """
    try:
        ind_model = gee_model.__class__(
            gee_model.endog,
            gee_model.exog,
            gee_model.groups,
            family=gee_model.family,
            cov_struct=sm.cov_struct.Independence(),
        )
        ind_result = ind_model.fit(start_params=gee_result.params, maxiter=1)
        mu  = ind_result.fittedvalues
        Q   = np.sum(gee_model.family.loglike_obs(gee_model.endog, mu))
        Ω_I = gee_result.cov_naive
        Ω_R = gee_result.cov_robust
        pen = np.trace(np.linalg.solve(Ω_I, Ω_R))
        return float(-2 * Q + 2 * pen)
    except Exception:
        return np.nan


def wald_pvalue(result, term_substr):
    """
    Joint Wald test for all parameters whose name contains `term_substr`.
    Returns (p_value, n_terms).
    """
    idx = [i for i, n in enumerate(result.model.exog_names)
           if term_substr in n.lower()]
    if not idx:
        return np.nan, 0
    try:
        R = np.zeros((len(idx), len(result.params)))
        for row, col in enumerate(idx):
            R[row, col] = 1.0
        wald = result.wald_test(R, scalar=False)
        return float(wald.pvalue), len(idx)
    except Exception:
        return np.nan, len(idx)


def glm_delta_aic(data, formula_r, formula_f, family):
    """
    ΔAIC = AIC(full) − AIC(reduced) via GLM.
    Negative = full model (with sex) is better.
    Returns NaN on failure.
    """
    try:
        m0 = smf.glm(formula_r, data=data, family=family).fit()
        m1 = smf.glm(formula_f, data=data, family=family).fit()
        return float(m1.aic - m0.aic)
    except Exception:
        return np.nan


def fit_gee_pair(data, formula_r, formula_f, family):
    """Fit reduced and full GEE models. Returns (gee_r, res_r, gee_f, res_f)."""
    gee_r = smf.gee(formula_r, groups="subject", data=data,
                    family=family, cov_struct=sm.cov_struct.Exchangeable())
    res_r = gee_r.fit()
    gee_f = smf.gee(formula_f, groups="subject", data=data,
                    family=family, cov_struct=sm.cov_struct.Exchangeable())
    res_f = gee_f.fit()
    return gee_r, res_r, gee_f, res_f


def run_metric(label, data, formula_r, formula_f, glm_family, gee_family):
    """Full pipeline for one metric. Returns dict with all statistics."""
    gee_r, res_r, gee_f, res_f = fit_gee_pair(
        data, formula_r, formula_f, gee_family
    )

    wald_sex,  n_sex  = wald_pvalue(res_f, "sex")
    wald_cond, n_cond = wald_pvalue(res_r, "condition")  # from reduced model

    delta_aic = glm_delta_aic(data, formula_r, formula_f, glm_family)

    qic_r     = compute_qic(gee_r, res_r)
    qic_f     = compute_qic(gee_f, res_f)
    delta_qic = (qic_f - qic_r
                 if not (np.isnan(qic_r) or np.isnan(qic_f))
                 else np.nan)

    sex_sig  = (wald_sex  < 0.05) if not np.isnan(wald_sex)  else False
    cond_sig = (wald_cond < 0.05) if not np.isnan(wald_cond) else False

    return {
        "Metric":           label,
        "Wald_p_sex":       round(wald_sex,  4),
        "n_sex_terms":      n_sex,
        "Wald_p_condition": round(wald_cond, 4),
        "n_cond_terms":     n_cond,
        "ΔAIC":             round(delta_aic, 3),
        "QIC_reduced":      round(qic_r,     3) if not np.isnan(qic_r) else np.nan,
        "QIC_full":         round(qic_f,     3) if not np.isnan(qic_f) else np.nan,
        "ΔQIC":             round(delta_qic, 3) if not np.isnan(delta_qic) else np.nan,
        "Sex_effect":       sex_sig,
        "Condition_effect": cond_sig,
        "Decision_sex":     "Include sex" if sex_sig else "Exclude sex",
    }

# ── Formulas ──────────────────────────────────────────────────────────────────

FORMULA_R = "{metric} ~ condition + state"
FORMULA_F = "{metric} ~ condition + state + sex + condition:sex"

# ── 1. Coverage ───────────────────────────────────────────────────────────────

df_cov  = df.copy()
fam_cov = sm.families.Binomial()
res_cov = run_metric(
    label     = "Coverage",
    data      = df_cov,
    formula_r = FORMULA_R.format(metric="coverage"),
    formula_f = FORMULA_F.format(metric="coverage"),
    glm_family= fam_cov,
    gee_family= fam_cov,
)

# ── 2. Lifespan ───────────────────────────────────────────────────────────────

df_life  = df[df["lifespan"] > 0].copy()  # Gamma requires strictly positive
fam_life = sm.families.Gamma(link=sm.families.links.Log())
res_life = run_metric(
    label     = "Lifespan",
    data      = df_life,
    formula_r = FORMULA_R.format(metric="lifespan"),
    formula_f = FORMULA_F.format(metric="lifespan"),
    glm_family= fam_life,
    gee_family= fam_life,
)

# ── 3. Occurrences ────────────────────────────────────────────────────────────

df_occ   = df.copy()
mu_occ   = df_occ["occurrences"].mean()
var_occ  = df_occ["occurrences"].var()
alpha_nb = max((var_occ - mu_occ) / (mu_occ ** 2), 1e-6)
fam_occ  = sm.families.NegativeBinomial(alpha=alpha_nb)
res_occ  = run_metric(
    label     = "Occurrences",
    data      = df_occ,
    formula_r = FORMULA_R.format(metric="occurrences"),
    formula_f = FORMULA_F.format(metric="occurrences"),
    glm_family= fam_occ,
    gee_family= fam_occ,
)

# ── Compile results ───────────────────────────────────────────────────────────

results    = [res_cov, res_life, res_occ]
df_results = pd.DataFrame(results)

# ── Manuscript comparison ─────────────────────────────────────────────────────
# Values printed next to the computed ones; no tolerance is applied.

MANUSCRIPT_VALUES = {
    "Coverage":    {"Wald_p_sex": 0.309,  "ΔAIC":  +4.0,   "ΔQIC":  +0.116},
    "Lifespan":    {"Wald_p_sex": 0.251,  "ΔAIC": -190.1,  "ΔQIC":  -0.693},
    "Occurrences": {"Wald_p_sex": 0.056,  "ΔAIC": -205.1,  "ΔQIC": -213.3 },
}

discrepancy_lines = []
for row in results:
    m   = row["Metric"]
    ref = MANUSCRIPT_VALUES.get(m, {})
    for stat in ["Wald_p_sex", "ΔAIC", "ΔQIC"]:
        ref_val = ref.get(stat)
        got_val = row.get(stat)
        if ref_val is None or got_val is None or (
                isinstance(got_val, float) and np.isnan(got_val)):
            discrepancy_lines.append(
                f"  {m:<14} | {stat:<12}: manuscript = {ref_val!s:<10} "
                f"computed = {got_val!s:<12}  [comparison not possible]"
            )
        else:
            discrepancy_lines.append(
                f"  {m:<14} | {stat:<12}: manuscript = {ref_val!s:<10} "
                f"computed = {got_val!s:<12}"
            )

# ── Overdispersion note ───────────────────────────────────────────────────────

nb_note = (
    f"Negative Binomial overdispersion α (method of moments): "
    f"μ = {mu_occ:.4f}, Var = {var_occ:.4f}, α = {alpha_nb:.4f}"
)

# ── Build text summary ────────────────────────────────────────────────────────

lines = []
lines.append("=" * 72)
lines.append("GEE SEX BIAS & CONDITION EFFECTS — SUMMARY")
lines.append("=" * 72)
lines.append("")
lines.append(sex_report)
lines.append("")
lines.append(nb_note)
lines.append("")
lines.append(
    df_results[[
        "Metric", "Wald_p_sex", "Wald_p_condition",
        "ΔAIC", "ΔQIC", "Sex_effect", "Condition_effect", "Decision_sex"
    ]].to_string(index=False)
)
lines.append("")
lines.append("QIC values (for completeness):")
lines.append(
    df_results[["Metric", "QIC_reduced", "QIC_full", "ΔQIC"]].to_string(index=False)
)
lines.append("")
lines.append("─" * 72)
lines.append("MANUSCRIPT vs. COMPUTED VALUES")
lines.append("─" * 72)
lines.extend(discrepancy_lines)
lines.append("")
lines.append("─" * 72)
lines.append("COVERAGE — METHOD NOTE (to correct in manuscript)")
lines.append("─" * 72)
lines.append(
    "The Methods text states coverage was fitted with Binomial/logit\n"
    "using epoch duration as analytic weights. statsmodels GEE does not\n"
    "expose a weights parameter equivalent to GLM weights; the code\n"
    "executed to produce the reported results did NOT pass any weights.\n"
    "All reported values are reproduced WITHOUT weights. The Methods\n"
    "sentence 'using epoch duration as analytic weights' should be\n"
    "removed or corrected. No weights are added in this script."
)
lines.append("=" * 72)

summary_text = "\n".join(lines)

# ── Write outputs (no 'initials' column in either file) ──────────────────────

df_results.to_csv(CSV_OUT, index=False)
TXT_OUT.write_text(summary_text, encoding="utf-8")

# ── Console ───────────────────────────────────────────────────────────────────

print(summary_text)
print(f"\nCSV: {CSV_OUT.name}")
print(f"TXT: {TXT_OUT.name}")
