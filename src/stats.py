"""Disparity modelling.

Word errors are modelled as counts rather than comparing raw WER, because
accent groups differ in utterance length distribution and in the number of
speakers contributing, and because utterances are clustered within speakers.

Specification:
    errors_ij ~ Poisson(mu_ij)
    log(mu_ij) = beta_0 + beta_g * group_i + log(ref_len_ij)

The log reference length enters as an OFFSET, not a covariate. Its coefficient
is fixed at 1, which constrains the model to describe errors per reference
token, which is exactly a word error rate. Standard errors are clustered on
speaker to accommodate within-speaker dependence.

Exponentiated coefficients are disparity rate ratios: the multiplicative factor
by which a group's error rate exceeds the reference variety, adjusted for
utterance length and speaker clustering.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


@dataclass
class FitResult:
    family: str
    table: pd.DataFrame
    pearson_chi2_dof: float
    n_observations: int
    n_speakers: int
    formula: str
    converged: bool

    def rate_ratio(self, group: str) -> Optional[float]:
        rows = [i for i in self.table.index if group in i]
        return float(self.table.loc[rows[0], "rate_ratio"]) if rows else None


def _overdispersion(model_result) -> float:
    """Pearson chi-squared divided by residual degrees of freedom.

    A Poisson model assumes variance equals the mean. Values materially above
    1 indicate overdispersion, under which Poisson standard errors are too
    small and significance is overstated.
    """
    try:
        return float(model_result.pearson_chi2 / model_result.df_resid)
    except Exception:
        return float("nan")


def fit_disparity_model(
    df: pd.DataFrame,
    errors_col: str,
    reflen_col: str,
    group_col: str = "accent_group",
    speaker_col: str = "speaker_id",
    reference_group: str = "us_baseline",
    covariates: Optional[List[str]] = None,
    overdispersion_threshold: float = 1.5,
) -> FitResult:
    """Fit the disparity model, escalating to negative binomial if overdispersed.

    The escalation rule is applied automatically and the adopted family is
    reported, so that the specification is selected by a stated criterion
    rather than by inspection of which result looks preferable.
    """
    covariates = covariates or []
    needed = [errors_col, reflen_col, group_col, speaker_col] + covariates
    d = df.dropna(subset=needed).copy()
    d = d[d[reflen_col] > 0]
    if d.empty:
        raise ValueError("No usable rows for modelling after filtering.")

    present = [g for g in d[group_col].unique() if g != reference_group]
    if reference_group not in set(d[group_col]):
        raise ValueError(
            f"Reference group {reference_group!r} absent; disparity is undefined."
        )
    d[group_col] = pd.Categorical(
        d[group_col], categories=[reference_group] + sorted(present)
    )

    d["_errors"] = d[errors_col].astype(float).round().astype(int)
    d["_offset"] = np.log(d[reflen_col].astype(float))

    terms = [f"C({group_col})"] + [f"C({c})" for c in covariates]
    formula = "_errors ~ " + " + ".join(terms)

    def _fit(family):
        model = smf.glm(formula, data=d, family=family, offset=d["_offset"])
        return model.fit(cov_type="cluster", cov_kwds={"groups": d[speaker_col]})

    res = _fit(sm.families.Poisson())
    disp = _overdispersion(res)
    family_used = "poisson"

    if np.isfinite(disp) and disp > overdispersion_threshold:
        # Estimate alpha from the Poisson Pearson dispersion, then refit.
        alpha = max((disp - 1.0) / max(d["_errors"].mean(), 1e-6), 1e-6)
        try:
            res = _fit(sm.families.NegativeBinomial(alpha=alpha))
            family_used = f"negative_binomial(alpha={alpha:.4f})"
        except Exception:
            family_used = "poisson (negative binomial failed to converge)"

    ci = res.conf_int()
    table = pd.DataFrame({
        "coefficient": res.params,
        "rate_ratio": np.exp(res.params),
        "rr_ci_low": np.exp(ci[0]),
        "rr_ci_high": np.exp(ci[1]),
        "std_err": res.bse,
        "p_value": res.pvalues,
    }).round(4)

    return FitResult(
        family=family_used,
        table=table,
        pearson_chi2_dof=disp,
        n_observations=int(len(d)),
        n_speakers=int(d[speaker_col].nunique()),
        formula=formula,
        converged=bool(getattr(res, "converged", True)),
    )


def fit_intersectional(
    df: pd.DataFrame,
    errors_col: str,
    reflen_col: str,
    covariate: str,
    group_col: str = "accent_group",
    speaker_col: str = "speaker_id",
    reference_group: str = "us_baseline",
    min_cell: int = 20,
) -> Optional[FitResult]:
    """Fit an accent-by-demographic interaction, or return None if too sparse.

    Returning None rather than a fitted model is deliberate. Interaction terms
    estimated on near-empty cells are unstable, and reporting them at low power
    would misrepresent the evidence. A None result is reported in the paper as
    a stated power limitation.
    """
    d = df.dropna(subset=[errors_col, reflen_col, group_col, speaker_col, covariate]).copy()
    d = d[d[reflen_col] > 0]
    if d.empty:
        return None

    # Drop groups whose demographic cells are too sparse, rather than
    # abandoning the whole analysis. A group with almost no speakers of one
    # gender cannot support an interaction term, but its presence should not
    # prevent the groups that can from being estimated. Dropped groups are
    # named in the log and reported in the paper.
    cells = d.groupby([group_col, covariate], observed=True).size()
    if cells.empty:
        return None
    counts = cells.unstack(fill_value=0)
    viable = [g for g in counts.index if counts.loc[g].min() >= min_cell]
    if reference_group not in viable or len(viable) < 2:
        return None
    dropped = [g for g in counts.index if g not in viable]
    if dropped:
        import logging as _lg
        _lg.getLogger(__name__).warning(
            "intersectional: excluding %s for sparse %s cells", dropped, covariate)
    d = d[d[group_col].isin(viable)].copy()

    d["_errors"] = d[errors_col].astype(float).round().astype(int)
    d["_offset"] = np.log(d[reflen_col].astype(float))
    present = [g for g in d[group_col].unique() if g != reference_group]
    d[group_col] = pd.Categorical(
        d[group_col], categories=[reference_group] + sorted(present)
    )

    formula = f"_errors ~ C({group_col}) * C({covariate})"
    try:
        res = smf.glm(formula, data=d, family=sm.families.Poisson(),
                      offset=d["_offset"]).fit(
            cov_type="cluster", cov_kwds={"groups": d[speaker_col]})
    except Exception:
        return None

    ci = res.conf_int()
    table = pd.DataFrame({
        "coefficient": res.params,
        "rate_ratio": np.exp(res.params),
        "rr_ci_low": np.exp(ci[0]),
        "rr_ci_high": np.exp(ci[1]),
        "p_value": res.pvalues,
    }).round(4)

    return FitResult(
        family="poisson",
        table=table,
        pearson_chi2_dof=_overdispersion(res),
        n_observations=int(len(d)),
        n_speakers=int(d[speaker_col].nunique()),
        formula=formula,
        converged=True,
    )
