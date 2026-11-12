"""Gini coefficient, top-share count, the paired BCa bootstrap interval, the
matched-pairs rank-biserial correlation, and the two-way cluster-robust logit
covariance with its joint Wald test."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

N_BOOT = 10000


def compute_gini(values: np.ndarray) -> float:
    """Gini coefficient of |values|: 0 when all are equal, towards 1 when one holds all."""
    values = np.abs(values)
    sorted_values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(sorted_values)
    if cumsum[-1] == 0:
        return 0.0
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * cumsum[-1]) - (
        n + 1
    ) / n
    return float(gini)


def top_share_count(desc_values: np.ndarray, share: float):
    """Smallest number of leading entries of descending `desc_values` whose sum
    reaches `share` of the total (a NumPy integer)."""
    return np.searchsorted(np.cumsum(desc_values), share * np.sum(desc_values)) + 1


def bca_bootstrap_ci(data_within, data_cross, n_boot=N_BOOT, seed=0, alpha=0.05):
    """BCa bootstrap CI on mean(data_within) - mean(data_cross), resampling pairs.

    data_within[i] and data_cross[i] belong to the same unit (capsule) i.
    Returns (theta_hat, lo, hi, diagnostics).
    """
    rng = np.random.default_rng(seed)
    n = len(data_within)
    theta_hat = np.mean(data_within) - np.mean(data_cross)

    idx = rng.integers(0, n, size=(n_boot, n))
    boot_w = data_within[idx].mean(axis=1)
    boot_c = data_cross[idx].mean(axis=1)
    boot_theta = boot_w - boot_c

    # Bias correction
    z0 = sp_stats.norm.ppf(np.mean(boot_theta < theta_hat))
    # Acceleration via jackknife
    jack_theta = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        jack_theta[i] = np.mean(data_within[mask]) - np.mean(data_cross[mask])
    jack_mean = jack_theta.mean()
    num = np.sum((jack_mean - jack_theta) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack_theta) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0

    z_alpha_lo = sp_stats.norm.ppf(alpha / 2)
    z_alpha_hi = sp_stats.norm.ppf(1 - alpha / 2)

    def bca_quantile(z_alpha):
        z = z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))
        return sp_stats.norm.cdf(z)

    lo_q = bca_quantile(z_alpha_lo)
    hi_q = bca_quantile(z_alpha_hi)
    # The two ways BCa silently stops being BCa: a degenerate bias correction (every
    # replicate on one side of theta_hat, so z0 is infinite), and a target quantile
    # outside [0, 1], where the clip below substitutes an extreme order statistic.
    diagnostics = {
        "z0_degenerate": bool(not np.isfinite(z0)),
        "quantile_clipped_low": bool(not (0.0 <= lo_q <= 1.0)),
        "quantile_clipped_high": bool(not (0.0 <= hi_q <= 1.0)),
        "acceleration_a": float(a),
        "z0": float(z0) if np.isfinite(z0) else None,
    }
    lo = np.quantile(boot_theta, np.clip(lo_q, 0, 1))
    hi = np.quantile(boot_theta, np.clip(hi_q, 0, 1))
    return float(theta_hat), float(lo), float(hi), diagnostics


def rank_biserial_matched(diffs):
    """Matched-pairs rank-biserial correlation for the Wilcoxon signed-rank test."""
    diffs = diffs[diffs != 0]
    n = len(diffs)
    if n == 0:
        return 0.0
    ranks = sp_stats.rankdata(np.abs(diffs))
    r_plus = ranks[diffs > 0].sum()
    r_minus = ranks[diffs < 0].sum()
    total = r_plus + r_minus
    return float((r_plus - r_minus) / total) if total > 0 else 0.0


def two_way_logit_cov(fit, g_cap, g_img):
    """Two-way (Cameron-Gelbach-Miller) cluster-robust covariance of a fitted
    statsmodels logit: V_cap + V_img - V_(cap x img), from the score contributions
    s_i = (y_i - p_i) x_i and the bread fit.cov_params().

    g_cap, g_img: integer cluster codes, one per observation. A non-PSD result is
    projected to the nearest PSD matrix. Returns (cov, n_neg), n_neg being the
    number of negative eigenvalues before projection (0: no projection).
    """
    exog = fit.model.exog
    resid = fit.model.endog - fit.predict()
    score_obs = resid[:, None] * exog
    bread = fit.cov_params().values
    g_both = pd.Categorical(
        pd.Series(g_cap).astype(str) + "_" + pd.Series(g_img).astype(str)
    ).codes

    def clustered_cov(groups):
        groups = np.asarray(groups)
        n_g = groups.max() + 1
        sums = np.zeros((n_g, score_obs.shape[1]))
        np.add.at(sums, groups, score_obs)
        meat = sums.T @ sums
        # Finite-cluster correction, as statsmodels applies for cov_type='cluster'
        k = exog.shape[1]
        corr = (n_g / (n_g - 1.0)) * ((len(groups) - 1.0) / (len(groups) - k))
        return bread @ (meat * corr) @ bread

    cov = clustered_cov(g_cap) + clustered_cov(g_img) - clustered_cov(g_both)
    eigval, eigvec = np.linalg.eigh(cov)
    n_neg = int((eigval < 0).sum())
    if n_neg:
        cov = eigvec @ np.diag(np.clip(eigval, 0, None)) @ eigvec.T
    return cov, n_neg


def joint_wald(params, cov, names, terms):
    """Per-term and joint Wald tests of the coefficients named in `terms`.

    params: coefficient values in the order of `names`; cov: their covariance.
    Returns (b, se, z, p, chi2, chi2_p): arrays over `terms` for the first four,
    floats for the joint chi-square (df = len(terms)) and its p-value.
    """
    idx = [names.index(t) for t in terms]
    b = np.asarray(params)[idx]
    V = cov[np.ix_(idx, idx)]
    se = np.sqrt(np.diag(V))
    z = b / se
    p_individual = 2 * sp_stats.norm.sf(np.abs(z))
    wald_chi2 = float(b @ np.linalg.solve(V, b))
    wald_p = float(sp_stats.chi2.sf(wald_chi2, df=len(b)))
    return b, se, z, p_individual, wald_chi2, wald_p
