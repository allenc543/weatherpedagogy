"""Reusable statistics computation helpers."""
import numpy as np
import pandas as pd
from scipy import stats


def descriptive_stats(series):
    """Compute comprehensive descriptive statistics for a numeric series."""
    return {
        "count": len(series),
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "min": series.min(),
        "max": series.max(),
        "q25": series.quantile(0.25),
        "q75": series.quantile(0.75),
        "iqr": series.quantile(0.75) - series.quantile(0.25),
        "skewness": series.skew(),
        "kurtosis": series.kurtosis(),
    }


def bootstrap_ci(data, stat_func=np.mean, n_boot=1000, ci=95, seed=42):
    """Compute bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    boot_stats = []
    for _ in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stats.append(stat_func(sample))
    lower = np.percentile(boot_stats, (100 - ci) / 2)
    upper = np.percentile(boot_stats, 100 - (100 - ci) / 2)
    return lower, upper, boot_stats


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (group1.mean() - group2.mean()) / pooled_std


def normality_test(data):
    """Run Shapiro-Wilk test (on sample if too large) and return stat, p-value."""
    if len(data) > 5000:
        data = np.random.RandomState(42).choice(data, 5000, replace=False)
    stat, p = stats.shapiro(data)
    return stat, p


def correlation_matrix(df, method="pearson"):
    """Compute correlation matrix for numeric columns."""
    numeric = df.select_dtypes(include=[np.number])
    return numeric.corr(method=method)


def perform_ttest(group1, group2, equal_var=False):
    """Perform independent samples t-test."""
    stat, p = stats.ttest_ind(group1, group2, equal_var=equal_var)
    d = cohens_d(group1, group2)
    return {"t_stat": stat, "p_value": p, "cohens_d": d}


def perform_anova(*groups):
    """Perform one-way ANOVA."""
    stat, p = stats.f_oneway(*groups)
    return {"f_stat": stat, "p_value": p}


def ks_test(data1, data2):
    """Perform Kolmogorov-Smirnov two-sample test."""
    stat, p = stats.ks_2samp(data1, data2)
    return {"ks_stat": stat, "p_value": p}
