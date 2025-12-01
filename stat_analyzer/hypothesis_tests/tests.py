from typing import Dict
import pandas as pd
from scipy import stats

ResultDict = Dict[str, float]
def run_pearson(df: pd.DataFrame, col1: str,
                col2: str) -> ResultDict:
    x = df[col1].astype(float)
    y = df[col2].astype(float)
    stat, p = stats.pearsonr(x, y)
    return {
        "test": "pearson",
        "statistic": float(stat),
        "p_value": float(p),
    }

def run_spearman(df: pd.DataFrame, col1: str,
                 col2: str) -> ResultDict:
    x = df[col1].astype(float)
    y = df[col2].astype(float)
    stat, p = stats.spearmanr(x, y)
    return {
        "test": "spearman",
        "statistic": float(stat),
        "p_value": float(p),
    }

def run_ttest_ind(df: pd.DataFrame, group_col: str,
                  target_col: str) -> ResultDict:
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError("t тест вимагає рівно дві групи")
    g1 = df[df[group_col] == groups[0]][target_col].astype(float)
    g2 = df[df[group_col] == groups[1]][target_col].astype(float)
    stat, p = stats.ttest_ind(g1, g2, equal_var=False, nan_policy="omit")
    return {
        "test": "ttest",
        "statistic": float(stat),
        "p_value": float(p),
    }

def run_mannwhitney(df: pd.DataFrame, group_col: str,
                    target_col: str,) -> ResultDict:
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError("Mann Whitney вимагає рівно дві групи")
    g1 = df[df[group_col] == groups[0]][target_col].astype(float)
    g2 = df[df[group_col] == groups[1]][target_col].astype(float)
    stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    return {
        "test": "mannwhitney",
        "statistic": float(stat),
        "p_value": float(p),
    }

def run_anova(df: pd.DataFrame, group_col: str,
              target_col: str) -> ResultDict:
    groups = []
    for g in df[group_col].dropna().unique():
        groups.append(df[df[group_col] == g][target_col].astype(float))
    if len(groups) < 2:
        raise ValueError("ANOVA вимагає щонайменше дві групи")
    stat, p = stats.f_oneway(*groups)
    return {
        "test": "anova",
        "statistic": float(stat),
        "p_value": float(p),
    }

def run_kruskal(df: pd.DataFrame, group_col: str,
                target_col: str) -> ResultDict:
    groups = []
    for g in df[group_col].dropna().unique():
        groups.append(df[df[group_col] == g][target_col].astype(float))
    if len(groups) < 2:
        raise ValueError("Kruskal вимагає щонайменше дві групи")
    stat, p = stats.kruskal(*groups)
    return {
        "test": "kruskal",
        "statistic": float(stat),
        "p_value": float(p),
    }

def run_chi(df: pd.DataFrame, col1: str,
             col2: str) -> ResultDict:
    table = pd.crosstab(df[col1], df[col2])
    stat, p, dof, expected = stats.chi2_contingency(table)
    return {
        "test": "chi2",
        "statistic": float(stat),
        "p_value": float(p),
        "dof": float(dof),
    }
