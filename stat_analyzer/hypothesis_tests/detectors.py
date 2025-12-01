from typing import List
import pandas as pd


def detect_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "categorical"


def _ensure_category_first(df: pd.DataFrame, col1: str,
                           col2: str,) -> tuple[str, str]:

    t1 = detect_type(df[col1])
    t2 = detect_type(df[col2])

    if t1 == "categorical" and t2 == "numeric":
        return col1, col2
    if t1 == "numeric" and t2 == "categorical":
        return col2, col1
    return col1, col2


def suggest_tests(df: pd.DataFrame, col1: str,
                  col2: str,) -> List[str]:
    t1 = detect_type(df[col1])
    t2 = detect_type(df[col2])

    # numeric vs numeric
    if t1 == "numeric" and t2 == "numeric":
        return ["pearson", "spearman"]

    # categorical vs categorical
    if t1 == "categorical" and t2 == "categorical":
        return ["chi2"]

    # categorical vs numeric
    cat_col, num_col = _ensure_category_first(df, col1, col2)
    t_cat = detect_type(df[cat_col])
    t_num = detect_type(df[num_col])

    if t_cat == "categorical" and t_num == "numeric":
        n_groups = df[cat_col].nunique(dropna=True)
        if n_groups <= 1:
            return []
        if n_groups == 2:
            return ["ttest", "mannwhitney"]
        if n_groups >= 3:
            return ["anova", "kruskal"]
    return []
