from typing import Dict, List, Optional
import pandas as pd
from .detectors import detect_type, suggest_tests
from . import tests as test_impl


TEST_FUNCTIONS = {
    "pearson": test_impl.run_pearson,
    "spearman": test_impl.run_spearman,
    "ttest": test_impl.run_ttest_ind,
    "mannwhitney": test_impl.run_mannwhitney,
    "anova": test_impl.run_anova,
    "kruskal": test_impl.run_kruskal,
    "chi2": test_impl.run_chi,
}

def _normalize_cat_num(df: pd.DataFrame, col1: str,
                       col2: str,) -> tuple[str, str]:
    t1 = detect_type(df[col1])
    t2 = detect_type(df[col2])

    if t1 == "categorical" and t2 == "numeric":
        return col1, col2
    if t1 == "numeric" and t2 == "categorical":
        return col2, col1
    raise ValueError("Очікується одна категоріальна і одна числова змінна")

def run_test_by_name(df: pd.DataFrame, test_name: str,
                     col1: str,
                     col2: str,) -> Dict[str, float]:
    if test_name not in TEST_FUNCTIONS:
        raise ValueError(f"Невідомий тест {test_name!r}")
    if test_name in {"pearson", "spearman", "chi2"}:
        func = TEST_FUNCTIONS[test_name]
        return func(df, col1, col2)
    if test_name in {"ttest", "mannwhitney", "anova", "kruskal"}:
        group_col, target_col = _normalize_cat_num(df, col1, col2)
        func = TEST_FUNCTIONS[test_name]
        return func(df, group_col, target_col)
    func = TEST_FUNCTIONS[test_name]
    return func(df, col1, col2)


def interpret_result(description: str, result: Dict[str, float],
                     alpha: float = 0.05,) -> str:

    test_name = result.get("test", "unknown")
    stat = result.get("statistic", float("nan"))
    p_value = result.get("p_value", float("nan"))

    if p_value < alpha:
        decision = (
            "Нульова гіпотеза відхиляється. "
            "Є статистично значуща різниця або залежність."
        )
    else:
        decision = (
            "Немає підстав відхиляти нульову гіпотезу. "
            "Статистично значущої різниці або залежності не виявлено."
        )

    report = (
        f"Гіпотеза: {description}\n"
        f"Тест: {test_name}\n"
        f"Статистика: {stat:.4f}\n"
        f"p value: {p_value:.4f}\n"
        f"Рішення при alpha = {alpha:.2f}: {decision}"
    )
    return report

def infer_preferred_from_text(
    description: str,
    possible_tests: list[str],
) -> Optional[str]:
    """
    Спробувати обрати тест на основі тексту гіпотези.
    Ключові ідеї:
    - "кореляц", "зв'язок" -> pearson або spearman
    - "різниц" / "відрізня" / "порівня" -> ttest або anova
    - "робастн", "непараметрич" -> mannwhitney або kruskal
    - "незалежн", "залежн" для категоріальних змінних -> chi2
    """
    if not description:
        return None

    text = description.lower()

    def pick_first(candidates: list[str]) -> Optional[str]:
        for name in candidates:
            if name in possible_tests:
                return name
        return None

    # Кореляція
    if any(word in text for word in ["кореляц", "зв'язок", "зв’язок", "зв`язок", "correlat"]):
        preferred = pick_first(["pearson", "spearman"])
        if preferred:
            return preferred

    # Непараметричні, робастні методи
    if any(word in text for word in ["робастн", "непараметрич", "без припущень"]):
        preferred = pick_first(["mannwhitney", "kruskal", "spearman"])
        if preferred:
            return preferred

    # Різниця між групами, порівняння середніх
    if any(word in text for word in ["різниц", "відрізня", "порівня", "mean", "середн"]):
        preferred = pick_first(["ttest", "anova"])
        if preferred:
            return preferred

    # Незалежність, залежність між категоріями
    if any(word in text for word in ["незалежн", "залежн", "independenc", "dependenc"]):
        preferred = pick_first(["chi2"])
        if preferred:
            return preferred

    return None

def run_or_suggest(df: pd.DataFrame, col1: str, col2: str,
                   description: Optional[str] = None, auto: bool = False, preferred: Optional[str] = None,
                   alpha: float = 0.05,) -> Dict[str, object]:
    tests = suggest_tests(df, col1, col2)
    if not tests:
        return {
            "mode": "none",
            "message": "Не вдалося підібрати підходящий тест для цих змінних.",
            "possible_tests": [],
        }

    if preferred is not None:
        if preferred not in tests:
            return {
                "mode": "suggest",
                "message": (
                    f"Тест {preferred!r} не підходить для цієї пари змінних. "
                    f"Можливі тести: {tests}."
                ),
                "possible_tests": tests,
            }
        result = run_test_by_name(df, preferred, col1, col2)
        desc = description or f"Автоматична гіпотеза для {col1} і {col2}"
        report = interpret_result(desc, result, alpha=alpha)
        return {
            "mode": "run",
            "used_test": preferred,
            "result": result,
            "report": report,
        }

    if len(tests) == 1 and auto:
        test_name = tests[0]
        result = run_test_by_name(df, test_name, col1, col2)
        desc = description or f"Автоматична гіпотеза для {col1} і {col2}"
        report = interpret_result(desc, result, alpha=alpha)
        return {
            "mode": "run",
            "used_test": test_name,
            "result": result,
            "report": report,
        }

    if len(tests) == 1 and not auto:
        return {
            "mode": "suggest",
            "message": "Для цієї пари змінних підходить один тест.",
            "possible_tests": tests,
        }
    if len(tests) > 1 and auto:
        test_name = tests[0]
        result = run_test_by_name(df, test_name, col1, col2)
        desc = description or f"Автоматична гіпотеза для {col1} і {col2}"
        report = interpret_result(desc, result, alpha=alpha)
        return {
            "mode": "run",
            "used_test": test_name,
            "available_tests": tests,
            "result": result,
            "report": report,
        }
    return {
        "mode": "suggest",
        "message": "Можна застосувати кілька тестів, оберіть потрібний.",
        "possible_tests": tests,
    }


def run_all_presets(df: pd.DataFrame, presets: List[dict],
                    auto: bool = True, alpha: float = 0.05) -> List[str]:
    reports: List[str] = []
    for h in presets:
        cols = h["cols"]
        description = h.get("description", f"Гіпотеза для {cols}")
        col1, col2 = cols[0], cols[1]

        result = run_or_suggest(
            df,
            col1=col1,
            col2=col2,
            description=description,
            auto=auto,
            alpha=alpha,
        )
        if result["mode"] == "run":
            reports.append(result["report"])
        else:
            msg = (
                f"Для гіпотези {h.get('name', cols)} "
                f"не вдалося автоматично запустити тест. "
                f"Можливі тести: {result.get('possible_tests', [])}"
            )
            reports.append(msg)
    return reports
