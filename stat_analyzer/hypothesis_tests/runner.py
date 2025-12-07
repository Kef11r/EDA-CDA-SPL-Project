import json
import pandas as pd
from typing import Dict
from .detectors import detect_type, suggest_tests
from . import tests as test_impl

data = pd.read_csv("data/raw/vgsales.csv")

TEST_FUNCTIONS = {
    "pearson": test_impl.run_pearson,
    "spearman": test_impl.run_spearman,
    "ttest": test_impl.run_ttest_ind,
    "mannwhitney": test_impl.run_mannwhitney,
    "anova": test_impl.run_anova,
    "kruskal": test_impl.run_kruskal,
    "chi2": test_impl.run_chi,
}
def load_custom_test(test_config_path: str) -> Dict:
    """
    Загружаємо JSON конфігураційний файл з можливістю для користувача використання
    кастомних тестів
    """
    with open(test_config_path, 'r') as json_file:
        test_config = json.load(json_file)

    custom_tests_dict = {}
    for custom_test in test_config['custom_tests']:
        test_name = custom_test['name']
        test_func = globals().get(custom_test['function'])
        if test_func:
            custom_tests_dict[test_name] = test_func
    return custom_tests_dict

custom_tests = load_custom_test("test_config.json")
TEST_FUNCTIONS.update(custom_tests)

def run_or_suggest(df, col1, col2, description=None, auto=False):
    possible_tests = suggest_tests(df, col1, col2)
    if not possible_tests:
        return {"mode": "none", "message": "Не вдалося підібрати підходящий тест для цих змінних."}

    if auto:
        # If auto = True, run the first test from the list
        test_name = possible_tests[0]
        result = run_test_by_name(df, test_name, col1, col2)
        report = interpret_result(description or f"Автоматична гіпотеза для {col1} і {col2}", result)
        return {"mode": "run", "used_test": test_name, "result": result, "report": report}
    return {"mode": "suggest", "message": "Можна застосувати кілька тестів, оберіть потрібний.",
            "possible_tests": possible_tests}

def run_test_by_name(df, test_name, col1, col2):
    if test_name not in TEST_FUNCTIONS:
        raise ValueError(f"Невідомий тест {test_name!r}")
    test_func = TEST_FUNCTIONS[test_name]
    return test_func(df, col1, col2)

def interpret_result(description, result, alpha=0.05):
    test_name = result.get("test", "unknown")
    statistic = result.get("statistic", float("nan"))
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
        f"Статистика: {statistic:.4f}\n"
        f"p value: {p_value:.4f}\n"
        f"Рішення при alpha = {alpha:.4f}: {decision}"
    )
    return report

def run_all_presets(df, presets, auto=True):
    reports = []
    for hypothesis in presets:
        cols = hypothesis["cols"]
        description = hypothesis.get("description", f"Гіпотеза для {cols}")
        col1, col2 = cols[0], cols[1]
        # Complete test based on a hypothesis
        result = run_or_suggest(df, col1, col2, description=description, auto=auto)
        if result["mode"] == "run":
            reports.append(result["report"])
        else:
            msg = (
                f"Для гіпотези {hypothesis.get('name', cols)} "
                f"не вдалося автоматично запустити тест. "
                f"Можливі тести: {result.get('possible_tests', [])}"
            )
            reports.append(msg)
    return reports