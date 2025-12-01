import sys
import pandas as pd
from . import eda
from .hypothesis_tests import (HYPOTHESES, run_or_suggest, run_test_by_name,
)


def load_dataset() -> pd.DataFrame:
    df = eda.load_data()
    return df


def print_columns(df: pd.DataFrame) -> None:
    print("\nДоступні колонки:")
    for col in df.columns:
        print(f"  {col:20}  dtype = {df[col].dtype}")
    print()


def run_basic_eda(df: pd.DataFrame) -> None:
    print("\n=== Базова інформація про датасет ===")
    eda.basic_info(df)

    print("\n=== Описова статистика для числових змінних ===")
    print(eda.numerical_summary(df))

    print("\n=== Частоти для категоріальних змінних (топ 5) ===")
    cat_summary = eda.categorical_summary(df)
    for col, vc in cat_summary.items():
        print(f"\nКолонка: {col}")
        print(vc)
    print("\n=== Кореляційна матриця для числових змінних ===")
    print(eda.correlation_matrix(df))

def choose_columns(df: pd.DataFrame) -> tuple[str, str]:
    print_columns(df)
    col1 = input("Введіть назву першої змінної: ").strip()
    col2 = input("Введіть назву другої змінної: ").strip()

    if col1 not in df.columns or col2 not in df.columns:
        print("Помилка: одна з колонок не знайдена у датасеті.")
        raise SystemExit(1)
    return col1, col2

def run_hypothesis_interactive(df: pd.DataFrame) -> None:
    col1, col2 = choose_columns(df)
    description = input(
        "Коротко опишіть гіпотезу (можна залишити порожнім): "
    ).strip()
    if not description:
        description = f"Гіпотеза для змінних {col1} і {col2}"

    result = run_or_suggest(
        df,
        col1=col1,
        col2=col2,
        description=description,
        auto=False,
    )

    if result["mode"] == "run":
        print("\n=== Результат тесту ===")
        print(result["report"])
        return

    if result["mode"] == "none":
        print(result["message"])
        return

    if result["mode"] == "suggest":
        tests = result.get("possible_tests", [])
        if not tests:
            print(result["message"])
            return

        print("\n", result["message"])
        print("Можливі тести:")
        for i, tname in enumerate(tests, start=1):
            print(f"  {i}. {tname}")
        choice = input(
            "Оберіть номер тесту який запустити (або Enter щоб вийти): "
        ).strip()
        if not choice:
            return
        try:
            idx = int(choice) - 1
            test_name = tests[idx]
        except (ValueError, IndexError):
            print("Некоректний вибір.")
            return
        result_dict = run_test_by_name(df, test_name, col1, col2)
        from .hypothesis_tests.runner import interpret_result

        report = interpret_result(description, result_dict)
        print("\n=== Результат обраного тесту ===")
        print(report)


def run_presets(df: pd.DataFrame) -> None:
    from .hypothesis_tests import run_all_presets
    print("\n=== Запуск усіх наперед визначених гіпотез ===")
    reports = run_all_presets(df, HYPOTHESES, auto=True)
    for rep in reports:
        print(rep)
        print("-" * 60)

def print_menu() -> None:
    print("\n=== Меню аналізу vgsales ===")
    print("1. Базовий EDA")
    print("2. Перевірити власну гіпотезу (обрати змінні)")
    print("3. Запустити всі наперед задані гіпотези")
    print("0. Вихід")

def main() -> None:
    df = load_dataset()
    while True:
        print_menu()
        choice = input("Ваш вибір: ").strip()
        if choice == "1":
            run_basic_eda(df)
        elif choice == "2":
            run_hypothesis_interactive(df)
        elif choice == "3":
            run_presets(df)
        elif choice == "0":
            print("Завершення роботи.")
            sys.exit(0)
        else:
            print("Невідомий пункт меню, спробуйте ще раз.")
if __name__ == "__main__":
    main()
