import sys
import pandas as pd
from .eda import load_data, basic_info, numerical_summary, categorical_summary, correlation_matrix
from .hypothesis_tests import (HYPOTHESES, run_test_by_name, suggest_tests)
from stat_analyzer.ai.ai_agent import recommend_tests_from_hypothesis
from stat_analyzer.hypothesis_tests.runner import load_custom_test
from .hypothesis_tests.plots import (
    plot_histogram,
    plot_boxplot,
    plot_correlation_heatmap,
    plot_bar_counts,
    plot_pairplot,
)


def load_dataset() -> pd.DataFrame:
    """Load dataset from eda.py"""
    df = load_data()
    return df

def print_columns(df: pd.DataFrame) -> None:
    """All columns with dtypes"""
    print("\nДоступні колонки:")
    for col in df.columns:
        print(f"{col:20} dtype = {df[col].dtype}")
    print()

def run_basic_eda(df: pd.DataFrame) -> None:
    """Main info about the dataset"""
    print("\n=== Базова інформація про датасет ===")
    basic_info(df)

    print("\n=== Описова статистика для числових змінних ===")
    print(numerical_summary(df))

    print("\n=== Частоти для категоріальних змінних (топ 5) ===")
    cat_summary = categorical_summary(df)
    for col, vc in cat_summary.items():
        print(f"\nКолонка: {col}")
        print(vc)

    print("\n=== Кореляційна матриця для числових змінних ===")
    print(correlation_matrix(df))

def choose_columns(df: pd.DataFrame) -> tuple[str, str]:
    """User chooses columns to use"""
    print("\n Доступні колонки:")
    for idx, col in enumerate(df.columns):
        print(f"{idx}: {col}")
    try:
        col1_idx = int(input("Enter the first column index: "))
        col2_idx = int(input("Enter the second column index: "))
        if col1_idx not in range(len(df.columns)) and col2_idx not in range(len(df.columns)):
            print("Обрано помилкові індекси.")
            raise SystemExit(1)
        col1 = df.columns[col1_idx]
        col2 = df.columns[col2_idx]
    except ValueError:
        print("Потрібно ввести ціле число.")
        raise SystemExit(1)
    return col1, col2

def run_hypothesis_interactive(df: pd.DataFrame) -> None:
    """Chooses the hypothesis test and the run of the test"""
    col1, col2 = choose_columns(df)
    description = input("Коротко опишіть гіпотезу (можна залишити порожнім): ").strip()
    if not description:
        description = f"Гіпотеза для змінних {col1} і {col2}"
    available_tests = suggest_tests(df, col1, col2)

    # AI agent start
    print("\n=== АІ рекомендація щодо вибору тесту===")
    print("Sending request to LLM…")
    try:
        ai_result = recommend_tests_from_hypothesis(description, available_tests)
        rec_tests = ai_result["recommended_tests"]
        explanation = ai_result["explanation"]
        columns_comment = ai_result["columns_comment"]
        if rec_tests:
            print("\nAI рекомендує такі тести (у порядку пріоритету):")
            for i, t in enumerate(rec_tests, start=1):
                print(f"{i}. {t}")
        else:
            print("\nAI не зміг обрати конкретні тести, показую всі технічно можливі.")

        if columns_comment:
            print("\nКоментар від АІ щодо датасету:")
            print(columns_comment)

        print("\nПояснення від AI:")
        print(explanation)
    except Exception as e:
        print(f"\nНе вдалося отримати відповідь від АІ. Помилка: {e}")
        rec_tests = []

    if rec_tests:
        ordered_tests = rec_tests + [t for t in available_tests if t not in rec_tests]
    else:
        ordered_tests = available_tests
    print("\nМожливі тести для цієї пари змінних:")
    for i, tname in enumerate(ordered_tests, start=1):
        print(f"  {i}. {tname}")
    choice = input(
        "Оберіть номер тесту який запустити (або Enter щоб вийти): ").strip()
    if not choice:
        return
    try:
        idx = int(choice) - 1
        test_name = ordered_tests[idx]
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

def run_plots(df: pd.DataFrame) -> None:
    while True:
        print("\n=== Графіки ===")
        print("1. Histogram (гістограма)")
        print("2. Boxplot (ящик з вусами)")
        print("3. Correlation Heatmap (теплова карта)")
        print("4. Bar Counts (частоти категорій)")
        print("5. Pairplot (парні графіки)")
        print("0. Назад")

        choice = input("Ваш вибір: ").strip()

        if choice == "1":
            print_columns(df)
            col = input("Колонка для гістограми: ").strip()
            plot_histogram(df, col)

        elif choice == "2":
            print_columns(df)
            col = input("Колонка для boxplot: ").strip()
            plot_boxplot(df, col)

        elif choice == "3":
            plot_correlation_heatmap(df)

        elif choice == "4":
            print_columns(df)
            col = input("Колонка для barplot: ").strip()
            plot_bar_counts(df, col)

        elif choice == "5":
            print("Створюю pairplot... Це може тривати декілька секунд.")
            plot_pairplot(df)

        elif choice == "0":
            return

        else:
            print("Некоректний вибір.")

def print_menu() -> None:
    print("\n=== Меню аналізу vgsales ===")
    print("1. Базовий EDA")
    print("2. Перевірити власну гіпотезу (обрати змінні)")
    print("3. Запустити всі наперед задані гіпотези")
    print("4. Побудувати графіки")
    print("0. Вихід")

def main() -> None:
    """Main function"""
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
        elif choice == "4":
            run_plots(df)
        elif choice == "0":
            print("Завершення роботи.")
            sys.exit(0)
        else:
            print("Невідомий пункт меню, спробуйте ще раз.")
if __name__ == "__main__":
    main()
