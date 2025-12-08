# EDA-CDA-SPL-Project
# stat_analyzer

`stat_analyzer` це невелика навчальна бібліотека для EDA та перевірки статистичних гіпотез на основі датасету, що вказаний у конфігурації з можливістю:

- виконувати базовий EDA (shape, пропуски, описова статистика, категоріальні частоти, кореляції)
- автоматично підбирати можливі статистичні тести на основі типів змінних
- запускати тести (Pearson, Spearman, t test, Mann Whitney, ANOVA, Kruskal, Chi square)
- використовувати наперед задані гіпотези з конфігурацій `presets.py`
- отримувати рекомендації від LLM (моделі через OpenRouter) щодо вибору тестів
- будувати базові графіки (гістограми, boxplot, heatmap, barplot, pairplot)
- взаємодіяти як через інтерфейс командного рядка, так і як звичайну Python бібліотеку через `import`


## 1. Встановлення та запуск через CLI

### 1.1. Клонування репозиторію

```bash
git clone <URL_твого_репозиторію>
cd EDA-CDA-SPL-Project
```

### 1.2. Створити та активувати віртуальне оточення
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# або cmd
.\.venv\Scripts\activate.bat
```
### 1.3. Встановити залежності
```
pip install pandas scipy matplotlib seaborn
pip install langchain-openai langchain-core openai python-dotenv
```
### 1.4. Підготовка .env для АІ
```
OPENROUTER_URL="https://openrouter.ai/api/v1"
GEMINI_API_KEY="sk-or-v1-...тут_твій_OpenRouter_API_ключ..."
```
### 1.5. Запуск інтерфейсу

Запускати через термінал 'python -m stat_analyzer' python package
Вигляд меню:
```
=== Меню аналізу vgsales ===
1. Базовий EDA
2. Перевірити власну гіпотезу (обрати змінні)
3. Запустити всі наперед задані гіпотези
4. Побудувати графіки
0. Вихід
```

## 2. Використання бібліотеки через import
### 2.1. EDA функції (stat_analyzer.eda):
```
load_data(path: Path | str) завантажує CSV датасет

basic_info(df) друкує форму, типи, пропуски

numerical_summary(df, columns=None) повертає .describe() по числових

categorical_summary(df, columns=None, top_n=5) повертає топ значень по категоріях

correlation_matrix(df, columns=None) будує кореляційну матрицю по числових
```
### 2.2. Статистичні тести та авто підбір (stat_analyzer.hypothesis_tests):
```
suggest_tests(df, col1, col2) повертає список тестів для пари змінних

run_test_by_name(df, test_name, col1, col2) запускає відповідний тест

run_or_suggest(...) або запускає тест, або повертає список можливих
```
### 2.3. Візуалізації (stat_analyzer.hypothesis_tests.plots):
```
plot_histogram(df, column). Будує гістограму для числової змінної.
Використовується для оцінки розподілу даних, пошуку асиметрії, мод, вибросів.

plot_boxplot(df, column). Створює boxplot для вибраної числової змінної.
Дає змогу побачити медіану, межі квартилів та можливі аномалії.

plot_correlation_heatmap(df). Будує теплову карту кореляцій між усіма числовими змінними у датасеті.
Зручно для виявлення лінійних залежностей.

plot_bar_counts(df, column). Показує кількість появ кожної категорії у вибраній категоріальній змінній.
Корисно для аналізу балансу класів та частот.

plot_pairplot(df). Будує набір парних графіків (scatterplot matrix) між числовими змінними.
Дозволяє візуально оцінити можливі залежності та структуру даних.
```





