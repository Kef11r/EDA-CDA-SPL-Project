import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(style="whitegrid")

def plot_histogram(df: pd.DataFrame, column: str, bins: int = 30) -> None:
    """Plots a histogram for a numerical variable with a KDE curve."""
    # Побудова гістограми числової змінної.
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column], bins=bins, kde=True)
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()


def plot_boxplot(df: pd.DataFrame, column: str) -> None:
    """Generates a boxplot to visualize distribution and outliers."""
    # Бохсплот для виявлення викидів.
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot of {column}")
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, columns: str = None) -> None:
    """Visualizes the correlation matrix as a heatmap."""
    # Теплова карта кореляції.
    if columns:
        data = df[list(columns)]
    else:
        data = df.select_dtypes(include="number")
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()


def plot_bar_counts(df: pd.DataFrame, column: str, top_n: int = 10) -> None:
    """Plots a bar chart for the top N frequent categories."""
    # Barplot частот категоріальної змінної.
    counts = df[column].value_counts().head(top_n)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=counts.values, y=counts.index)
    plt.title(f"Top {top_n} categories of {column}")
    plt.xlabel("Count")
    plt.ylabel(column)
    plt.show()


def plot_pairplot(df: pd.DataFrame, columns: str = None) -> None:
    """Generates a pairplot to visualize relationships between numerical variables."""
    if columns:
        sns.pairplot(df[columns])
    else:
        sns.pairplot(df)
    plt.show()