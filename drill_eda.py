import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def compute_summary(df):
    numeric_df = df.select_dtypes(include=["number"])

    summary = pd.DataFrame({
        "count": numeric_df.count(),
        "mean": numeric_df.mean(),
        "median": numeric_df.median(),
        "std": numeric_df.std(),
        "min": numeric_df.min(),
        "max": numeric_df.max()
    }).T

    summary.to_csv("output/summary.csv")
    return summary


def plot_distributions(df, columns, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_correlation(df, output_path):
    numeric_df = df.select_dtypes(include=["number"])

    corr = numeric_df.corr(method="pearson")

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    os.makedirs("output", exist_ok=True)

    df = pd.read_csv("data/sample_sales.csv")

    compute_summary(df)

    df["total"] = df["quantity"] * df["unit_price"]

    columns_to_plot = ["quantity", "unit_price", "total", "quantity"]

    plot_distributions(df, columns_to_plot, "output/distributions.png")

    plot_correlation(df, "output/correlation.png")
if __name__ == "__main__":
    main()