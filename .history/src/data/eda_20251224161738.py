import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def data_summary(df: pd.DataFrame):
    """Generate a summary of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to summarize.
    """

    summary = df.describe()

    return summary


def plot_histogram_for_numeric_columns(df: pd.DataFrame):
    """Plot histograms for all numeric columns in the DataFrame to check the frequency distribution.

    Args:
        df (pd.DataFrame): The DataFrame containing numeric columns.
    """

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols.remove('Churn Value')

    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(data=df, x=col, kde=True)
        plt.show()


def boxplot_for_numeric_columns(df: pd.DataFrame):
    """Plot boxplots for all numeric columns in the DataFrame to check relationship and the outlier between churn and numeric features.

    Args:
        df (pd.DataFrame): The DataFrame containing numeric columns.
    """

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols.remove('Churn Value')

    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df, x='Churn Value', y=col)
        plt.show()


def plot_heatmap_correlation(df: pd.DataFrame):
    """Plot a heatmap to visualize the correlation between numeric features in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing numeric columns.
    """

    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.show()
