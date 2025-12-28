import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def data_summary(df: pd.DataFrame):
    """Generate a summary of the DataFrame including data types, 
    non-null counts, and basic statistics.

    Args:
        df (pd.DataFrame): The DataFrame to summarize.
    """

    summary = df.describe()

    return summary


def plot_histogram_for_numeric_columns(df: pd.DataFrame):
    """Plot histograms for all numeric columns in the DataFrame.

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
    """Plot boxplots for all numeric columns in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing numeric columns.
    """

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols.remove('Churn Value')

    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col])
        plt.show()
