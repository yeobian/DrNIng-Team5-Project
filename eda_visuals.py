"""
File: eda_visuals.py
Purpose: Run exploratory data analysis (EDA).
"""

import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(df):
    # Basic summary
    print(df.describe())
    
    # Histogram
    df.hist(bins=20, figsize=(15, 10))
    plt.tight_layout()
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()
