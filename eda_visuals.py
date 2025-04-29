"""
File: eda_visuals.py
Purpose: Run exploratory data analysis (EDA).
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

#%%
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

run_eda

#%%
# Data Loading and Overview
data_path = "cleaned_data.csv"
df = pd.read_csv(data_path)

#%%
# EDA for Life Expectancy Dataset
# 
# This script loads and cleans the data, runs descriptive statistics,
# and creates various visualizations for univariate, bivariate, and 
# multivariate analysis. It also demonstrates how to perform a simple 
# statistical test (t-test) as part of significance testing.

# Set visual style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


# Display first few rows, structure and summary statistics
print("Data Head:")
print(df.head(), "\n")

print("Data Info:")
print(df.info(), "\n")

print("Descriptive Statistics:")
print(df.describe(), "\n")

# Check for missing values
print("Missing Values by Column:")
print(df.isnull().sum(), "\n")


#%%
# Univariate Analysis
# Plotting histograms for all numeric columns
df.hist(bins=20, edgecolor='black', figsize=(15, 10))
plt.tight_layout()
plt.suptitle("Histograms for Numeric Variables", y=1.02)
plt.show()

# Boxplots to inspect outliers
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# Violin plots
if 'Year' in df.columns:
    plt.figure()
    sns.violinplot(x="Year", y="Life expectancy", data=df)
    plt.title("Life Expectancy Distribution by Year")
    plt.show()
    
#%%
# Bivariate Analysis
# Correlation heatmap of numeric variables
corr_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Scatter plot between two key variables: GDP vs Life expectancy (adjust column names as in your data)
if set(['GDP', 'Life expectancy']).issubset(df.columns):
    plt.figure()
    sns.scatterplot(x="GDP", y="Life expectancy", data=df)
    plt.title("GDP vs. Life Expectancy")
    plt.xlabel("GDP")
    plt.ylabel("Life Expectancy")
    plt.show()

# Use a regression plot (from Seaborn) to overlay a fitted line
if set(['GDP', 'Life expectancy']).issubset(df.columns):
    plt.figure()
    sns.regplot(x="GDP", y="Life expectancy", data=df, scatter_kws={'s':50}, ci=95)
    plt.title("Regression Plot: GDP vs Life Expectancy")
    plt.show()


#%%
# Multivariate Analysis
# Pairplot of selected features (adjust list as needed)
selected_features = ['Life expectancy', 'GDP', 'Adult Mortality', 'Schooling']
available_features = [col for col in selected_features if col in df.columns]
if len(available_features) > 1:
    sns.pairplot(df[available_features])
    plt.suptitle("Pairplot of Selected Features", y=1.02)
    plt.show()

# Pivot table example
if 'Year' in df.columns and 'Life expectancy' in df.columns:
    pivot_table = pd.pivot_table(df, values='Life expectancy', index='Year', aggfunc=np.mean)
    print("Pivot Table: Average Life Expectancy by Year")
    print(pivot_table)
    pivot_table.plot(kind='bar', legend=False)
    plt.title("Average Life Expectancy by Year")
    plt.ylabel("Life Expectancy")
    plt.show()

# Significance Testing Example
if 'Life expectancy' in df.columns:
    sample = df['Life expectancy'].dropna()
    hypothesized_mean = 70
    t_stat, p_value = stats.ttest_1samp(sample, popmean=hypothesized_mean)
    print(f"One-sample t-test for Life Expectancy (mu = {hypothesized_mean}):")
    print("t-statistic: {:.3f}, p-value: {:.3f}".format(t_stat, p_value))
    if p_value < 0.05:
        print("Result: Reject the null hypothesis at the 5% significance level. The mean is significantly different from 70.\n")
    else:
        print("Result: Fail to reject the null hypothesis. The mean is not significantly different from 70.\n")

# Feature Engineering Example
if 'Life expectancy' in df.columns:
    bins = [0, 65, 75, np.inf]
    labels = ['Low', 'Medium', 'High']
    df['LE_category'] = pd.cut(df['Life expectancy'], bins=bins, labels=labels)
    print("Counts by Life Expectancy Category:")
    print(df['LE_category'].value_counts(), "\n")
    
    df['LE_category'].value_counts().plot(kind='bar')
    plt.title("Counts of Life Expectancy Categories")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.show()

#%%
# Additional Insights
if set(['GDP', 'LE_category']).issubset(df.columns):
    avg_gdp_by_category = df.groupby('LE_category')['GDP'].mean().reset_index()
    print("Average GDP by Life Expectancy Category:")
    print(avg_gdp_by_category, "\n")
    sns.barplot(x='LE_category', y='GDP', data=avg_gdp_by_category)
    plt.title("Average GDP by Life Expectancy Category")
    plt.show()

#%%
if 'Year' in df.columns:
    plt.figure(figsize=(10,5))
    df.groupby('Year')['Life expectancy'].mean().plot(marker='o')
    plt.title("Global Average Life Expectancy Over Time")
    plt.ylabel("Life Expectancy")
    plt.xlabel("Year")
    plt.grid(True)
    plt.show()

#%%
from sklearn.decomposition import PCA
import numpy as np

# pick numeric features, drop missing rows
features = df.select_dtypes(include=[np.number]).drop(columns=['Year'], errors='ignore')
X = features.dropna()
pca = PCA(n_components=2)
pcs = pca.fit_transform(X)
pca_df = pd.DataFrame(pcs, columns=['PC1','PC2'], index=X.index)

plt.figure(figsize=(8,6))
sns.scatterplot(x='PC1', y='PC2', data=pca_df, alpha=0.7)
plt.title(f"PCA (2 components) — {pca.explained_variance_ratio_.sum():.2%} variance explained")
plt.show()

#%%
from sklearn.cluster import KMeans

# elbow method
inertia = []
K = range(1,10)
for k in K:
    inertia.append(KMeans(n_clusters=k, random_state=42).fit(pca_df).inertia_)
plt.figure()
plt.plot(K, inertia, '-o')
plt.xlabel('k'); plt.ylabel('Inertia'); plt.title('Elbow Method for KMeans')
plt.show()

# pick k (e.g. 3) and visualize
k = 3
km = KMeans(n_clusters=k, random_state=42).fit(pca_df)
pca_df['Cluster'] = km.labels_
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette='tab10', data=pca_df)
plt.title(f"KMeans (k={k}) on PCA space")
plt.show()
# %%
