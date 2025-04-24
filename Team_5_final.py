"""
File: data_cleaning.py
Purpose: Clean life expectancy dataset and prepare for merging with socio-economic factors.
"""
#%%
import pandas as pd

#%%
df = pd.read_csv("world_bank_indicators.csv")
df

#%%
imp_columns = ["country", "year", 
           "Total debt service (% of exports of goods, services and primary income)", 
           "Total natural resources rents (% of GDP)",
           "Total reserves (includes gold, current US$)",
           "Trained teachers in primary education (% of total teachers)",
           "Unemployment, female (% of female labor force) (modeled ILO estimate)",
           "Unemployment, male (% of male labor force) (modeled ILO estimate)",
           "Unemployment, total (% of total labor force) (modeled ILO estimate)",
           "Cause of death, by communicable diseases and maternal, prenatal and nutrition conditions (% of total)",
           "Cause of death, by injury (% of total)",
           "Cause of death, by non-communicable diseases (% of total)",
           "Central government debt, total (% of GDP)",
           "Children out of school, primary, female",
           "Children out of school, primary, male",
           "Death rate, crude (per 1,000 people)",
           "Expense (% of GDP)",
           "GDP (current US$)",
           "GDP per capita (current US$)",
           "Hospital beds (per 1,000 people)",
           "Life expectancy at birth, female (years)",
           "Life expectancy at birth, male (years)",
           "Life expectancy at birth, total (years)",
           "Literacy rate, adult female (% of females ages 15 and above)",
           "Literacy rate, adult male (% of males ages 15 and above)",
           "Literacy rate, adult total (% of people ages 15 and above)",
           "Literacy rate, youth female (% of females ages 15-24)",
           "Literacy rate, youth male (% of males ages 15-24)",
           "Literacy rate, youth total (% of people ages 15-24)",
           "Population ages 0-14 (% of total population)",
           "Population ages 15-64 (% of total population)",
           "Population ages 65 and above (% of total population)",
           "Population, female (% of total population)",
           "Population, total",
           "Refugee population by country or territory of origin",
           "Rural population",
           "Rural population (% of total population)",
           "Tax revenue (% of GDP)",
           "Total reserves (includes gold, current US$)",          
          ]
df = df[imp_columns]

#%%
df.info()

#%%
columns_with_nulls = df.isna().sum()
columns_with_high_nulls = columns_with_nulls[columns_with_nulls > 5000]
columns_with_low_nulls = columns_with_nulls[columns_with_nulls < 5000]

print(columns_with_high_nulls)

#%%
columns_with_low_nulls

#%%
# Converting country names to uppercase
df['country'] = df['country'].str.upper()
df['country'].unique()

#%%
# Country codes with full names for mapping.
# Define a dictionary mapping ISO ALPHA-3 country codes to full country names
country_code_to_name = {
    'ABW': 'Aruba', 'AFE': 'Africa Eastern and Southern', 'AFG': 'Afghanistan', 'AFW': 'Africa Western and Central', 
    'AGO': 'Angola', 'ALB': 'Albania', 'AND': 'Andorra', 'ARB': 'Arab World', 'ARE': 'United Arab Emirates',
    'ARG': 'Argentina', 'ARM': 'Armenia', 'ASM': 'American Samoa', 'ATG': 'Antigua and Barbuda', 
    'AUS': 'Australia', 'AUT': 'Austria', 'AZE': 'Azerbaijan', 'BDI': 'Burundi', 'BEL': 'Belgium',
    'BEN': 'Benin', 'BFA': 'Burkina Faso', 'BGD': 'Bangladesh', 'BGR': 'Bulgaria', 'BHR': 'Bahrain', 
    'BHS': 'Bahamas', 'BIH': 'Bosnia and Herzegovina', 'BLR': 'Belarus', 'BLZ': 'Belize',
    'BMU': 'Bermuda', 'BOL': 'Bolivia', 'BRA': 'Brazil', 'BRB': 'Barbados', 'BRN': 'Brunei Darussalam', 
    'BTN': 'Bhutan', 'BWA': 'Botswana', 'CAF': 'Central African Republic', 'CAN': 'Canada',
    'CEB': 'Central Europe and the Baltics', 'CHE': 'Switzerland', 'CHI': 'Channel Islands', 'CHL': 'Chile', 
    'CHN': 'China', 'CIV': 'Côte d\'Ivoire', 'CMR': 'Cameroon', 'COD': 'Congo (Kinshasa)',
    'COG': 'Congo (Brazzaville)', 'COL': 'Colombia', 'COM': 'Comoros', 'CPV': 'Cape Verde', 
    'CRI': 'Costa Rica', 'CSS': 'Caribbean small states', 'CUB': 'Cuba', 'CUW': 'Curaçao', 
    'CYM': 'Cayman Islands', 'CYP': 'Cyprus', 'CZE': 'Czech Republic', 'DEU': 'Germany', 
    'DJI': 'Djibouti', 'DMA': 'Dominica', 'DNK': 'Denmark', 'DOM': 'Dominican Republic', 
    'DZA': 'Algeria', 'EAP': 'East Asia & Pacific', 'EAR': 'Early-demographic dividend', 
    'EAS': 'East Asia & Pacific (excluding high income)', 'ECA': 'Europe & Central Asia', 
    'ECS': 'Europe & Central Asia (excluding high income)', 'ECU': 'Ecuador', 'EGY': 'Egypt', 
    'EMU': 'Euro area', 'ERI': 'Eritrea', 'ESP': 'Spain', 'EST': 'Estonia', 'ETH': 'Ethiopia', 
    'EUU': 'European Union', 'FCS': 'Fragile and conflict affected situations', 'FIN': 'Finland', 
    'FJI': 'Fiji', 'FRA': 'France', 'FRO': 'Faroe Islands', 'FSM': 'Micronesia, Federated States of', 
    'GAB': 'Gabon', 'GBR': 'United Kingdom', 'GEO': 'Georgia', 'GHA': 'Ghana', 'GIB': 'Gibraltar', 
    'GIN': 'Guinea', 'GMB': 'Gambia', 'GNB': 'Guinea-Bissau', 'GNQ': 'Equatorial Guinea', 
    'GRC': 'Greece', 'GRD': 'Grenada', 'GRL': 'Greenland', 'GTM': 'Guatemala', 'GUM': 'Guam', 
    'GUY': 'Guyana', 'HKG': 'Hong Kong, SAR China', 'HND': 'Honduras', 'HPC': 'Heavily indebted poor countries', 
    'HRV': 'Croatia', 'HTI': 'Haiti', 'HUN': 'Hungary', 'IBD': 'IBRD only', 'IBT': 'IDA & IBRD total', 
    'IDA': 'IDA total', 'IDB': 'IDA blend', 'IDN': 'Indonesia', 'IDX': 'IDA only', 'IMN': 'Isle of Man', 
    'IND': 'India', 'IRL': 'Ireland', 'IRN': 'Iran, Islamic Republic of', 'IRQ': 'Iraq', 
    'ISL': 'Iceland', 'ISR': 'Israel', 'ITA': 'Italy', 'JAM': 'Jamaica', 'JOR': 'Jordan', 
    'JPN': 'Japan', 'KAZ': 'Kazakhstan', 'KEN': 'Kenya', 'KGZ': 'Kyrgyzstan', 'KHM': 'Cambodia', 
    'KIR': 'Kiribati', 'KNA': 'Saint Kitts and Nevis', 'KOR': 'Korea (South)', 'KWT': 'Kuwait', 
    'LAC': 'Latin America & Caribbean', 'LAO': 'Lao PDR', 'LBN': 'Lebanon', 'LBR': 'Liberia', 
    'LBY': 'Libya', 'LCA': 'Saint Lucia', 'LCN': 'Latin America & Caribbean (excluding high income)', 
    'LDC': 'Least developed countries: UN classification', 'LIE': 'Liechtenstein', 'LKA': 'Sri Lanka', 
    'LMY': 'Low & middle income', 'LSO': 'Lesotho', 'LTE': 'Late-demographic dividend', 'LTU': 'Lithuania', 
    'LUX': 'Luxembourg', 'LVA': 'Latvia', 'MAC': 'Macao, SAR China', 'MAF': 'Saint-Martin (French part)', 
    'MAR': 'Morocco', 'MCO': 'Monaco', 'MDA': 'Moldova', 'MDG': 'Madagascar', 'MDV': 'Maldives', 
    'MEA': 'Middle East & North Africa', 'MEX': 'Mexico', 'MHL': 'Marshall Islands', 'MIC': 'Middle income', 
    'MKD': 'North Macedonia', 'MLI': 'Mali', 'MLT': 'Malta', 'MMR': 'Myanmar', 'MNA': 'Middle East & North Africa (excluding high income)', 
    'MNE': 'Montenegro', 'MNG': 'Mongolia', 'MNP': 'Northern Mariana Islands', 'MOZ': 'Mozambique', 
    'MRT': 'Mauritania', 'MUS': 'Mauritius', 'MWI': 'Malawi', 'MYS': 'Malaysia', 'NAC': 'North America', 
    'NAM': 'Namibia', 'NCL': 'New Caledonia', 'NER': 'Niger', 'NGA': 'Nigeria', 'NIC': 'Nicaragua', 
    'NLD': 'Netherlands', 'NOR': 'Norway', 'NPL': 'Nepal', 'NRU': 'Nauru', 'NZL': 'New Zealand', 
    'OED': 'OECD members', 'OMN': 'Oman', 'OSS': 'Other small states', 'PAK': 'Pakistan', 
    'PAN': 'Panama', 'PER': 'Peru', 'PHL': 'Philippines', 'PLW': 'Palau', 'PNG': 'Papua New Guinea', 
    'POL': 'Poland', 'PRE': 'Pre-demographic dividend', 'PRI': 'Puerto Rico', 'PRK': 'Korea (North)', 
    'PRT': 'Portugal', 'PRY': 'Paraguay', 'PSE': 'Palestinian Territory', 'PSS': 'Pacific island small states', 
    'PST': 'Post-demographic dividend', 'PYF': 'French Polynesia', 'QAT': 'Qatar', 'ROU': 'Romania', 
    'RUS': 'Russian Federation', 'RWA': 'Rwanda', 'SAS': 'South Asia', 'SAU': 'Saudi Arabia', 
    'SDN': 'Sudan', 'SEN': 'Senegal', 'SGP': 'Singapore', 'SLB': 'Solomon Islands', 'SLE': 'Sierra Leone', 
    'SLV': 'El Salvador', 'SMR': 'San Marino', 'SOM': 'Somalia', 'SRB': 'Serbia', 'SSA': 'Sub-Saharan Africa', 
    'SSD': 'South Sudan', 'SSF': 'Sub-Saharan Africa (excluding high income)', 'SST': 'Small states', 
    'STP': 'São Tomé and Principe', 'SUR': 'Suriname', 'SVK': 'Slovak Republic', 'SVN': 'Slovenia', 
    'SWE': 'Sweden', 'SWZ': 'Eswatini', 'SXM': 'Sint Maarten (Dutch part)', 'SYC': 'Seychelles', 
    'SYR': 'Syrian Arab Republic', 'TCA': 'Turks and Caicos Islands', 'TCD': 'Chad', 
    'TEA': 'East Asia & Pacific (IDA & IBRD countries)', 'TEC': 'Europe & Central Asia (IDA & IBRD countries)', 
    'TGO': 'Togo', 'THA': 'Thailand', 'TJK': 'Tajikistan', 'TKM': 'Turkmenistan', 
    'TLA': 'Latin America & Caribbean (IDA & IBRD countries)', 'TLS': 'Timor-Leste', 
    'TMN': 'Middle East & North Africa (IDA & IBRD countries)', 'TON': 'Tonga', 
    'TSA': 'South Asia (IDA & IBRD)', 'TSS': 'Sub-Saharan Africa (IDA & IBRD countries)', 
    'TTO': 'Trinidad and Tobago', 'TUN': 'Tunisia', 'TUR': 'Turkey', 'TUV': 'Tuvalu', 
    'TZA': 'Tanzania', 'UGA': 'Uganda', 'UKR': 'Ukraine', 'URY': 'Uruguay', 'USA': 'United States', 
    'UZB': 'Uzbekistan', 
    'VCT': 'Saint Vincent and the Grenadines',
    'VEN': 'Venezuela, Bolivarian Republic of',
    'VGB': 'Virgin Islands (British)',
    'VIR': 'Virgin Islands (U.S.)',
    'VNM': 'Viet Nam',
    'VUT': 'Vanuatu',
    'WLF': 'Wallis and Futuna',
    'WSM': 'Samoa',
    'YEM': 'Yemen',
    'ZAF': 'South Africa',
    'ZMB': 'Zambia',
    'ZWE': 'Zimbabwe',
    'WLD': 'World',
    'XKX': 'Kosovo',   
}


#%%
# Mapping country codes to full names
unmapped_codes = []

def safe_map(code):
    if code in country_code_to_name:
        return country_code_to_name[code]
    else:
        unmapped_codes.append(code)
        return code  

df['country'] = df['country'].apply(safe_map)

#%%
df['country'].unique()

# %%
df['country'].isna().sum()
# %%
unique_unmapped_codes = list(set(unmapped_codes))
# %%
print(unique_unmapped_codes)
# %%
df.isna().sum()

# %%
# Dropping rows with unmapped country codes
df = df[~df['country'].isin(unique_unmapped_codes)]
df.isna().sum()

# %%
# Dropping columns with too many missing values
null_counts = df.isnull().sum()

columns_with_low_nulls = null_counts[null_counts < 5000].index.tolist()

df = df[columns_with_low_nulls]
# %%
df.isna().sum()

# %%
# Dropping rows with years from 1960 to 1990
df = df[~df['year'].between(1960, 1990)]

# %%

# Fill GDP (current US$) nulls with country-specific mean
df['GDP (current US$)'] = df['GDP (current US$)'].fillna(
    df.groupby('country')['GDP (current US$)'].transform('mean')
)

# Fill GDP per capita (current US$) nulls with country-specific mean
df['GDP per capita (current US$)'] = df['GDP per capita (current US$)'].fillna(
    df.groupby('country')['GDP per capita (current US$)'].transform('mean')
)

df['Life expectancy at birth, male (years)'] = df['Life expectancy at birth, male (years)'].fillna(
    df.groupby('country')['Life expectancy at birth, male (years)'].transform('mean')
)

df['Life expectancy at birth, female (years)'] = df['Life expectancy at birth, female (years)'].fillna(
    df.groupby('country')['Life expectancy at birth, female (years)'].transform('mean')
)

df['Life expectancy at birth, total (years)'] = df['Life expectancy at birth, total (years)'].fillna(
    df.groupby('country')['Life expectancy at birth, total (years)'].transform('mean')
)

df['Rural population'] = df['Rural population'].fillna(
    df.groupby('country')['Rural population'].transform('mean')
)

df['Rural population (% of total population)'] = df['Rural population (% of total population)'].fillna(
    df.groupby('country')['Rural population (% of total population)'].transform('mean')
)

df['Death rate, crude (per 1,000 people)'] = df['Death rate, crude (per 1,000 people)'].fillna(
    df.groupby('country')['Death rate, crude (per 1,000 people)'].transform('mean')
)
# %%
df.isna().sum()
# %%
missing_rp = df[df['Rural population (% of total population)'].isna() | df['Rural population'].isna()]
missing_rp['country'].unique()
# %%
missing_le = df[df['Life expectancy at birth, female (years)'].isna() | df['Life expectancy at birth, male (years)'].isna()]
missing_gdp = df[df['GDP (current US$)'].isnull() | df['GDP per capita (current US$)'].isnull()]
missing_gdp['country'].unique()
missing_le['country'].unique()

# %%
missing_population = df[df['Population, total'].isna()]
missing_population['country'].unique()


# %%
countries_to_remove = ['Gibraltar', 'Korea (North)', 'Virgin Islands (British)', 'Andorra', 'American Samoa', 'Monaco', 'Northern Mariana Islands',
       'San Marino', 'Saint-Martin (French part)', 'Kosovo']
df = df[~df['country'].isin(countries_to_remove)]
df
# %%
df.isna().sum()

# %%
df = df.reset_index(drop=True)
df.to_csv("cleaned_data.csv", index=False)

# %%
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
# Bivariate Analysis — Correlation Heatmap
numeric_df = df.select_dtypes(include=['number'])  # only numeric columns

corr_matrix = numeric_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap of Numeric Variables")
plt.tight_layout()
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

"""
File: main_final_model.py
Purpose: Run full pipeline: cleaning, EDA, and modeling.
"""

#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler

# %%
df = pd.read_csv('cleaned_data.csv')
df

# %%
df.info()

# %%
df.describe()

# %%
# Check missing values
plt.figure(figsize=(10, 4))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.tight_layout()
plt.show()

# %%
numeric_df = df.select_dtypes(include=[np.number])

correlation_matrix = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# =========================
# New EDA Plots
# =========================

# %%
# Distribution of life expectancy
plt.figure(figsize=(8, 5))
sns.histplot(df['Life expectancy at birth, total (years)'], kde=True, bins=30, color='skyblue')
plt.title("Distribution of Life Expectancy")
plt.xlabel("Life Expectancy at Birth (Years)")
plt.tight_layout()
plt.show()

# %%
# Top 10 countries with highest average life expectancy
top_life = df.groupby('country')['Life expectancy at birth, total (years)'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_life.values, y=top_life.index, palette="crest")
plt.title("Top 10 Countries by Average Life Expectancy")
plt.xlabel("Average Life Expectancy")
plt.ylabel("Country")
plt.tight_layout()
plt.show()

# %%
# GDP per capita vs Life Expectancy
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='GDP per capita (current US$)', y='Life expectancy at birth, total (years)', alpha=0.6)
plt.title("GDP per Capita vs Life Expectancy")
plt.xscale("log")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Pairplot for key variables
key_vars = ['GDP per capita (current US$)', 'Death rate, crude (per 1,000 people)',
            'Population ages 65 and above (% of total population)',
            'Life expectancy at birth, total (years)']

sns.pairplot(df[key_vars], kind='scatter', diag_kind='kde')
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.tight_layout()
plt.show()

# =========================
# Preprocessing Before Modeling
# =========================

# %%
df = df.drop(columns=["Life expectancy at birth, female (years)", "Life expectancy at birth, male (years)"])

#%%
label_encoder = LabelEncoder()
df['country'] = label_encoder.fit_transform(df['country'])
country_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Country Encoding Mapping:", country_mapping)
df.head()

# %%
country_mapping_df = pd.DataFrame(list(country_mapping.items()), columns=['Country', 'Encoded Value'])
print(country_mapping_df)

# %%
df.info()

# %%
df = df.sort_values(by='year')
df.head()

# %%
train_df = df[df['year'] <= 2022]
test_df = df[df['year'] > 2022]

X_train = train_df.drop(columns=['Life expectancy at birth, total (years)'])
y_train = train_df['Life expectancy at birth, total (years)']

X_test = test_df.drop(columns=['Life expectancy at birth, total (years)'])
y_test = test_df['Life expectancy at birth, total (years)']

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
# %%
rf_model = RandomForestRegressor(random_state=42)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

mse_rfr = mean_squared_error(y_test, y_pred)
r2_rfr = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse_rfr)
print("R-squared (R2):", r2_rfr)
# %%
dt_model = DecisionTreeRegressor(random_state=42)

dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print("Decision Tree Regressor - Mean Squared Error (MSE):", mse_dt)
print("Decision Tree Regressor - R-squared (R2):", r2_dt)
# %%
br_model = BayesianRidge()

br_model.fit(X_train, y_train)

y_pred_br = br_model.predict(X_test)

mse_br = mean_squared_error(y_test, y_pred_br)
r2_br = r2_score(y_test, y_pred_br)

print("Bayesian Ridge Regressor - Mean Squared Error (MSE):", mse_br)
print("Bayesian Ridge Regressor - R-squared (R2):", r2_br)

#%%
from sklearn.neighbors import KNeighborsRegressor

knn_model = KNeighborsRegressor(n_neighbors=7)

knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)

# Evaluate
mse_knn = mean_squared_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)

# Display the results in the same format
print("KNN Regressor - Mean Squared Error (MSE):", mse_knn)
print("KNN Regressor - R-squared (R2):", r2_knn)

# %%
# Import missing numpy module due to environment reset
import numpy as np
import matplotlib.pyplot as plt

# Prepare the existing performance results including KNN
model_names = ["Random Forest", "Decision Tree", "Bayesian Ridge", "KNN Regressor"]
rmse_values = [0.862, 1.326, 1.853, np.sqrt(0.551)]  # KNN RMSE from MSE
r2_values = [0.991, 0.980, 0.961, 0.994]

# Plotting RMSE and R² comparison
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# RMSE plot
axs[0].bar(model_names, rmse_values, color='cornflowerblue')
axs[0].set_title("RMSE of Regression Models")
axs[0].set_ylabel("Root Mean Squared Error")
axs[0].set_ylim(0, max(rmse_values) + 0.5)

# R² plot
axs[1].bar(model_names, r2_values, color='mediumseagreen')
axs[1].set_title("R² of Regression Models")
axs[1].set_ylabel("R² Score")
axs[1].set_ylim(0.9, 1.01)

plt.tight_layout()
plt.show()

# %%
# Feature importance from Random Forest Regressor
feature_importances = rf_model.feature_importances_

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Display the feature importance
print(importance_df)

# Plot the feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette="viridis")

plt.title("Feature Importance from Random Forest Regressor", fontsize=16)
plt.xlabel("Importance", fontsize=12)
plt.ylabel("Feature", fontsize=12)

plt.tight_layout()
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet

# ========= Preprocessing =========
# Group by year to create a time series
ts_df = df.groupby('year')['Life expectancy at birth, total (years)'].mean().reset_index()
ts_df.columns = ['year', 'life_expectancy']
ts_df.set_index('year', inplace=True)

# ========= Accuracy Function =========
def evaluate_model(true, pred, model_name="Model"):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    mape = np.mean(np.abs((true - pred) / true)) * 100
    r2 = r2_score(true, pred)
    print(f"\n--- {model_name} Accuracy ---")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²  : {r2:.4f}")

# ========= ARIMA =========
print("\n--- ARIMA ---")
arima_model = ARIMA(ts_df, order=(1, 1, 1))
arima_result = arima_model.fit()
arima_pred_in = arima_result.predict(start=1, end=len(ts_df)-1, typ="levels")
evaluate_model(ts_df.iloc[1:, 0], arima_pred_in, "ARIMA")

# Forecast next 5 years
arima_forecast = arima_result.forecast(steps=5)
arima_index = range(ts_df.index[-1] + 1, ts_df.index[-1] + 6)

plt.figure(figsize=(10, 5))
plt.plot(ts_df, label='Observed')
plt.plot(arima_index, arima_forecast, label='ARIMA Forecast', marker='o')
plt.title("ARIMA Forecast")
plt.xlabel("Year")
plt.ylabel("Life Expectancy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ========= SARIMAX =========
print("\n--- SARIMAX ---")
sarimax_model = SARIMAX(ts_df, order=(1, 1, 1), seasonal_order=(0, 1, 1, 4))
sarimax_result = sarimax_model.fit(disp=False)
sarimax_pred_in = sarimax_result.predict(start=1, end=len(ts_df)-1, typ="levels")
evaluate_model(ts_df.iloc[1:, 0], sarimax_pred_in, "SARIMAX")

sarimax_forecast = sarimax_result.forecast(steps=5)

plt.figure(figsize=(10, 5))
plt.plot(ts_df, label='Observed')
plt.plot(arima_index, sarimax_forecast, label='SARIMAX Forecast', marker='x')
plt.title("SARIMAX Forecast")
plt.xlabel("Year")
plt.ylabel("Life Expectancy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ========= Prophet =========
print("\n--- Prophet ---")
df_prophet = ts_df.reset_index()
df_prophet.columns = ['ds', 'y']
prophet_model = Prophet()
prophet_model.fit(df_prophet)

# In-sample prediction for accuracy
future_all = prophet_model.make_future_dataframe(periods=0, freq='Y')
forecast_all = prophet_model.predict(future_all)
evaluate_model(df_prophet['y'], forecast_all['yhat'], "Prophet")

# Forecast 5 years ahead
future_5 = prophet_model.make_future_dataframe(periods=5, freq='Y')
forecast_5 = prophet_model.predict(future_5)

# Plot
fig = prophet_model.plot(forecast_5)
plt.title("Prophet Forecast")
plt.xlabel("Year")
plt.ylabel("Life Expectancy")
plt.grid(True)
plt.tight_layout()
plt.show()

# ========= ETS =========
print("\n--- ETS (Exponential Smoothing) ---")
ets_model = ExponentialSmoothing(ts_df, trend='add', seasonal=None)
ets_result = ets_model.fit()
ets_pred_in = ets_result.fittedvalues
evaluate_model(ts_df['life_expectancy'], ets_pred_in, "ETS")

ets_forecast = ets_result.forecast(steps=5)

plt.figure(figsize=(10, 5))
plt.plot(ts_df, label='Observed')
plt.plot(arima_index, ets_forecast, label='ETS Forecast', marker='s')
plt.title("ETS Forecast")
plt.xlabel("Year")
plt.ylabel("Life Expectancy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Among the four time series models evaluated, ETS (Exponential Smoothing) and ARIMA delivered the most accurate results. 
# ETS had the lowest MAE (0.2187) and RMSE (0.5428), along with the highest R² (0.9541), indicating an excellent fit to the data. 
# ARIMA also performed well, with a low MAPE (0.63%) and a strong R² of 0.9318, making it a reliable forecasting model. 
# On the other hand, SARIMAX and Prophet underperformed in this context, with SARIMAX showing unusually high error values and a negative R², suggesting poor model fit. 
# Prophet’s accuracy was even lower, with the highest MAPE (13.41%) and an R² of -13.52, indicating that it failed to capture the underlying trend effectively. 
# Based on these metrics, ETS emerges as the best-performing model, followed closely by ARIMA, for forecasting global life expectancy in this dataset.
#%%
