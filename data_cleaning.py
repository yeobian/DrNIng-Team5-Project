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
len(df.columns)
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
