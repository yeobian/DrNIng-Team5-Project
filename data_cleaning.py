"""
File: data_cleaning.py
Purpose: Clean life expectancy dataset and prepare for merging with socio-economic factors.
"""

import pandas as pd

# Load dataset
df = pd.read_csv('1- life-expectancy.csv')

# Clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(' ', '_')
    .str.replace(r'[^\w_]', '', regex=True)
)

# Rename for clarity
df = df.rename(columns={
    'entity': 'country',
    'period_life_expectancy_at_birth__sex_all__age_0': 'life_expectancy'
})

# Drop unnecessary column
df = df.drop(columns=['code'])  # code has missing values and is redundant

# Filter year range if needed
df = df[(df['year'] >= 2000) & (df['year'] <= 2015)]

# Final column order
df = df[['country', 'year', 'life_expectancy']]

# Sort for readability
df = df.sort_values(by=['country', 'year'])

# Save cleaned file
df.to_csv('cleaned_life_expectancy.csv', index=False)

# Summary
print("Cleaned file saved: cleaned_life_expectancy.csv")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
