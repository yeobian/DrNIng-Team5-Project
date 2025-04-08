"""
File: data_cleaning.py
Purpose: Load and clean the dataset.
"""

import pandas as pd

def clean_data():
    # Load dataset
    df = pd.read_csv("life_expectancy.csv")  # Update path if needed
    
    # Basic cleaning
    df = df.dropna()  # or use fillna() where appropriate
    # Convert data types, if necessary
    # Normalize/scale features if needed
    
    # Return cleaned DataFrame
    return df
