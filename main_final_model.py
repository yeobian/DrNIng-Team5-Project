"""
File: main_final_model.py
Purpose: Run full pipeline: cleaning, EDA, and modeling.
"""

from data_cleaning import clean_data
from eda_visuals import run_eda
from modeling_mlr import run_mlr
from modeling_tree_rf import run_tree_rf

def main():
    df_clean = clean_data()
    run_eda(df_clean)
    run_mlr(df_clean)
    run_tree_rf(df_clean)

if __name__ == "__main__":
    main()
