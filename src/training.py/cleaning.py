# data_cleaning.py

import pandas as pd
import sqlite3
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_customer_data(sqlite_db_path: str, output_csv_path: str = None) -> pd.DataFrame:
    start_time = time.time()
    logging.info("Starting data cleaning process...")

    # Load data from the summary table
    conn = sqlite3.connect(sqlite_db_path)
    df = pd.read_sql("SELECT * FROM customer_summary;", conn)
    conn.close()
    logging.info(f"Loaded {len(df)} records from 'customer_summary'")

    # --- Cleaning ---
    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Handle missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        logging.warning(f"Missing values detected in: {missing_cols}")
        df.dropna(inplace=True)
        logging.info("Dropped rows with missing values.")

    # Encode ordinal IncomeLevel
    if 'IncomeLevel' in df.columns:
        income_map = {'Low': 0, 'Medium': 2, 'High': 3}
        df['IncomeLevelEncoded'] = df['IncomeLevel'].map(income_map)
        logging.info("Encoded 'IncomeLevel' into 'IncomeLevelEncoded'.")

    # Drop unneeded columns
    drop_cols = ['CustomerID', 'LastLoginDate', 'Churn', 'AgeGroup', 'IncomeLevel']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')
    logging.info("Dropped unnecessary columns.")

    # Save if output path given
    if output_csv_path:
        df.to_csv(output_csv_path, index=False)
        logging.info(f"Cleaned data saved to {output_csv_path}")

    elapsed = time.time() - start_time
    logging.info(f"Data cleaning completed in {elapsed:.2f} seconds.")
    
    return df


if __name__ == "__main__":
    # Example usage
    cleaned_df = clean_customer_data("lloyds_churn.db", "cleaned_customer_data.csv")
