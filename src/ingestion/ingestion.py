import pandas as pd
import logging
import time
from sqlalchemy import create_engine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def excel_to_sqlite(excel_path, sqlite_db_path):
    start_time = time.time()
    logging.info(f"Starting to process Excel file: {excel_path}")

    # Create SQLite engine (file created if not exists)
    engine = create_engine(f'sqlite:///{sqlite_db_path}')

    # Read all sheets
    try:
        xls = pd.ExcelFile(excel_path)
        sheet_names = xls.sheet_names
        logging.info(f"Found sheets: {sheet_names}")
    except Exception as e:
        logging.error(f"Failed to read Excel file: {e}")
        return

    for sheet in sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet)
            table_name = sheet.lower().replace(' ', '_')
            logging.info(f"Processing sheet '{sheet}' -> table '{table_name}' with {len(df)} rows")

            # Write to SQLite
            with engine.begin() as conn:
                df.to_sql(table_name, conn, if_exists='replace', index=False)

            logging.info(f"Successfully saved '{table_name}' to SQLite")
        except Exception as e:
            logging.error(f"Failed to process sheet '{sheet}': {e}")

    elapsed = time.time() - start_time
    logging.info(f"Finished processing in {elapsed:.2f} seconds")

if __name__ == "__main__":
    excel_file_path = '../data/raw/Customer_Churn_Data_Large (1).xlsx'  # Change to your Excel file path
    sqlite_db_file = 'lloyds_churn.db'       # SQLite database filename

    excel_to_sqlite(excel_file_path, sqlite_db_file)