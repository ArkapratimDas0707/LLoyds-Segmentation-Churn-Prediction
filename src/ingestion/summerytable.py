import pandas as pd
import logging
import time
import sqlite3


sqlite_db_file = 'lloyds_churn.db'  # <-- Ensure this is defined globally

def create_summary_table(sqlite_db_file):
    conn = sqlite3.connect(sqlite_db_file)

    summary_query = """
    CREATE TABLE IF NOT EXISTS customer_summary AS
    SELECT
        cd.CustomerID,
        cd.Age,
        cs.ChurnStatus,
        SUM(th.AmountSpent) AS TotalSpent
    FROM customer_demographics cd
    LEFT JOIN churn_status cs ON cd.CustomerID = cs.CustomerID
    LEFT JOIN transaction_history th ON cd.CustomerID = th.CustomerID
    GROUP BY cd.CustomerID, cd.Age, cs.ChurnStatus;
    """

    conn.execute("DROP TABLE IF EXISTS customer_summary;")  # Optional: Reset table
    conn.execute(summary_query)
    conn.commit()
    conn.close()
    logging.info("Created 'customer_summary' table successfully.")

# Call it
create_summary_table(sqlite_db_file)