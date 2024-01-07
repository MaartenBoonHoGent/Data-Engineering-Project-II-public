from typing import List, Dict
from sqlalchemy.engine import Engine
import os
from sqlalchemy import create_engine, text
import json
import dotenv

dotenv.load_dotenv()
JSON_PATH = os.getenv("IMPORT_DATA_PATH")


def enable_foreign_keys(conn, tables: List[str]) -> None:
    connection = conn.connect()
    for table in tables:
        connection.execute(text(f"ALTER TABLE {table} WITH CHECK CHECK CONSTRAINT ALL"))
        connection.commit()
    connection.close()


def create_conn():
    driver = os.getenv("DB_DRIVER")
    server = os.getenv("DB_SERVER")
    database = os.getenv("DB_NAME")
    trusted_connection = os.getenv("DB_TRUSTED_CONNECTION")

    return create_engine(
        f"mssql+pyodbc://{server}/{database}?trusted_connection={trusted_connection}&driver={driver}"
    )


with open(JSON_PATH, "r") as f:
    import_data: List[Dict] = json.load(f)


conn = create_conn()
table_names = [table["table_name"] for table in import_data]
enable_foreign_keys(conn, table_names)
