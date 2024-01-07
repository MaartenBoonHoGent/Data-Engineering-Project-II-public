import dotenv
import os
import json
from sqlalchemy import create_engine, text
import time

dotenv.load_dotenv()

JSON_PATH = os.getenv("IMPORT_FOREIGN_KEYS_PATH")

assert os.path.exists(JSON_PATH), f"File '{JSON_PATH}' does not exist."

with open(JSON_PATH, "r") as f:
    fk_data = json.load(f)


def create_conn():
    driver = os.getenv("DB_DRIVER")
    server = os.getenv("DB_SERVER")
    database = os.getenv("DB_NAME")
    trusted_connection = os.getenv("DB_TRUSTED_CONNECTION")

    return create_engine(
        f"mssql+pyodbc://{server}/{database}?trusted_connection={trusted_connection}&driver={driver}"
    )


def remove_orphaned_keys(connection):
    conn = connection.connect()
    # Go over each record
    for record in fk_data:
        primary_key = record["primary_key"]
        table = record["table"]
        foreign_keys = record["foreign_keys"]

        for foreign_key_data in foreign_keys:
            foreign_key = foreign_key_data["foreign_key"]
            referenced_table = foreign_key_data["table"]

            # Check if the foreign key exists in the referenced table
            query = text(
                f"SELECT COUNT(*) FROM {referenced_table} WHERE {foreign_key} NOT IN (SELECT {primary_key} FROM {table})"
            )
            result = conn.execute(query).fetchone()

            if result and result[0] > 0:
                # There are orphaned foreign keys, you can remove them here.
                # For example, you can use the DELETE statement:
                delete_query = text(
                    f"DELETE FROM {referenced_table} WHERE {foreign_key} NOT IN (SELECT {primary_key} FROM {table})"
                )
                conn.execute(delete_query)
                conn.commit()
                print(
                    f"Removed orphaned foreign keys from {referenced_table} where {foreign_key} not in {table}"
                )
                time.sleep(1)
            else:
                print(f"No orphaned foreign keys found in {referenced_table}")

    conn.close()


if __name__ == "__main__":
    conn = create_conn()
    remove_orphaned_keys(conn)
