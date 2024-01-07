import dotenv
import os
import json
from typing import Dict, List
import pandas as pd
import datetime
from sqlalchemy import create_engine, text

dotenv.load_dotenv()

UTC_OFFSET = int(os.getenv("UTC_OFFSET"))


def main():
    # Remove the 'file_name' key from the db_info.json file
    db_info = load_info()
    # Load the tables_files
    tables_files = load_tables()
    table_names = [table["table_name"] for table in db_info]
    # Disable the foreign key checks
    log("Disabling foreign keys")
    disable_foreign_keys(create_conn(), table_names)

    log(f"Process will import {len(tables_files)} file(s)")

    for value in db_info:
        value["table_name"] = str(value["table_name"]).encode("latin1").decode("utf-8")

    tables_files = {
        k: v
        for k, v in sorted(
            tables_files.items(),
            key=lambda item: int(get_db_info(item[1], db_info)["order"]),
        )
    }

    # Loop through the tables_files
    counter = 0
    max_counter = len(tables_files)
    for file_name, table_name in tables_files.items():
        # Get the db_info for the table
        table_info = get_db_info(table_name, db_info)

        # Use all of this to either add or update the table
        counter += 1
        log(f"({counter}/{max_counter})")
        add_or_update_table(file_name, table_name, table_info)

    log("Enabling foreign keys")
    enable_foreign_keys(create_conn(), table_names)


def log(message: str, table_name: str = None):
    if table_name is not None:
        message = f"{table_name}: {message}"
    print(f"[{datetime.datetime.now()}] {message}")


def add_or_update_table(file_name: str, table_name: str, table_info: Dict):
    log(f"Processing {table_name}", table_name=table_name)
    # Get the data from the csv file
    data = get_new_data(file_name, table_info)

    # Change the data to match the table_info
    log("Renaming columns", table_name=table_name)
    data = rename_columns(data, table_info["columns"])
    # Remove any duplicate primary keys
    log("Removing duplicates", table_name=table_name)
    data = remove_duplicates(data, table_info["primary_key"])
    # Perform any refactors on the data
    log("Refactoring data", table_name=table_name)
    data = refactor_data(data, table_info["refactor"])
    # Create a connection to the database
    conn = create_conn()
    # Remove any primary keys that are already in the database
    log("Removing existing primary keys in the database", table_name=table_name)
    remove_existing_primary_keys(data, table_name, table_info["primary_key"], conn)
    # Load the data into the database
    check_foreign_keys(data, table_name, table_info["foreign_keys"], conn)
    log("Loading data into the database", table_name=table_name)
    write_to_db(data, table_name, conn)

    log("Deleting complete duplicates", table_name=table_name)
    delete_complete_duplicates(table_name, data.columns.to_list(), conn)


def refactor_data(data: pd.DataFrame, refactors: Dict[str, Dict]) -> pd.DataFrame:
    if refactors is None or len(refactors) == 0:
        return data

    for column_name, functions in refactors.items():
        # Check if the functions is a list
        if not isinstance(functions, list):
            functions = [functions]
        # Check if the function exists

        for functionName in functions:
            data[column_name] = data[column_name].apply(
                lambda x: eval(functionName)(x) if x is not None else None
            )

    return data


def remove_duplicates(data: pd.DataFrame, primary_key: str) -> pd.DataFrame:
    data.drop_duplicates(inplace=True)
    if primary_key is None:
        return data

    data.dropna(subset=[primary_key], inplace=True)
    data.drop_duplicates(subset=primary_key, inplace=True)
    return data


def rename_columns(data: pd.DataFrame, columns: Dict[str, str]) -> pd.DataFrame:
    # bring all the columns to lower case
    data.columns = [column.lower() for column in data.columns]
    # Bring the first value of each column to lower case as well
    columns = {key.lower(): value for key, value in columns.items()}
    # Rename the columns
    data.rename(columns=columns, inplace=True)
    return data


# ----------------- #
# db functions      #
# ----------------- #


def remove_existing_primary_keys(
    data: pd.DataFrame, table_name: str, primary_key: str, conn
):
    if table_is_empty(table_name, conn) or primary_key is None:
        return

    connection = conn.connect()
    table_size = len(data.index)
    chunk_size = int(len(data.index) / 10)
    chunk_size = max(
        min(chunk_size, 20_000), 5_000
    )  # Make sure the chunk size is at least 20_000

    for j in range(0, table_size, chunk_size):
        # Get the existing primary keys

        chunk = data[j : j + chunk_size]
        existing_primary_keys_query = f"""
        DELETE FROM {table_name}
        WHERE {primary_key} IN ( '{"', '".join([str(c) for c in chunk[primary_key]])}' )
        """

        print(
            f"Removing existing primary keys. ({j + len(chunk.index)}/{table_size}), ({(j + len(chunk.index)) / table_size * 100}%)",
            end="\r",
        )

        connection.execute(text(existing_primary_keys_query))
        connection.commit()
    connection.close()


def delete_complete_duplicates(table_name: str, columns: List, conn):
    """
    Deletes all the rows where each column is the same directly on the SQL server
    """

    # Execute SQL query to remove duplicates in the database
    delete_query = f"""
        WITH CTE AS (
            SELECT 
                *, 
                ROW_NUMBER() OVER (PARTITION BY {', '.join(columns)} ORDER BY (SELECT NULL)) AS RowNum
            FROM 
                {table_name}
        )
        DELETE FROM CTE WHERE RowNum > 1;
    """

    # Execute the delete query
    connection = conn.connect()
    connection.execute(text(delete_query))
    connection.commit()
    connection.close()


def table_is_empty(table_name: str, conn) -> bool:
    query = f"""
    SELECT COUNT(*) AS count
    FROM {table_name}
    """

    return pd.read_sql(query, conn).iloc[0]["count"] == 0


def create_conn():
    driver = os.getenv("DB_DRIVER")
    server = os.getenv("DB_SERVER")
    database = os.getenv("DB_NAME")
    trusted_connection = os.getenv("DB_TRUSTED_CONNECTION")

    return create_engine(
        f"mssql+pyodbc://{server}/{database}?trusted_connection={trusted_connection}&driver={driver}"
    )


def disable_foreign_keys(conn, tables: List[str]) -> None:
    connection = conn.connect()
    for table in tables:
        connection.execute(text(f"ALTER TABLE {table} NOCHECK CONSTRAINT ALL"))
        connection.commit()
    connection.close()


def enable_foreign_keys(conn, tables: List[str]) -> None:
    connection = conn.connect()
    for table in tables:
        connection.execute(text(f"ALTER TABLE {table} WITH CHECK CHECK CONSTRAINT ALL"))
        connection.commit()
    connection.close()


def write_to_db(data: pd.DataFrame, table_name: str, conn):
    table_size = len(data.index)
    chunk_size = int(len(data.index) / 10)
    chunk_size = max(
        min(chunk_size, 100_000), 10_000
    )  # Make sure the chunk size is at least 100_000
    for j in range(0, table_size, chunk_size):
        chunk = data[j : j + chunk_size]
        chunk.to_sql(table_name, conn, if_exists="append", index=False)
        print(
            f"Data written to database. ({j + len(chunk.index)}/{table_size}) ({(j + len(chunk.index)) / table_size * 100}%)",
            end="\r",
        )
        del chunk
    log("Data written to database.", table_name=table_name)


def check_foreign_keys(
    data: pd.DataFrame, table_name: str, foreign_keys: List, conn
) -> pd.DataFrame:
    if foreign_keys is None or len(foreign_keys) == 0:
        return data

    connection = conn.connect()
    for foreign_key in foreign_keys:
        column_name = foreign_key["foreign_key"]
        referenced_table = foreign_key["table"]
        primary_key = foreign_key["primary_key"]

        # Get the referenced primary keys
        referenced_primary_keys = pd.read_sql(
            f"""
            SELECT {primary_key}
            FROM {referenced_table}
            """,
            connection,
        )

        # Get the primary keys that are not in the referenced table
        missing_primary_keys = data[
            ~data[column_name].isin(referenced_primary_keys[primary_key])
        ]

        # Remove the rows that are not in the referenced table
        data.drop(
            data[~data[column_name].isin(referenced_primary_keys[primary_key])].index,
            inplace=True,
        )

        # Log the missing primary keys
        log(
            f"Removed {len(missing_primary_keys.index)} rows",
            table_name=table_name,
        )

    connection.close()
    return data


# ----------------- #
# Loading functions #
# ----------------- #
def get_new_data(file_name: str, table_info: Dict) -> pd.DataFrame:
    # Get the data from the csv file

    path = (
        os.path.join(os.getenv("IMPORT_CSV_PATH"), file_name)
        .encode("latin1")
        .decode("utf-8")
    )

    csv_info = get_csv_info(table_info)
    print(path)
    assert os.path.exists(path), f"File '{path}' does not exist."
    return pd.read_csv(
        path,
        sep=csv_info["delimiter"],
        encoding=csv_info["encoding"],
    )


def get_db_info(table_name: str, db_info: Dict) -> Dict:
    for table in db_info:
        if table["table_name"] == table_name:
            return table

    raise ValueError(f"Table {table_name} not found in db_info")


def load_tables() -> Dict[str, str]:
    return json.load(
        open(
            os.path.join(
                os.getenv("EPIC_2_SAVE_LOCATION"), os.getenv("TABLES_FILE_NAME")
            )
        )
    )


def load_info():
    return json.load(
        open(
            os.path.join(
                os.getenv("EPIC_2_SAVE_LOCATION"), os.getenv("DB_INFO_FILE_NAME")
            )
        )
    )


# ----------------- #
# Helper functions  #
# ----------------- #


def get_csv_info(table_info) -> Dict:
    # Set default return value
    return_value = {
        "encoding": "utf-8",
        "delimiter": ",",
    }
    # Check if the table has a custom encoding
    if "info" not in table_info:
        return return_value
    # Check if the encoding is set
    for key in return_value.keys():
        if key in table_info["info"]:
            return_value[key] = table_info["info"][key]
    return return_value


def to_date(string) -> datetime.date or None:
    # Make sure the string is formatted correctly
    # Different way of dealing with unformatted dates -- Priority: Low
    string = str(string).strip()
    if string is None or len(string.split("-")) != 3:
        return None
    return datetime.datetime.strptime(string, "%d-%m-%Y").date()


def to_float(string) -> float or None:
    string = str(string).strip()
    string = string.replace(",", ".")
    try:
        return float(string)
    except ValueError:
        return None


def to_datetime(string) -> datetime.datetime or None:
    string = str(string).strip()

    if string is None or len(string.split("-")) != 3:
        return None
    return datetime.datetime.strptime(string, "%d-%m-%Y %H:%M:%S")


def to_datetime_utc(string) -> datetime.datetime or None:
    string = str(string).lower()
    # Remove optional (UTC) from the string
    assert "(utc)" in string, f"String '{string}' might not be utc."
    string = string.replace("(utc)", "")
    string = str(string).strip()

    if string is None or len(string.split("-")) != 3:
        return None

    utc_datetime = datetime.datetime.strptime(string, "%m-%d-%Y %H:%M:%S")
    # Local datetime is Brussels time
    local_datetime = utc_datetime + datetime.timedelta(hours=UTC_OFFSET)
    return local_datetime


def ja_nee(string: str) -> int:
    assert string.lower() in ["ja", "nee"], f"String '{string}' is not 'ja' or 'nee'."
    return 1 if string.lower() == "ja" else (0 if string.lower() == "nee" else None)


def clean_nan(string) -> str or None:
    string = str(string).strip()
    if string is None:
        return None
    if string.lower() == "nan":
        return None
    return string.strip()


if __name__ == "__main__":
    main()
