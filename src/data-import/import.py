import datetime
import dotenv
import json
import os
from typing import Dict, List
import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path
from remove_orphaned_keys import remove_orphaned_keys

global UTC_OFFSET
global MODE


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


def empty_table(table_name: str, conn) -> None:
    connection = conn.connect()

    log(f"Emptying table '{table_name}'...", overrideable=True, table_name=table_name)
    connection.execute(text(f"DELETE FROM {table_name}"))
    connection.commit()
    log(f"Table '{table_name}' emptied.", overrideable=True, table_name=table_name)
    # Close the connection
    connection.close()


def log(message, overrideable=False, table_name=None):
    print(
        f"""[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]{(' [' + table_name + ']') if
        table_name is not None else ''} {message} {' ' * 25}""",
        end="\r" if overrideable else "\n",
    )


def to_date(string) -> datetime.date or None:
    # Make sure the string is formatted correctly
    # TODO: Different way of dealing with unformatted dates -- Priority: Low
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


def clean_nan(string) -> str or None:
    string = str(string).strip()
    if string is None:
        return None
    if string.lower() == "nan":
        return None
    return string.strip()


def clean_df(df: pd.DataFrame, refactor: Dict) -> pd.DataFrame:
    # Remove full empty rows
    df = df.dropna(how="all")
    # Implement a system for refactoring the data
    for column_name, functions in refactor.items():
        # Check if the functions is a list
        if not isinstance(functions, list):
            functions = [functions]
        # Check if the function exists

        for functionName in functions:
            df[column_name] = df[column_name].apply(
                lambda x: eval(functionName)(x) if x is not None else None
            )

    # Make sure all the empty values (NaN, None, etc.) are None
    df = df.where(pd.notnull(df), None)
    return df


def create_conn():
    driver = os.getenv("DB_DRIVER")
    server = os.getenv("DB_SERVER")
    database = os.getenv("DB_NAME")
    trusted_connection = os.getenv("DB_TRUSTED_CONNECTION")

    return create_engine(
        f"mssql+pyodbc://{server}/{database}?trusted_connection={trusted_connection}&driver={driver}"
    )


def ja_nee(string: str) -> int:
    assert string.lower() in ["ja", "nee"], f"String '{string}' is not 'ja' or 'nee'."
    return 1 if string.lower() == "ja" else (0 if string.lower() == "nee" else None)


def get_info(table) -> Dict:
    # Set default return value
    return_value = {
        "encoding": "utf-8",
        "delimiter": ",",
    }
    # Check if the table has a custom encoding
    if "info" not in table:
        return return_value
    # Check if the encoding is set
    for key in return_value.keys():
        if key in table["info"]:
            return_value[key] = table["info"][key]
    return return_value


def main():
    log("Starting import...")
    # Start .env file
    dotenv.load_dotenv(dotenv_path=Path(".", ".env"))

    # Load the importData as a json file
    JSON_PATH = os.getenv("IMPORT_DATA_PATH")
    CSV_PATH = os.getenv("IMPORT_CSV_PATH")
    # Set the global UTC_OFFSET
    global UTC_OFFSET
    UTC_OFFSET = int(os.getenv("UTC_OFFSET"))
    # Set the global MODE
    global MODE
    MODE = int(os.getenv("MODE"))
    log(f"Mode: {MODE}")
    log(f"Loading import data from '{JSON_PATH}'...")
    assert os.path.exists(JSON_PATH), f"File '{JSON_PATH}' does not exist."
    with open(JSON_PATH, "r") as f:
        import_data: List[Dict] = json.load(f)
    log(f"Import data loaded: {len(import_data)} records.")
    table_names = [table["table_name"] for table in import_data]
    # Go over each record
    conn = create_conn()
    disable_foreign_keys(conn, table_names)
    for i, table in enumerate(import_data):
        file_location, table_name, table_columns, table_refactors = (
            os.path.join(CSV_PATH, table["file_name"].encode("latin1").decode("utf-8")),
            table["table_name"].encode("latin1").decode("utf-8"),
            table["columns"],
            table["refactor"],
        )

        table_info = get_info(table)

        log(f"{table_info}")
        log(f"Importing table ({i + 1}/({len(import_data)})) '{table_name}'...")
        # Load the data from the file
        assert os.path.exists(file_location), f"File '{file_location}' does not exist."
        df = pd.read_csv(
            file_location, sep=table_info["delimiter"], encoding=table_info["encoding"]
        )

        log(f"Data loaded: {len(df.index)} records.", table_name=table_name)
        if MODE == 2:
            df = df.sample(frac=0.1)
            log(
                f"Since the mode is 2, only 10% of the data is used. {len(df.index)} rows",
                table_name=table_name,
            )
        # Rename the columns
        # Bring the keys to lowercase
        table_columns = {key.lower(): value for key, value in table_columns.items()}
        # Bring the column names to lowercase
        df.columns = [column.lower() for column in df.columns]
        df = df.rename(columns=table_columns)

        log(f"Columns renamed: {', '.join(df.columns)}", table_name=table_name)
        df = clean_df(df, table_refactors)

        log("Data cleaned.", table_name=table_name)
        empty_table(table_name, conn)
        # Manually wait 3 seconds to make sure the table is empty
        connection = conn.connect()
        count = connection.execute(
            text(f"SELECT COUNT(*) FROM {table_name}")
        ).fetchone()[0]
        assert count == 0, f"Table '{table_name}' is not empty, {count} records found."
        connection.close()
        log("Table emptied.", table_name=table_name)
        log("Writing data to database...", table_name=table_name)

        table_size = len(df.index)
        chunk_size = int(len(df.index) / 10)
        chunk_size = max(
            min(chunk_size, 100_000), 10_000
        )  # Make sure the chunk size is at least 100_000

        for j in range(0, table_size, chunk_size):
            chunk = df[j : j + chunk_size]
            chunk.to_sql(table_name, conn, if_exists="append", index=False)
            log(
                f"Data written to database. ({j + len(chunk.index)}/{table_size})",
                table_name=table_name,
                overrideable=True,
            )
            del chunk
        log("Data written to database.", table_name=table_name)
        del df

    # Get the foreign keys that don't link to a primary key (orphaned foreign keys)
    log("Checking for orphaned foreign keys...")
    remove_orphaned_keys(conn)
    remove_orphaned_keys(conn)
    remove_orphaned_keys(conn)
    enable_foreign_keys(conn, table_names)
    log("Import complete.")


if __name__ == "__main__":
    main()
