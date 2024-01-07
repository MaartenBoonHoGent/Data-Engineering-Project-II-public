# Imports

import datetime
import dotenv
import json
import os
from typing import Dict, List
import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path
import dotenv


global UTC_OFFSET
global MODE


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


def ask_input(question: str, check_functions: list, error_messages: list = []):
    # Ask the user for input
    user_input = input(question)
    # Check if the input is valid
    for check_function, error_message in zip(check_functions, error_messages):
        if not check_function(user_input):
            # Print the error message
            print(error_message)
            # The input is not valid, ask again
            return ask_input(question, check_functions)
    # The input is valid, return it
    return user_input


def append_data(df: pd.DataFrame, table_name: str, connection):
    # Append the data to the database

    df.to_sql(
        table_name,
        connection,
        if_exists="append",
        index=False,
    )


def main():
    log("Starting add.py. You can cancel the process at any time by pressing CTRL + C.")
    # Load the .env file
    dotenv.load_dotenv()
    # Get the UTC offset
    # Load the importData as a json file
    JSON_PATH = os.getenv("IMPORT_DATA_PATH")
    CSV_PATH = os.getenv("IMPORT_CSV_PATH")
    CSV_PATH = ask_input(
        f"What folder are the CSV files in? (default: {CSV_PATH})\n",
        [lambda x: os.path.exists(x)],
        ["Provided folder does not exist."],
    )

    CSV_PATH = Path(CSV_PATH)
    CSV_PATH = str(CSV_PATH)

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

    c = "y"
    while c.lower() in ["y", "yes", "ja", "j", ""]:
        # Ask the filename and table name
        filename = ask_input(
            "What is the name of the file you want to import? \n",
            [lambda x: os.path.exists(os.path.join(CSV_PATH, x))],
            ["Provided file does not exist."],
        )

        table_name = ask_input(
            f"What is the name of the table you want to import to? \n Available tables: {', '.join([table['table_name'] for table in import_data])}\n",
            [lambda x: x in [table["table_name"] for table in import_data]],
            [
                "Provided table does not exist. Available tables: "
                + ", ".join([table["table_name"] for table in import_data])
            ],
        )

        table = [
            table
            for table in import_data
            if table["table_name"].encode("latin1").decode("utf-8") == table_name
        ][0]
        table_refactors = table["refactor"]
        table_columns = table["columns"]
        table_info = get_info(table)
        # Get the file location
        file_location = os.path.join(CSV_PATH, filename)

        # Get the df
        df = pd.read_csv(
            file_location, sep=table_info["delimiter"], encoding=table_info["encoding"]
        )
        log(f"Data loaded: {len(df.index)} records.", table_name=table_name)
        table_columns = {key.lower(): value for key, value in table_columns.items()}
        # Bring the column names to lowercase
        df.columns = [column.lower() for column in df.columns]
        df = df.rename(columns=table_columns)

        log(f"Columns renamed: {', '.join(df.columns)}", table_name=table_name)
        df = clean_df(df, table_refactors)

        log("Data cleaned.", table_name=table_name)
        print(df.head())
        if ask_input(
            "Does this look correct? (y/n)\n",
            [lambda x: x.lower() in ["y", "yes", "ja", "j", "n", "no", "nee", "nee"]],
            ["Please answer with yes or no."],
        ) in ["n", "no", "nee", "nee"]:
            log("Aborting import.", table_name=table_name)
            continue

        log("Writing data to database...", table_name=table_name)
        conn = create_conn()
        append_data(df, table_name, conn)
        log("Data written to database.", table_name=table_name)
        conn.close()

        c = ask_input(
            "Do you want to import another file? (y/n)\n",
            [lambda x: x.lower() in ["y", "yes", "ja", "j", "n", "no", "nee", "nee"]],
            ["Please answer with yes or no."],
        )


if __name__ == "__main__":
    main()
