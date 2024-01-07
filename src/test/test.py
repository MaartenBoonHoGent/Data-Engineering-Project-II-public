import pandas as pd
import re

# Load the afspraak_alle

path = "C:\\Users\\Maarten Boon\\Documents\\school\\jaar3\\DEP II\\wetransfer_hogent-dataproject-voka-data_2023-09-29_0701\\cdi pageviews.csv"

df = pd.read_csv(path, sep=";", encoding='latin1')


def test_unique_identifier(value) -> bool:
    # Check if the value is in the format of a unique identifier
    # The format is: 00000000-0000-0000-0000-000000000000
    if value is None:
        return True

    if not isinstance(value, str):
        return False

    regex = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    return re.match(regex, value.lower()) is not None


# Get the columns where all values are either None or a unique identifier
columns = [column for column in df.columns if df[column].apply(test_unique_identifier).all()]
print(columns)
