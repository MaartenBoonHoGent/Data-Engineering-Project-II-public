from tensorflow.keras.models import load_model
import dotenv
import os
import json
import pandas as pd
import datetime
from sqlalchemy import create_engine, text
import pickle
import matplotlib.pyplot as plt
from contact_dataframe import create_df_contacts

# Load the environment variables
dotenv.load_dotenv()


def create_conn():
    driver = os.getenv("DB_DRIVER")
    server = os.getenv("DB_SERVER")
    database = os.getenv("DB_NAME")
    trusted_connection = os.getenv("DB_TRUSTED_CONNECTION")

    return create_engine(
        f"mssql+pyodbc://{server}/{database}?trusted_connection={trusted_connection}&driver={driver}"
    )


connection = create_conn()


def get_contact_data() -> pd.DataFrame:
    return create_df_contacts(write_to_csv=False, only_active_contacts=True)


def get_campaign_data(
    campaign_ids: list,
) -> pd.DataFrame:
    query_campaigns = f"""
    SELECT
    c.crm_campagne_id, c.crm_campagne_type_campagne, c.crm_campagne_soort_campagne
    FROM campagne c
    WHERE c.crm_campagne_id IN ( '{"', '".join([str(c) for c in campaign_ids])}' )
    """

    df_campaigns = pd.read_sql(query_campaigns, connection)
    return df_campaigns


def time():
    print("Current time: ", datetime.datetime.now().strftime("%H:%M:%S"))


# Load the model
time()
model = load_model(os.path.join(os.getenv("EPIC_3_SAVE_LOCATION"), "model.keras"))

# Load the data
ids_file = os.path.join(os.getenv("EPIC_5_SAVE_LOCATION"), "generation_data.json")

with open(ids_file, "r") as f:
    ids = json.load(f)

# Get the data from the database
time()
contacts: pd.DataFrame = get_contact_data()
time()
campaigns: pd.DataFrame = get_campaign_data([ids["campaign"]])

print("")
print(len(campaigns.index))
print(len(contacts.index))

df = pd.merge(contacts.assign(key=1), campaigns.assign(key=1), on="key").drop(
    "key", axis=1
)

time()
assert len(df.index) == len(contacts.index) * len(
    campaigns.index
), "Lengths not equal, something went wrong"
columns_to_remove = [
    "crm_persoon_id",
    "crm_account_id",
    "crm_contact_id",
    "crm_campagne_id",
]

for column in df.columns:
    print(column)
# Key columns
key_columns = ["crm_contact_id", "crm_campagne_id", "crm_persoon_marketingcommunicatie"]

keys = df[key_columns]
# Remove the key columns
# df.drop(key_columns, axis=1, inplace=True)
df.drop(columns_to_remove, axis=1, inplace=True)

time()
# Load the encoder, imputer & scaler

with open(
    os.path.join(os.getenv("EPIC_3_SAVE_LOCATION"), "label_encoder.pkl"), "rb"
) as f:
    encoder = pickle.load(f)

with open(os.path.join(os.getenv("EPIC_3_SAVE_LOCATION"), "imputer.pkl"), "rb") as f:
    imputer = pickle.load(f)

with open(os.path.join(os.getenv("EPIC_3_SAVE_LOCATION"), "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

# Encode the data
df_objects = df.select_dtypes(include="object").columns

for column in df_objects:
    df[column] = encoder.fit_transform(df[column])

# Impute the data
df = imputer.transform(df)
# Scale the data
df = scaler.transform(df)

# Predict the data
predictions = model.predict(df)

# Add the predictions to the keys
keys["prediction"] = predictions
# Add a column with the multiplier value
multipliers = json.load(
    open(os.path.join(os.getenv("EPIC_5_SAVE_LOCATION"), "multipliers.json"), "r")
)

# Bring all the values in the 'crm_persoon_marketingcommunicatie' column to lowercase
keys["crm_persoon_marketingcommunicatie"] = keys[
    "crm_persoon_marketingcommunicatie"
].str.lower()

# Add the multiplier column
keys["multiplier"] = keys["crm_persoon_marketingcommunicatie"].apply(
    lambda x: multipliers[x] if x in multipliers else multipliers["default"]
)

# Multiply the predictions with the multiplier
keys["prediction"] = keys["prediction"] * keys["multiplier"]

# Sort the dataframe by the prediction column
keys.sort_values(by="prediction", ascending=False, inplace=True)

# Save the predictions to a csv file
keys.to_csv(
    os.path.join(os.getenv("EPIC_5_SAVE_LOCATION"), "predictions.csv"), index=False
)
