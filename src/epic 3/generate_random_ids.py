import pandas as pd
import os
import dotenv
import json
from sqlalchemy import create_engine, text

dotenv.load_dotenv()

number_of_ids_contacts = int(
    input(f"For how many contacts do you want to generate random ids? ")
)
number_of_ids_campaigns = int(
    input(f"For how many campaigns do you want to generate random ids? ")
)

# Path: src/epic%203/generate_random_ids.py

query_contacts = f"SELECT TOP({number_of_ids_contacts}) crm_contact_id FROM dbo.contact ORDER BY RAND();"
query_campaigns = f"SELECT TOP({number_of_ids_campaigns}) crm_campagne_id FROM dbo.campagne ORDER BY RAND();"


def create_conn():
    driver = os.getenv("DB_DRIVER")
    server = os.getenv("DB_SERVER")
    database = os.getenv("DB_NAME")
    trusted_connection = os.getenv("DB_TRUSTED_CONNECTION")

    return create_engine(
        f"mssql+pyodbc://{server}/{database}?trusted_connection={trusted_connection}&driver={driver}"
    )


connection = create_conn()
df_contacts = pd.read_sql(query_contacts, connection)
df_campaigns = pd.read_sql(query_campaigns, connection)

# Bring both to an array
contacts = df_contacts["crm_contact_id"].to_numpy().tolist()
campaigns = df_campaigns["crm_campagne_id"].to_numpy().tolist()

print(f"contacts: {len(contacts)}")
print(f"campaigns: {len(campaigns)}")
# Create a dictionary with the random ids
random_ids = {"contacts": contacts, "campaigns": campaigns}


# Save the dictionary to a file
json_file = os.path.join(os.getenv("EPIC_3_SAVE_LOCATION"), "generation_data.json")

with open(json_file, "w") as f:
    json.dump(random_ids, f)
