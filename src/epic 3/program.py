from tensorflow.keras.models import load_model
import dotenv
import os
import json
import pandas as pd
from sqlalchemy import create_engine, text
import pickle
import matplotlib.pyplot as plt


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


def get_contact_data(contact_ids: list) -> pd.DataFrame:
    query_contacts = f"""
    SELECT 
    /*Contact*/
    co.crm_contact_id, co.crm_contact_functietitel, co.crm_contact_voka_medewerker, co.crm_persoon_id, co.crm_contact_account
    FROM contact AS co
    WHERE co.crm_contact_id IN ('{"','".join(contact_ids)}')
    """
    df_contacts = pd.read_sql(query_contacts, connection)

    query_persoon = f"""
    SELECT 
    p.crm_persoon_id,
    p.crm_persoon_mail_thema_duurzaamheid, p.crm_persoon_mail_thema_financieel_fiscaal, 
    p.crm_persoon_mail_thema_innovatie, p.crm_persoon_mail_thema_internationaal_ondernemen, 
    p.crm_persoon_mail_thema_mobiliteit, p.crm_persoon_mail_thema_omgeving, p.crm_persoon_mail_thema_sales_marketing_communicatie, p.crm_persoon_mail_thema_strategie_en_algemeen_management, 
    p.crm_Persoon_Mail_thema_talent, p.crm_persoon_mail_thema_welzijn, p.crm_persoon_mail_type_bevraging, 
    p.crm_persoon_mail_type_communities_en_projecten, p.crm_persoon_mail_type_netwerkevenementen, p.crm_persoon_mail_type_nieuwsbrieven, 
    p.crm_persoon_mail_type_opleidingen, p.crm_persoon_mail_type_persberichten_belangrijke_meldingen, p.crm_persoon_marketingcommunicatie
    FROM persoon AS p WHERE p.crm_persoon_id IN ('{"','".join(df_contacts.crm_persoon_id.unique().tolist())}')
    """

    df_persoon = pd.read_sql(query_persoon, connection)

    df_contacts = pd.merge(df_contacts, df_persoon, on="crm_persoon_id")

    # Account
    query_account = f"""
    SELECT
    /*Account*/
    a.crm_account_is_voka_entiteit, a.crm_account_ondernemingsaard, a.crm_account_ondernemingstype,
    a.crm_account_primaire_activiteit, a.crm_account_id
    FROM account AS a
    WHERE a.crm_account_id IN ('{"','".join(df_contacts.crm_contact_account.unique().tolist())}')
    """

    df_account = pd.read_sql(query_account, connection)
    df_account["klachten"] = 0

    query_klachten = f"""
    SELECT COUNT(*) AS 'aantal_klachten', crm_account_id FROM info_en_klachten 
    GROUP BY crm_account_id 
    HAVING crm_account_id IN ('{"','".join(df_contacts.crm_contact_account.unique().tolist())}')
    """

    df_klachten = pd.read_sql(query_klachten, connection)
    for index, row in df_klachten.iterrows():
        df_account.loc[
            df_account.crm_account_id == row["crm_account_id"], "klachten"
        ] = row["aantal_klachten"]

    query_activiteitscode = """
        SELECT

        /*Activiteitscode*/
        act.crm_activiteitscode_naam, aa.crm_account_id
        FROM account_activiteitscode AS aa
        INNER JOIN activiteitscode AS act ON act.crm_activiteitscode_id = aa.crm_activiteitscode_id
        """
    df_activiteitscode = pd.read_sql(query_activiteitscode, connection)
    df_activiteitscode_pivot = df_activiteitscode.pivot_table(
        index="crm_account_id",
        columns="crm_activiteitscode_naam",
        aggfunc=len,
        fill_value=0,
    )

    # Add the columns of df_activiteitscode_pivot to df_account, except for the index
    for column in df_activiteitscode_pivot.columns:
        df_account["activiteitscode_" + column] = 0

    # Add the values of df_activiteitscode_pivot to df_account
    amnt = len(df_activiteitscode_pivot.index)
    i = 0
    for index, row in df_activiteitscode_pivot.iterrows():
        print(f"{i} / {amnt}", end="\r")
        i += 1
        for column in df_activiteitscode_pivot.columns:
            if row[column] > 0:
                df_account.loc[
                    df_account.crm_account_id == index, "activiteitscode_" + column
                ] = 1

    # Add afspraak
    query_afspraak_account = """
    SELECT 
    aba.crm_afspraak_betreft_account_thema, aba.crm_afspraak_betreft_account_subthema, aba.crm_afspraak_betreft_account_onderwerp,
    aba.crm_afspraak_betreft_account_keyphrases, aba.crm_account_id
    FROM afspraak_betreft_account AS aba
    """

    df_afspraak_account = pd.read_sql(query_afspraak_account, connection)

    df_afspraak_account_thema = df_afspraak_account[
        ["crm_afspraak_betreft_account_thema", "crm_account_id"]
    ]

    df_afspraak_account_subthema = df_afspraak_account[
        ["crm_afspraak_betreft_account_subthema", "crm_account_id"]
    ]
    df_afspraak_account_keyphrases = df_afspraak_account[
        ["crm_afspraak_betreft_account_keyphrases", "crm_account_id"]
    ]

    # Pivot the dfs to get a column for each theme, with a 1 if the account has that theme
    df_afspraak_account_thema_pivot = df_afspraak_account_thema.pivot_table(
        index="crm_account_id",
        columns="crm_afspraak_betreft_account_thema",
        aggfunc=len,
        fill_value=0,
    )
    df_afspraak_account_subthema_pivot = df_afspraak_account_subthema.pivot_table(
        index="crm_account_id",
        columns="crm_afspraak_betreft_account_subthema",
        aggfunc=len,
        fill_value=0,
    )

    # Add the columns of df_afspraak_account_thema_pivot to df_account, except for the index
    for column in df_afspraak_account_thema_pivot.columns:
        df_account["afspraak_account_thema_" + column] = 0

    # Add the columns of df_afspraak_account_subthema_pivot to df_account, except for the index
    for column in df_afspraak_account_subthema_pivot.columns:
        df_account["afspraak_account_subthema_" + column] = 0

    # Add the values of df_afspraak_account_thema_pivot to df_account
    amnt = len(df_afspraak_account_thema_pivot.index)
    i = 0
    for index, row in df_afspraak_account_thema_pivot.iterrows():
        print(f"{i} / {amnt}", end="\r")
        i += 1
        for column in df_afspraak_account_thema_pivot.columns:
            if row[column] > 0:
                df_account.loc[
                    df_account.crm_account_id == index,
                    "afspraak_account_thema_" + column,
                ] = 1

    # Add the values of df_afspraak_account_subthema_pivot to df_account
    amnt = len(df_afspraak_account_subthema_pivot.index)
    i = 0
    for index, row in df_afspraak_account_subthema_pivot.iterrows():
        print(f"{i} / {amnt}", end="\r")
        i += 1
        for column in df_afspraak_account_subthema_pivot.columns:
            if row[column] > 0:
                df_account.loc[
                    df_account.crm_account_id == index,
                    "afspraak_account_subthema_" + column,
                ] = 1

    # Load the campaigns

    query_campaign = """
    SELECT
    c.crm_campagne_id, c.crm_campagne_naam, c.crm_campagne_startdatum, c.crm_campagne_type_campagne, c.crm_campagne_soort_campagne, i.crm_contact_id
    FROM campagne c INNER JOIN inschrijving AS i ON c.crm_campagne_id = i.crm_campagne_id
    """

    df_campaign = pd.read_sql(query_campaign, connection)
    unique_types = df_campaign.crm_campagne_type_campagne.unique()
    columns_keyphrases = ["crm_account_id"]
    [
        columns_keyphrases.append("afspraak_account_" + unique_type.lower())
        for unique_type in unique_types
    ]
    n = {}
    for column in columns_keyphrases:
        n[column] = []
    columns_keyphrases = n
    for i, acc_id in enumerate(df_afspraak_account_keyphrases.crm_account_id.unique()):
        columns_keyphrases["crm_account_id"].append(acc_id)
        # Get all the rows for this account
        df_acc = df_afspraak_account_keyphrases[
            df_afspraak_account_keyphrases.crm_account_id == acc_id
        ]

        unique_type_list = [0 for i in range(len(unique_types))]
        # Iterate over the rows
        for index, row in df_acc.iterrows():
            if row["crm_afspraak_betreft_account_keyphrases"] is None:
                continue
            for i, unique_type in enumerate(unique_types):
                if (
                    unique_type.lower()
                    in row["crm_afspraak_betreft_account_keyphrases"].lower()
                ):
                    unique_type_list[i] += 1
        for i, unique_type in enumerate(unique_types):
            name = "afspraak_account_" + unique_type.lower()
            columns_keyphrases[name].append(unique_type_list[i])

    df_afspraak_account_keyphrases_2 = pd.DataFrame(columns_keyphrases)
    df_afspraak_account_keyphrases_2.set_index("crm_account_id", inplace=True)
    for column in df_afspraak_account_keyphrases_2.columns:
        df_account[column] = 0

    # Add the values
    amnt = len(df_afspraak_account_keyphrases_2.index)
    i = 0
    for index, row in df_afspraak_account_keyphrases_2.iterrows():
        print(f"{i} / {amnt}", end="\r")
        i += 1
        for column in df_afspraak_account_keyphrases_2.columns:
            if row[column] > 0:
                df_account.loc[df_account.crm_account_id == index, column] = 1

    # Do the same for afspraak_contact_keyphrases
    query_afspraak_contact_keyphrases = """
    SELECT
    abc.crm_contact_id, abc.crm_afspraak_betreft_contactfiche_thema, abc.crm_afspraak_betreft_contactfiche_subthema, abc.crm_afspraak_betreft_contactfiche_onderwerp, abc.crm_afspraak_betreft_contactfiche_keyphrases
    FROM afspraak_betreft_contact AS abc 
    """

    df_afspraak_contact_keyphrases = pd.read_sql(
        query_afspraak_contact_keyphrases, connection
    )

    df_afspraak_contact_thema = df_afspraak_contact_keyphrases[
        ["crm_afspraak_betreft_contactfiche_thema", "crm_contact_id"]
    ]
    df_afspraak_contact_subthema = df_afspraak_contact_keyphrases[
        ["crm_afspraak_betreft_contactfiche_subthema", "crm_contact_id"]
    ]

    df_afspraak_contact_keyphrases = df_afspraak_contact_keyphrases[
        ["crm_afspraak_betreft_contactfiche_keyphrases", "crm_contact_id"]
    ]

    # Pivot the dfs to get a column for each theme, with a 1 if the account has that theme
    df_afspraak_contact_thema_pivot = df_afspraak_contact_thema.pivot_table(
        index="crm_contact_id",
        columns="crm_afspraak_betreft_contactfiche_thema",
        aggfunc=len,
        fill_value=0,
    )

    df_afspraak_contact_subthema_pivot = df_afspraak_contact_subthema.pivot_table(
        index="crm_contact_id",
        columns="crm_afspraak_betreft_contactfiche_subthema",
        aggfunc=len,
        fill_value=0,
    )
    df_afspraak_contact_keyphrases = pd.read_sql(
        query_afspraak_contact_keyphrases, connection
    )

    # Add the columns of df_afspraak_contact_thema_pivot to df_contacts, except for the index
    for column in df_afspraak_contact_thema_pivot.columns:
        df_contacts["afspraak_contact_thema_" + column] = 0

    # Add the columns of df_afspraak_contact_subthema_pivot to df_contacts, except for the index

    for column in df_afspraak_contact_subthema_pivot.columns:
        df_contacts["afspraak_contact_subthema_" + column] = 0

    # Add the values of df_afspraak_contact_thema_pivot to df_contacts
    amnt = len(df_afspraak_contact_thema_pivot.index)
    i = 0
    for index, row in df_afspraak_contact_thema_pivot.iterrows():
        print(f"{i} / {amnt}", end="\r")
        i += 1
        for column in df_afspraak_contact_thema_pivot.columns:
            if row[column] > 0:
                df_contacts.loc[
                    df_contacts.crm_contact_id == index,
                    "afspraak_contact_thema_" + column,
                ] = 1

    # Add the values of df_afspraak_contact_subthema_pivot to df_contacts
    amnt = len(df_afspraak_contact_subthema_pivot.index)
    i = 0
    for index, row in df_afspraak_contact_subthema_pivot.iterrows():
        print(f"{i} / {amnt}", end="\r")
        i += 1
        for column in df_afspraak_contact_subthema_pivot.columns:
            if row[column] > 0:
                df_contacts.loc[
                    df_contacts.crm_contact_id == index,
                    "afspraak_contact_subthema_" + column,
                ] = 1

    columns_keyphrases = ["crm_contact_id"]
    [
        columns_keyphrases.append("afspraak_contact_" + unique_type.lower())
        for unique_type in unique_types
    ]

    n = {}

    for column in columns_keyphrases:
        n[column] = []

    columns_keyphrases = n

    for i, cont_id in enumerate(df_afspraak_contact_keyphrases.crm_contact_id.unique()):
        df_c = df_afspraak_contact_keyphrases[
            df_afspraak_contact_keyphrases.crm_contact_id == cont_id
        ]
        columns_keyphrases["crm_contact_id"].append(cont_id)
        unique_type_list = [0 for i in range(len(unique_types))]

        for index, row in df_c.iterrows():
            if row["crm_afspraak_betreft_contactfiche_keyphrases"] is None:
                continue
            for i, unique_type in enumerate(unique_types):
                if (
                    unique_type.lower()
                    in row["crm_afspraak_betreft_contactfiche_keyphrases"].lower()
                ):
                    unique_type_list[i] += 1

        for i, unique_type in enumerate(unique_types):
            name = "afspraak_contact_" + unique_type.lower()
            columns_keyphrases[name].append(unique_type_list[i])

    df_afspraak_contact_keyphrases_2 = pd.DataFrame(columns_keyphrases)
    df_afspraak_contact_keyphrases_2.shape, df_afspraak_contact_keyphrases_2.crm_contact_id.nunique()

    df_afspraak_contact_keyphrases_2.set_index("crm_contact_id", inplace=True)
    for column in df_afspraak_contact_keyphrases_2.columns:
        df_contacts[column] = 0

    # Add the values
    amnt = len(df_afspraak_contact_keyphrases_2.index)
    i = 0
    for index, row in df_afspraak_contact_keyphrases_2.iterrows():
        print(f"{i} / {amnt}", end="\r")
        i += 1
        for column in df_afspraak_contact_keyphrases_2.columns:
            if row[column] > 0:
                df_contacts.loc[df_contacts.crm_contact_id == index, column] = 1

    df_contacts.rename(columns={"crm_contact_account": "crm_account_id"}, inplace=True)
    df_contacts = pd.merge(
        df_contacts,
        df_account,
        on="crm_account_id",
    )

    return df_contacts


def get_campaign_data() -> pd.DataFrame:
    query_campaigns = """
    SELECT
    c.crm_campagne_id, c.crm_campagne_type_campagne, c.crm_campagne_soort_campagne
    FROM campagne c
    WHERE crm_campagne_reden_van_status = 'Nieuw'
    """

    df_campaigns = pd.read_sql(query_campaigns, connection)
    return df_campaigns


# Load the model

model = load_model(os.path.join(os.getenv("EPIC_3_SAVE_LOCATION"), "model.keras"))

# Load the data
ids_file = os.path.join(os.getenv("EPIC_3_SAVE_LOCATION"), "generation_data.json")

with open(ids_file, "r") as f:
    ids = json.load(f)

# Get the data from the database
contacts: pd.DataFrame = get_contact_data(ids["contacts"])
campaigns: pd.DataFrame = get_campaign_data()

print("")
print(len(campaigns.index))
print(len(contacts.index))

df = pd.merge(contacts.assign(key=1), campaigns.assign(key=1), on="key").drop(
    "key", axis=1
)

assert len(df.index) == len(contacts.index) * len(
    campaigns.index
), "Lengths not equal, something went wrong"
columns_to_remove = [
    "crm_persoon_id",
    "crm_account_id",
]

# Key columns
key_columns = [
    "crm_contact_id",
    "crm_campagne_id",
]

keys = df[key_columns]
# Remove the key columns
df.drop(key_columns, axis=1, inplace=True)
df.drop(columns_to_remove, axis=1, inplace=True)

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
# Round the predictions
keys["prediction"] = keys["prediction"].round(4)
# Get the top 20 predictions for each contact
top_20 = (
    keys.groupby("crm_contact_id")
    .apply(lambda x: x.nlargest(20, "prediction"))
    .reset_index(drop=True)
)
top_20.sort_values(by=["crm_contact_id", "prediction"], inplace=True, ascending=False)

# Save the predictions as a csv
top_20.to_csv(
    os.path.join(os.getenv("EPIC_3_SAVE_LOCATION"), "predictions.csv"), index=False
)
