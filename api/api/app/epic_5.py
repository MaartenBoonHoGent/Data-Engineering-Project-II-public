import pandas as pd
import os
import dotenv
from api.app._db import create_conn
from tensorflow.keras.models import load_model
import json
from sqlalchemy import text
import pickle


dotenv.load_dotenv()

connection = create_conn()


def create_df_contacts(
    write_to_csv: bool = False,
    id_array: list = None,
    save_location: str = None,
    only_active_contacts: bool = False,
    engine=None,
) -> pd.DataFrame | None:
    # Start db

    array_conditional = id_array is not None and len(id_array) > 0
    if save_location is None:
        save_location = "contacts.csv"

    query_contacts = """
    SELECT 
    /*Contact*/
    co.crm_contact_id, co.crm_contact_functietitel, co.crm_contact_voka_medewerker, co.crm_persoon_id, co.crm_contact_account
    FROM contact AS co
    """

    print("Loading contacts...")

    if array_conditional:
        query_contacts += "WHERE co.crm_contact_id IN ("
        for i, id in enumerate(id_array):
            query_contacts += f"'{id}'"
            if i != len(id_array) - 1:
                query_contacts += ", "
        query_contacts += ")"

        if only_active_contacts:
            query_contacts += " AND co.crm_contact_status = 'Actief'"
    elif only_active_contacts:
        query_contacts += "WHERE co.crm_contact_status = 'Actief'"

    df_contacts = pd.read_sql(query_contacts, engine)

    query_persoon = """
    SELECT 
    p.crm_persoon_id,
    p.crm_persoon_mail_thema_duurzaamheid, p.crm_persoon_mail_thema_financieel_fiscaal, 
    p.crm_persoon_mail_thema_innovatie, p.crm_persoon_mail_thema_internationaal_ondernemen, 
    p.crm_persoon_mail_thema_mobiliteit, p.crm_persoon_mail_thema_omgeving, p.crm_persoon_mail_thema_sales_marketing_communicatie, p.crm_persoon_mail_thema_strategie_en_algemeen_management, 
    p.crm_Persoon_Mail_thema_talent, p.crm_persoon_mail_thema_welzijn, p.crm_persoon_mail_type_bevraging, 
    p.crm_persoon_mail_type_communities_en_projecten, p.crm_persoon_mail_type_netwerkevenementen, p.crm_persoon_mail_type_nieuwsbrieven, 
    p.crm_persoon_mail_type_opleidingen, p.crm_persoon_mail_type_persberichten_belangrijke_meldingen, p.crm_persoon_marketingcommunicatie
    FROM persoon AS p
    """
    if array_conditional:
        query_persoon += (
            " WHERE p.crm_persoon_id IN ("
            + "','".join(df_contacts.crm_persoon_id.unique().tolist())
            + ")"
        )

    print("Loading personen...")
    df_personen = pd.read_sql(query_persoon, engine)
    df_contacts = pd.merge(df_contacts, df_personen, on="crm_persoon_id")

    query_account = """
    SELECT
    /*Account*/
    a.crm_account_is_voka_entiteit, a.crm_account_ondernemingsaard, a.crm_account_ondernemingstype,
    a.crm_account_primaire_activiteit, a.crm_account_id
    FROM account AS a
    """
    if array_conditional:
        query_account += (
            " WHERE a.crm_account_id IN ("
            + "','".join(df_contacts.crm_contact_account.unique().tolist())
            + ")"
        )
    print("Loading accounts...")

    df_account = pd.read_sql(query_account, engine)

    # Add amount of complaints for each account
    df_account["klachten"] = 0
    query_klachten = """
    SELECT COUNT(*) AS 'aantal_klachten', crm_account_id FROM info_en_klachten 
    GROUP BY crm_account_id 
    """
    if array_conditional:
        query_klachten += (
            " HAVING crm_account_id IN ("
            + "','".join(df_contacts.crm_contact_account.unique().tolist())
            + ")"
        )

    print("Loading klachten...")

    df_klachten = pd.read_sql(query_klachten, engine)

    # Add amount of klachten to df_account
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

    print("Loading activiteitscodes...")

    df_activiteitscode = pd.read_sql(query_activiteitscode, engine)
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
    query_afspraak_account = """
    SELECT 
    aba.crm_afspraak_betreft_account_thema, aba.crm_afspraak_betreft_account_subthema, aba.crm_afspraak_betreft_account_onderwerp,
    aba.crm_afspraak_betreft_account_keyphrases, aba.crm_account_id
    FROM afspraak_betreft_account AS aba
    """

    print("Loading afspraak_betreft_account...")

    df_afspraak_account = pd.read_sql(query_afspraak_account, engine)
    # Split the df into 3 dfs, crm_afspraak_betreft_account_thema, crm_afspraak_betreft_account_subthema, crm_afspraak_betreft_account_keyphrases
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

    print("transforming keyphrases...")

    # Load the campaigns
    query_campaign = """
    SELECT
    c.crm_campagne_id, c.crm_campagne_naam, c.crm_campagne_startdatum, c.crm_campagne_type_campagne, c.crm_campagne_soort_campagne, i.crm_contact_id
    FROM campagne c INNER JOIN inschrijving AS i ON c.crm_campagne_id = i.crm_campagne_id
    """

    print("Loading campaigns...")

    df_campaign = pd.read_sql(query_campaign, engine)
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

    # Temporarily save the df_account
    print("Saving df_account...")
    df_account.to_csv("df_account.csv", index=False)

    # Delete the dfs that are no longer needed
    del df_account
    del df_afspraak_account_keyphrases_2
    del df_afspraak_account_keyphrases
    del df_afspraak_account_subthema_pivot
    del df_afspraak_account_thema_pivot
    del df_afspraak_account_subthema
    del df_afspraak_account_thema
    del df_afspraak_account
    del df_activiteitscode_pivot
    del df_activiteitscode
    del df_klachten
    del df_personen

    # Do the same for afspraak_contact_keyphrases

    query_afspraak_contact_keyphrases = """
    SELECT
    abc.crm_contact_id, abc.crm_afspraak_betreft_contactfiche_thema, abc.crm_afspraak_betreft_contactfiche_subthema, abc.crm_afspraak_betreft_contactfiche_onderwerp, abc.crm_afspraak_betreft_contactfiche_keyphrases
    FROM afspraak_betreft_contact AS abc
    """

    print("Loading afspraak_betreft_contact...")
    df_afspraak_contact_keyphrases = pd.read_sql(
        query_afspraak_contact_keyphrases, engine
    )
    df_afspraak_contact_keyphrases
    # Split the df into 3 dfs, crm_afspraak_betreft_contactfiche_thema, crm_afspraak_betreft_contactfiche_subthema, crm_afspraak_betreft_contactfiche_keyphrases
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

    print("transforming keyphrases...")
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

    # Now do the same for the keyphrases
    unique_types = df_campaign.crm_campagne_type_campagne.unique()
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
    # Set the index
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

    # Load the df_account again
    print("Loading df_account...")
    df_account = pd.read_csv("df_account.csv")

    df_contacts.rename(columns={"crm_contact_account": "crm_account_id"}, inplace=True)

    print("Joining dfs...")
    df_contacts = pd.merge(
        df_contacts,
        df_account,
        on="crm_account_id",
    )
    os.remove("df_account.csv")

    # Bring to csv
    if write_to_csv:
        df_contacts.to_csv(save_location, index=False)
        return

    return df_contacts


async def get_campaign_data(campaign_ids: list, connection) -> pd.DataFrame:
    query_campaigns = f"""
    SELECT
    c.crm_campagne_id, c.crm_campagne_type_campagne, c.crm_campagne_soort_campagne
    FROM campagne c
    WHERE c.crm_campagne_id IN ( '{"', '".join([str(c) for c in campaign_ids])}' )
    """

    df_campaigns = pd.read_sql(query_campaigns, connection)
    return df_campaigns


async def get_contact_data(connection) -> pd.DataFrame:
    return create_df_contacts(False, only_active_contacts=True, engine=connection)


async def main(input_data: dict) -> dict:
    """
    Main function
    """

    connection = create_conn()

    campaign_id: str = input_data["campaign_id"]
    min_viable_probability: str = input_data.get("min_viable_probability", 0.7)

    print(input_data)

    # TODO: Implement

    # Get the campaign data
    model = load_model(os.path.join(os.getenv("EPIC_3_SAVE_LOCATION"), "model.keras"))

    contacts: pd.DataFrame = await get_contact_data(connection=connection)
    print("gotten contacts")
    campaigns: pd.DataFrame = await get_campaign_data(
        [campaign_id], connection=connection
    )
    print("gotten campaigns")
    df = pd.merge(contacts.assign(key=1), campaigns.assign(key=1), on="key").drop(
        "key", axis=1
    )

    assert len(df.index) == len(contacts.index) * len(
        campaigns.index
    ), "Lengths not equal, something went wrong"
    columns_to_remove = [
        "crm_persoon_id",
        "crm_account_id",
        "crm_contact_id",
        "crm_campagne_id",
    ]
    # Key columns
    key_columns = [
        "crm_contact_id",
        "crm_campagne_id",
        "crm_persoon_marketingcommunicatie",
    ]

    keys = df[key_columns]
    # Remove the key columns
    # df.drop(key_columns, axis=1, inplace=True)
    df.drop(columns_to_remove, axis=1, inplace=True)
    print("joined")
    # Load the encoder, imputer & scaler

    with open(
        os.path.join(os.getenv("EPIC_3_SAVE_LOCATION"), "label_encoder.pkl"), "rb"
    ) as f:
        encoder = pickle.load(f)

    with open(
        os.path.join(os.getenv("EPIC_3_SAVE_LOCATION"), "imputer.pkl"), "rb"
    ) as f:
        imputer = pickle.load(f)

    with open(os.path.join(os.getenv("EPIC_3_SAVE_LOCATION"), "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    # Encode the data
    print("encoding")
    df_objects = df.select_dtypes(include="object").columns

    for column in df_objects:
        df[column] = encoder.fit_transform(df[column])

    # Impute the data
    print("imputing")
    df = imputer.transform(df)
    # Scale the data
    print("scaling")
    df = scaler.transform(df)

    # Predict the data

    print("predicting")
    predictions = model.predict(df)

    # Add the predictions to the keys
    keys["prediction"] = predictions
    # Add a column with the multiplier value
    multipliers = json.load(open(os.getenv("API_MULTIPLIERS_FILE"), "r"))

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

    # Remove all predictions with a probability lower than the min_viable_probability

    keys = keys[keys["prediction"] >= min_viable_probability]

    # Rename the columns
    keys.rename(
        {
            "crm_contact_id": "contact_id",
            "crm_campagne_id": "campaign_id",
            "prediction": "probability",
        },
        axis=1,
        inplace=True,
    )

    # Sort the dataframe by the prediction column
    keys.sort_values(by="probability", ascending=False, inplace=True)
    # Delete crm_persoon_marketingcommunicatie
    keys.drop("crm_persoon_marketingcommunicatie", axis=1, inplace=True)

    return keys.to_dict("records")


async def update_multipliers(input_data: dict) -> None:
    """
    Update the multipliers in the database
    """
    input_data = input_data["values"]
    # Make sure the input data has a 'default' key
    if "default" not in input_data:
        raise Exception("Input data must have a 'default' key")

    # Make sure all values are floats
    for key, value in input_data.items():
        if not isinstance(value, float):
            # Try to convert to float
            try:
                input_data[key] = float(value)
            except Exception as e:
                raise Exception(f"Value for key '{key}' must be a float")

    # Set the data to the file
    json.dump(input_data, open(os.getenv("API_MULTIPLIERS_FILE"), "w"), indent=4)
