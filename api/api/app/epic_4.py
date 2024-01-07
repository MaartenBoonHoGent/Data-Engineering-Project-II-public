import numpy as np

# Load the model & function
import joblib
from api.app.epic_3 import get_contact_data
from api.app._db import create_conn
import dotenv
import pickle
import os
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

dotenv.load_dotenv()


async def get_data(engine):
    query_contacts = """
    SELECT 
    /*Contact*/
    co.crm_contact_id, co.crm_contact_functietitel, co.crm_contact_voka_medewerker, co.crm_persoon_id, co.crm_contact_account
    FROM contact AS co
    """
    df_contacts = pd.read_sql(query_contacts, engine)
    query_personen = """
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
    df_personen = pd.read_sql(query_personen, engine)

    # Join the two tables on crm_persoon_id

    df_joined = pd.merge(df_contacts, df_personen, on="crm_persoon_id")
    # Make sure the number of rows is the same as the number of unique contacts
    account_query = """
    SELECT
    /*Account*/
    a.crm_account_is_voka_entiteit, a.crm_account_ondernemingsaard, a.crm_account_ondernemingstype,
    a.crm_account_primaire_activiteit, a.crm_account_id
    FROM account AS a
    """

    df_account = pd.read_sql(account_query, engine)
    # Add amount of complaints for each account
    df_account["klachten"] = 0
    query_klachten = """
    SELECT COUNT(*) AS 'aantal_klachten', crm_account_id FROM info_en_klachten 
    GROUP BY crm_account_id
    """

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
    # Add afspraak
    query_afspraak_account = """
    SELECT 
    aba.crm_afspraak_betreft_account_thema, aba.crm_afspraak_betreft_account_subthema, aba.crm_afspraak_betreft_account_onderwerp,
    aba.crm_afspraak_betreft_account_keyphrases, aba.crm_account_id
    FROM afspraak_betreft_account AS aba
    """

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
    # Load the campaigns
    query_campaign = """
    SELECT
    c.crm_campagne_id, c.crm_campagne_naam, c.crm_campagne_startdatum, c.crm_campagne_type_campagne, c.crm_campagne_soort_campagne, i.crm_contact_id
    FROM campagne c INNER JOIN inschrijving AS i ON c.crm_campagne_id = i.crm_campagne_id
    """

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
    # Set the index
    df_afspraak_account_keyphrases_2.set_index("crm_account_id", inplace=True)
    # Add to the df_account
    # Create the columns first
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

    df_afspraak_contact_keyphrases = pd.read_sql(
        query_afspraak_contact_keyphrases, engine
    )
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

    # Add to the df_contacts
    # Create the columns first

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
    df_account = pd.read_csv("df_account.csv")
    # Perform an inner join on df_contacts and df_account
    # Rename contact a crm_contact_account to crm_account_id
    df_contacts.rename(columns={"crm_contact_account": "crm_account_id"}, inplace=True)

    df_joined = pd.merge(
        df_contacts,
        df_account,
        on="crm_account_id",
    )

    return df_joined


def load_kmeans_model(filename="./data/kmeans_model.joblib"):
    # Load the KMeans model from a file
    return joblib.load(filename)


async def find_lookalikes(data, selected_id, top_n=10):
    id_column = "crm_contact_id"

    # load the model
    kmeans_model = load_kmeans_model(
        os.path.join(os.getenv("EPIC_4_SAVE_LOCATION"), "kmeans_model.joblib")
    )

    print("Loaded model")

    # Add cluster labels to the original dataset
    data["Cluster"] = kmeans_model.labels_

    # Select a row based on the specified ID
    selected_row = data[data[id_column] == selected_id]

    print("Selected row")
    # Check if selected_row is not empty
    if not selected_row.empty:
        # Get the cluster label of the selected row
        selected_cluster = selected_row["Cluster"].values[0]

        # Filter the dataset to get all points in the same cluster
        cluster_points = data[data["Cluster"] == selected_cluster]

        print("Cluster points")

        # Calculate the distance from the selected point to all points in the cluster
        distances = np.linalg.norm(
            cluster_points.drop(columns=["Cluster", id_column])
            - selected_row.drop(columns=["Cluster", id_column]),
            axis=1,
        )

        # Combine distances with the cluster_points DataFrame
        cluster_points["Distance"] = distances

        # Sort by distance and select the top N
        top_lookalikes = cluster_points.sort_values(by="Distance").head(top_n + 1)

        # Remove the cluster label and distance to avoid disrupting the original dataset
        data = data.drop(columns=["Cluster"])
        top_lookalikes = top_lookalikes.drop(columns=["Cluster", "Distance"])

        # Only keep the top N lookalikes
        top_lookalikes = top_lookalikes.head(top_n)

        # Filter the id
        top_lookalikes = top_lookalikes["crm_contact_id"]

        # skip the first row because it is the same as the selected row
        top_lookalikes = top_lookalikes[1:]

        # Get only the ids as a list
        top_lookalikes = top_lookalikes.values

        # Return the top N lookalikes
        return top_lookalikes.tolist()

    else:
        # Handle the case where selected_row is empty
        print(f"No row found with {id_column} equal to {selected_id}")
        return None


def format_df(df) -> pd.DataFrame:
    id_col = df["crm_contact_id"]
    df_clipped = df.drop(["crm_contact_id"], axis=1)

    print("loading imputer")
    print(
        os.path.normpath(os.path.join(os.getenv("EPIC_4_SAVE_LOCATION"), "imputer.pkl"))
    )
    with open(
        os.path.normpath(
            os.path.join(os.getenv("EPIC_4_SAVE_LOCATION"), "imputer.pkl")
        ),
        "rb",
    ) as f:
        imputer = pickle.load(f)

    print("Imputing")
    imputer.fit(df_clipped)
    df_clipped = pd.DataFrame(imputer.transform(df_clipped), columns=df_clipped.columns)
    print("Imputed")
    # Load the scaler
    with open(os.path.join(os.getenv("EPIC_4_SAVE_LOCATION"), "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    scaler.fit(df_clipped)
    df_scaled = scaler.transform(df_clipped)
    print("Scaled")
    df_scaled = pd.DataFrame(df_scaled, columns=df_clipped.columns)
    df = pd.concat([id_col, df_scaled], axis=1)
    return df


async def main(input_data: dict) -> dict:
    """
    Main function
    """
    conn = create_conn()

    # Get the contact data
    contact_data = await get_data(conn)
    print("Step 1")
    contact_data = format_df(contact_data)

    print("Step 2")
    # Find the lookalikes
    lookalikes = await find_lookalikes(
        contact_data, input_data["contact_id"], top_n=input_data["lookalike_count"]
    )

    print("Step 3")
    if lookalikes is not None:
        return {
            "contact_id": input_data["contact_id"],
            "similar_contacts": lookalikes.tolist(),
        }
    else:
        # Throw an error
        raise Exception("No lookalikes found")
