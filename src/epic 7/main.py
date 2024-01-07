"""
INPUT
- 1 contactid
- X aantal campagnes

---
DATA
- Contact: Contact_id (voor naam), Activiteit bedrijf, crm_account_id (voor naam),   functietitel ,  functie
- Campagne: crm_campagne_url_voka_be, crm_campagne_soort_campagne, crm_campagne_type_campagne, crm_campagne_startdatum, crm_campagne_naam, crm_campagne_naam_in_email


---
OUTPUT
- Text


INPUT
INPUT -> DATA
DATA -> PROMPT
PROMPT ENGINEERING -> OUTPUT (Gestructureerd)
"""
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import pandas as pd
import json
import requests
import time
from typing import Tuple

# Load environment variables from .env file
load_dotenv()


# Start db
def create_conn():
    driver = os.getenv("DB_DRIVER")
    server = os.getenv("DB_SERVER")
    database = os.getenv("DB_NAME")
    trusted_connection = os.getenv("DB_TRUSTED_CONNECTION")

    return create_engine(
        f"mssql+pyodbc://{server}/{database}?trusted_connection={trusted_connection}&driver={driver}"
    )


conn = create_conn()


def create_contact_object(contact_id: str) -> dict:
    # query
    query = f"""
    SELECT
        contact.crm_contact_id,
        account.crm_account_primaire_activiteit,
        contact.crm_contact_functietitel,
        account.crm_account_adres_plaats

    FROM
        contact
    JOIN
        account ON contact.crm_contact_account = account.crm_account_id
    WHERE
        contact.crm_contact_id = '{contact_id}';
    """

    # execute query
    result = pd.read_sql(query, conn)

    #  Bring the result to a list of dictionaries
    result = result.to_dict(orient="records")

    if len(result) > 1:
        raise ValueError("More than one contact found")
    elif len(result) == 0:
        raise ValueError("No contact found")
    return result[0]


def create_campaign_object(campaign_ids: list) -> list[dict]:
    query = """
        SELECT
            crm_campagne_url_voka_be,
            crm_campagne_soort_campagne,
            crm_campagne_type_campagne,
            crm_campagne_startdatum,
            crm_campagne_einddatum,
            crm_campagne_naam,
            crm_campagne_naam_in_email
        FROM
            campagne
        WHERE
            crm_campagne_id IN ('{}');
        """.format(
        "', '".join(_ for _ in campaign_ids)
    )

    # execute query into pandas dataframe
    campaign_objects = pd.read_sql(
        query,
        conn,
    )

    campaign_objects["crm_campagne_startdatum"] = campaign_objects[
        "crm_campagne_startdatum"
    ].dt.strftime("%d/%m/%Y")
    # Do the same for the end date
    campaign_objects["crm_campagne_einddatum"] = campaign_objects[
        "crm_campagne_einddatum"
    ].dt.strftime("%d/%m/%Y")

    #  Bring the result to a list of dictionaries
    campaign_objects = campaign_objects.to_dict(orient="records")
    return campaign_objects


# JSON in prompt

campaign_ids = [
    "B4E3E30A-E6CA-EC11-A7B5-000D3A20A90F",
    "1B445E22-A9CB-EC11-A7B5-000D3A20A90F",
    "5F62AF96-0E2D-ED11-9DB1-000D3A211D18",
]
testid = "E14A8BC8-DCC9-EC11-A7B5-000D3A20A90F"


def write_prompt(
    contact_object: dict,
    campaign_objects: list[dict],
) -> str:
    return f"""
Using the provided contact information and campaign details, generate a personalized email and the subject of the mail in Dutch. The recipient hast the following info: \n```{json.dumps(contact_object)}```. \n Explain the value of all the following campaigns for this person: \n```{json.dumps(campaign_objects)}```.\n When addressing the reader, please reference them as [NAME], so we can substitute this for the actual name later. Please format your response as a JSON object with the following structure. Write \\n for newlines, and \\t for tabs:

[
    {{
        "subject": "str",
        "body": "str"
    }},
    ...
]
"""

    # Prompt to AI


def _sendPrompt(prompt: str, key: str) -> Tuple[str, float] | Tuple[None, float]:
    returnVal = None
    cost = 0.0
    provider = "openai"
    temperature = 0.6
    max_attempts = 3
    url = "https://api.edenai.run/v2/text/generation"

    headers = {"Authorization": f"Bearer {key}"}

    for i in range(max_attempts):
        payload = {
            "text": prompt,
            "providers": ", ".join([provider]),
            "temperature": temperature,
            "max_tokens": 2048,
            "settings": {"openai": "text-davinci-003"},
        }
        response = requests.post(
            url,
            json=payload,
            headers=headers,
        )
        result = json.loads(response.text)
        if result[provider]["status"] == "success":
            returnVal = result[provider]["generated_text"].strip()
            cost = result[provider]["cost"]
            break

        time.sleep(3)
        if i == max_attempts - 1:
            raise Exception(f"Failed to generate text for prompt: {prompt} -- {result}")

    return (returnVal, cost)


def load_json_list_from_response(response: str) -> list:
    try:
        return json.loads(
            "[" + response[response.find("[") + 1 : response.rfind("]")] + "]"
        )
    except json.JSONDecodeError:
        raise Exception(f"Failed to load JSON from response: {response}")


def prompt_to_ai(prompt: str, contact_id: str, key: str) -> dict:
    result, cost = _sendPrompt(prompt, key)
    # Load the result into a dictionary
    result = load_json_list_from_response(result)
    # For each value in the result, replace [NAME] with the id of the contact

    for i in range(len(result)):
        result[i]["body"] = result[i]["body"].replace("[NAME]", f"[{contact_id}]")
        result[i]["subject"] = result[i]["subject"].replace("[NAME]", f"[{contact_id}]")

    return {
        "result": result,
        "cost": cost,
    }


def main():
    # Get contact id
    contact_id = "E14A8BC8-DCC9-EC11-A7B5-000D3A20A90F"
    key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMzA1ZTAzNjYtYzhmNC00OGMzLWFlZTQtOWNjMWExZjE4M2I5IiwidHlwZSI6ImFwaV90b2tlbiJ9.PBTvGyfPO03_1xv6Ug67jzjkQWp5l7g3ahWe5coJl9M"

    campaign_ids = [
        "B4E3E30A-E6CA-EC11-A7B5-000D3A20A90F",
        "1B445E22-A9CB-EC11-A7B5-000D3A20A90F",
        "5F62AF96-0E2D-ED11-9DB1-000D3A211D18",
    ]

    # Create contact object
    contact_object = create_contact_object(contact_id)

    # Create campaign object
    campaign_objects = create_campaign_object(campaign_ids)

    # Create prompt
    prompt = write_prompt(contact_object, campaign_objects)
    # Send prompt to AI
    response = prompt_to_ai(prompt, contact_id, key)
    # Get the response to a file
    with open("src/epic 7/data/response.json", "w") as f:
        json.dump(response, f, indent=4)


if __name__ == "__main__":
    main()
