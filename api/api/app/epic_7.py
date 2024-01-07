import pandas as pd
from api.app._db import create_conn
import json
import requests
import time
from typing import Tuple

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


async def _sendPrompt(
    prompt: str, key: str, provider="openai"
) -> Tuple[str, float] | Tuple[None, float]:
    returnVal = None
    cost = 0.0
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


async def prompt_to_ai(
    prompt: str, contact_id: str, key: str, provider: str = "openai"
) -> dict:
    result, cost = await _sendPrompt(prompt, key, provider=provider)
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


async def main(input_data: dict) -> dict:
    """
    Main function
    """
    contact_id = input_data["contact_id"]
    campaign_ids = input_data["campaign_ids"]

    key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMzA1ZTAzNjYtYzhmNC00OGMzLWFlZTQtOWNjMWExZjE4M2I5IiwidHlwZSI6ImFwaV90b2tlbiJ9.PBTvGyfPO03_1xv6Ug67jzjkQWp5l7g3ahWe5coJl9M"
    provider = "openai"
    # Create contact object
    contact_object = create_contact_object(contact_id)

    # Create campaign object
    campaign_objects = create_campaign_object(campaign_ids)

    # Create prompt
    prompt = write_prompt(contact_object, campaign_objects)
    # Send prompt to AI
    response = await prompt_to_ai(prompt, contact_id, key, provider)

    # Return the response, with following format:

    return {
        "title": response["result"][0]["subject"],
        "body": response["result"][0]["body"],
        "cost": response["cost"],
        "provider": provider,
    }
