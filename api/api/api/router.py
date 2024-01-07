from fastapi import APIRouter, Form, UploadFile
from fastapi.exceptions import HTTPException
from api.schemas.contacts_campaign import (
    ContactCampaignOutput,
    ContactCampaignInput,
)

from api.schemas.campaigns_contacts import (
    CampaignContactOutput,
    CampaignContactInput,
)
from api.schemas.write_mail import WriteMailOutput, WriteMailInput
from api.schemas.similar_contacts import SimilarContactOutput, SimilarContactInput
from api.schemas.update_multipliers import UpdateMultipliersInput

# Functions
from api.app.epic_2 import main as epic_2
from api.app.epic_3 import main as epic_3
from api.app.epic_4 import main as epic_4
from api.app.epic_5 import (
    main as epic_5,
    update_multipliers as update_multipliers_function,
)
from api.app.epic_7 import main as epic_7

from typing import List


api_router = APIRouter()


@api_router.post(
    "/add-data/",
    tags=["Add Data"],
    summary="Add data to the database",
    description="Add data to the database",
    response_description="The data was successfully added to the database (epic 2)",
    status_code=204,
)  # Accepts a file
async def add_data(
    file: UploadFile = Form(...),
    table_name: str = Form(...),
):
    try:
        return await epic_2(
            file=file,
            table_name=table_name,
        )
    except Exception as e:
        # Return a http error
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post(
    "/campaigns-contacts/",
    tags=["Campaigns Contacts"],
    summary="Returns a list of the likelyhood contacts will go to a campaign",
    description="Returns a list of the likelyhood contacts will go to a campaign (epic 3)",
    response_description="The list was successfully returned",
    response_model=List[ContactCampaignOutput],
    status_code=200,
)
async def campaigns_contacts(
    campaign_contact_input: CampaignContactInput,
):
    try:
        # Convert input to dict
        print(campaign_contact_input.model_dump())
        return await epic_3(campaign_contact_input.model_dump())
    except Exception as e:
        # Return a http error
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post(
    "/similar-contacts/",
    tags=["Similar Contacts"],
    summary="Returns a list of similar contacts for a given contact",
    description="Returns a list of similar contacts for a given contact (epic 4)",
    response_description="The list was successfully returned",
    response_model=SimilarContactOutput,
    status_code=200,
)
async def similar_contacts(
    similar_contact_input: SimilarContactInput,
):
    try:
        return await epic_4(similar_contact_input.model_dump())
    except Exception as e:
        # Return a http error
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post(
    "/contacts-campaign/",
    tags=["Contacts Campaign"],
    summary="Returns a list of contacts for a given campaign",
    description="Returns a list of contacts for a given campaign (epic 5)",
    response_description="The list was successfully returned",
    response_model=List[CampaignContactOutput],
    status_code=200,
)
async def contacts_campaign(
    contact_campaign_input: ContactCampaignInput,
):
    try:
        return await epic_5(contact_campaign_input.model_dump())
    except Exception as e:
        # Return a http error
        raise HTTPException(status_code=500, detail=str(e))


@api_router.put(
    "/update-multipliers/",
    tags=["Update Multipliers"],
    summary="Updates the multipliers for epic 5",
    description="Updates the multipliers for epic 5",
    response_description="The multipliers were successfully updated",
    response_model=None,
    status_code=204,
)
async def update_multipliers(
    update_multipliers_input: UpdateMultipliersInput,
):
    try:
        await update_multipliers_function(update_multipliers_input.model_dump())
    except Exception as e:
        # Return a http error
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post(
    "/write-mail/",
    tags=["Write Mail"],
    summary="Writes a mail for a given contact",
    description="Writes a mail for a given contact (epic 7)",
    response_description="The mail was successfully returned",
    response_model=WriteMailOutput,
    status_code=200,
)
async def write_mail(
    write_mail_input: WriteMailInput,
):
    try:
        return await epic_7(write_mail_input.model_dump())
    except Exception as e:
        # Return a http error
        raise HTTPException(status_code=500, detail=str(e))
