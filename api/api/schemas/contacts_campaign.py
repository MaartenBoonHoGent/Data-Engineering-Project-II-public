from pydantic import BaseModel
from typing import Optional


class ContactCampaignInput(BaseModel):
    campaign_id: str
    min_viable_probability: Optional[float] = 0.7


class ContactCampaignOutput(BaseModel):
    campaign_id: str
    contact_id: str
    probability: float
