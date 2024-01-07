from pydantic import BaseModel
from typing import Optional


class CampaignContactInput(BaseModel):
    contact_ids: list[str]
    min_viable_probability: Optional[float] = 0.7
    top_n: Optional[int] = 10


class CampaignContactOutput(BaseModel):
    campaign_id: str
    contact_id: str
    probability: float
    multiplier: float
