from pydantic import BaseModel


class WriteMailInput(BaseModel):
    contact_id: str
    campaign_ids: list[str]


class WriteMailOutput(BaseModel):
    title: str
    body: str
    cost: float
    provider: str
