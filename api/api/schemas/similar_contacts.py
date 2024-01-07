from pydantic import BaseModel
from typing import Optional


class SimilarContactInput(BaseModel):
    contact_id: str
    lookalike_count: Optional[int] = 5


class SimilarContactOutput(BaseModel):
    contact_id: str
    similar_contacts: list[str]
