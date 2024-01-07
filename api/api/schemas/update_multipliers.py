from pydantic import BaseModel


class UpdateMultipliersInput(BaseModel):
    values: dict
