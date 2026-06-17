from pydantic import BaseModel


class PredictRequest(BaseModel):
    text_a: str
    text_b: str


class PredictResponse(BaseModel):
    text_a: str
    text_b: str
    probability: float
    label: int
    verdict: str
