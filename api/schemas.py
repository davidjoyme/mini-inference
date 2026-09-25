"""api/schemas.py - request formats for the HTTP API."""
from pydantic import BaseModel


class InferenceRequest(BaseModel):
    prompt: str
    max_new_tokens: int | None = None   # None = use the default from config.py
    temperature: float | None = None