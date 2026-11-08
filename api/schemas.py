from pydantic import BaseModel

class InferenceRequest(BaseModel):
    prompt: str
    max_new_tokens: int | None = None   
    temperature: float | None = None