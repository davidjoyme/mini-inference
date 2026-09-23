from fastapi import FastAPI
from pydantic import BaseModel 
from transformers import pipeline


generator = pipeline("text-generation", model='allenai/OLMoE-1B-7B-0125-Instruct', device_map="auto")
app=FastAPI()

class mini_inference(BaseModel):
     input : str
     max_new_tokens:int =200

@app.post("/generate")
def hello(engine:mini_inference):
    messages = [{"role":"user" ,"content":engine.input}]
    result = generator(messages, max_new_tokens=engine.max_new_tokens)

    reply = result[0]["generated_text"][-1]["content"]
    return {"input": engine.input, "output": reply}

