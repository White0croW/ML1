from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Загрузка предварительно обученной модели
sentiment_analysis = pipeline("sentiment-analysis")


class TextInput(BaseModel):
    text: str


@app.post("/predict/")
async def predict(input: TextInput):
    result = sentiment_analysis(input.text)[0]
    return {"label": result["label"], "score": result["score"]}
