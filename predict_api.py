import os

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import joblib
import requests
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()
model = joblib.load("model.pkl")
SECRET_TOKEN = os.getenv("SECRET_TOKEN")
species_map = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

# load model
MODEL_PATH = "model.pkl"
MODEL_URL = "https://github.com/Ssebi1/mlops/releases/download/latest/model.pkl"

if not os.path.exists(MODEL_PATH):
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(r.content)


class Input(BaseModel):
    data: list[float]


@app.post("/predict")
def predict(input: Input, x_token: str = Header(...)):
    if x_token != SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    prediction = model.predict([input.data])
    return {
        "prediction": int(prediction[0]),
        "species": species_map[int(prediction[0])]
    }
