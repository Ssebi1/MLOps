import os

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import joblib
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
