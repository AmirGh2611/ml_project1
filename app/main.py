from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException

APP_VERSION = "1.0.0"

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
ml: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    bundle = joblib.load(MODEL_PATH)
    ml["scaler"] = bundle["scaler"]
    ml["model"] = bundle["model"]
    ml["target_names"] = ["setosa", "versicolor", "virginica"]
    yield
    ml.clear()


app = FastAPI(
    title="Iris Classifier API",
    version=APP_VERSION,
    lifespan=lifespan,
)


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, examples=[5.1])
    sepal_width: float = Field(..., gt=0, examples=[3.5])
    petal_length: float = Field(..., gt=0, examples=[1.4])
    petal_width: float = Field(..., gt=0, examples=[0.2])


class Prediction(BaseModel):
    predicted_class: int
    predicted_label: str
    probabilities: list[float]


# ---------- endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(features: IrisFeatures):
    if "model" not in ml:
        raise HTTPException(status_code=503, detail="Model not loaded")
    X = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]])
    X_scaled = ml["scaler"].transform(X)
    model = ml["model"]
    pred = int(model.predict(X_scaled)[0])
    probs = model.predict_proba(X_scaled)[0].tolist()
    return Prediction(
        predicted_class=pred,
        predicted_label=ml["target_names"][pred],
        probabilities=[round(p, 4) for p in probs],
    )

@app.get("/version")
def version():
    return {"version": APP_VERSION}