from fastapi import FastAPI, Request
from .models import PredictResponse, PredictRequest
from .logger_config import setup_logger
from .ml_loader import load_cross_encoder
from .config import config
from contextlib import asynccontextmanager
import math

logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    app.state.model = load_cross_encoder(config.MODEL_PATH, config.DEVICE)
    yield
    logger.info("Shutting down...")
    app.state.model = None


app = FastAPI(title="Text Similarity Service", root_path="/api/v1", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "code": 200, "model_load": app.state.model is not None}


@app.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest, request: Request):
    model = request.app.state.model

    logits = model.predict([[body.text_a, body.text_b]])[0]
    probability = 1 / (1 + math.exp(-logits))

    label = 1 if probability >= config.THRESHOLD else 0

    verdict = "Same" if label == 1 else "not same"
    logger.info(f"Prediction: {probability:.3f} -> {verdict}")
    return PredictResponse(
        text_a=body.text_a,
        text_b=body.text_b,
        probability=probability,
        label=label,
        verdict=verdict,
    )
