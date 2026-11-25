"""FastAPI service for the trained Cleveland heart-disease classifier."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.features import FEATURE_DESCRIPTIONS, FEATURE_NAMES
from src.train import RANDOM_SEED

MODEL_PATH = Path(__file__).resolve().parent / "models" / "heart_model.joblib"


class FeatureInput(BaseModel):
    """Clinical predictors in the encoding used by the UCI Cleveland dataset."""

    model_config = ConfigDict(extra="forbid", strict=True)

    age: Annotated[int, Field(ge=18, le=100, description=FEATURE_DESCRIPTIONS["age"])]
    sex: Annotated[int, Field(ge=0, le=1, description=FEATURE_DESCRIPTIONS["sex"])]
    cp: Annotated[int, Field(ge=1, le=4, description=FEATURE_DESCRIPTIONS["cp"])]
    trestbps: Annotated[
        int, Field(ge=70, le=250, description=FEATURE_DESCRIPTIONS["trestbps"])
    ]
    chol: Annotated[int, Field(ge=50, le=700, description=FEATURE_DESCRIPTIONS["chol"])]
    fbs: Annotated[int, Field(ge=0, le=1, description=FEATURE_DESCRIPTIONS["fbs"])]
    restecg: Annotated[
        int, Field(ge=0, le=2, description=FEATURE_DESCRIPTIONS["restecg"])
    ]
    thalach: Annotated[
        int, Field(ge=60, le=250, description=FEATURE_DESCRIPTIONS["thalach"])
    ]
    exang: Annotated[int, Field(ge=0, le=1, description=FEATURE_DESCRIPTIONS["exang"])]
    oldpeak: Annotated[
        float, Field(ge=-2, le=10, description=FEATURE_DESCRIPTIONS["oldpeak"])
    ]
    slope: Annotated[int, Field(ge=1, le=3, description=FEATURE_DESCRIPTIONS["slope"])]
    ca: Annotated[
        float | None,
        Field(default=None, ge=0, le=3, description=FEATURE_DESCRIPTIONS["ca"]),
    ]
    thal: Annotated[
        float | None,
        Field(default=None, ge=3, le=7, description=FEATURE_DESCRIPTIONS["thal"]),
    ]

    @field_validator("ca")
    @classmethod
    def validate_ca_category(cls, value: float | None) -> float | None:
        if value is not None and value not in {0, 1, 2, 3}:
            raise ValueError("ca must be one of 0, 1, 2, or 3.")
        return value

    @field_validator("thal")
    @classmethod
    def validate_thal_category(cls, value: float | None) -> float | None:
        if value is not None and value not in {3, 6, 7}:
            raise ValueError("thal must be one of 3, 6, or 7.")
        return value


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_name: str


class ModelInfoResponse(BaseModel):
    feature_schema: list[dict[str, str]]
    random_seed: int
    metrics: dict[str, dict[str, dict[str, float]] | dict[str, float]]


class PredictionResponse(BaseModel):
    prediction: Literal[0, 1]
    label: str
    disease_probability: float = Field(ge=0, le=1)
    risk_tier: Literal["Low", "Moderate", "High"]
    model_name: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Train the model first with `python src/train.py`."
        )
    app.state.model_artifact = joblib.load(MODEL_PATH)
    yield


app = FastAPI(
    title="UCI Heart Disease Model API",
    description="Educational API for the UCI Cleveland Heart Disease dataset.",
    version="1.0.0",
    lifespan=lifespan,
)


def risk_tier(probability: float) -> Literal["Low", "Moderate", "High"]:
    if probability < 0.33:
        return "Low"
    if probability < 0.66:
        return "Moderate"
    return "High"


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    artifact = request.app.state.model_artifact
    return HealthResponse(status="ok", model_name=artifact["model_name"])


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    artifact = request.app.state.model_artifact
    return ModelInfoResponse(
        feature_schema=[
            {"name": name, "description": FEATURE_DESCRIPTIONS[name]}
            for name in FEATURE_NAMES
        ],
        random_seed=RANDOM_SEED,
        metrics=artifact["metrics"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(
    payload: FeatureInput, request: Request
) -> PredictionResponse:
    artifact = request.app.state.model_artifact
    pipeline = artifact["pipeline"]
    features = pd.DataFrame(
        [payload.model_dump()], columns=artifact["feature_names"]
    )
    prediction = 1 if pipeline.predict(features)[0] == 1 else 0
    positive_class_index = list(pipeline.classes_).index(1)
    probability = float(pipeline.predict_proba(features)[0][positive_class_index])
    return PredictionResponse(
        prediction=prediction,
        label="Heart disease detected" if prediction else "No heart disease detected",
        disease_probability=probability,
        risk_tier=risk_tier(probability),
        model_name=artifact["model_name"],
    )
