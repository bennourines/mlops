# app.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator, ValidationInfo
import joblib
import pandas as pd
from typing import Dict, Optional
from pipelines.training_pipeline import retrain_model

# Variables globales pour le modèle et les données
model = None
feature_names = []
all_states = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle et les données au démarrage de l'API"""
    global model, feature_names, all_states
    try:
        model = joblib.load("churn_model.joblib")
        processed_data = joblib.load("processed_data.joblib")
        feature_names = processed_data["X_train"].columns.tolist()
        all_states = [col for col in feature_names if col.startswith("State_")]
    except Exception as e:
        raise RuntimeError(f"Échec du chargement: {str(e)}")
    yield


app = FastAPI(
    title="API de Prédiction de Désabonnement Clients",
    lifespan=lifespan,
    version="1.0.0",
)


class InputData(BaseModel):
    state: str
    account_length: float
    area_code: float
    international_plan: str
    voice_mail_plan: str
    number_vmail_messages: float
    total_day_minutes: float
    total_day_calls: float
    total_eve_minutes: float
    total_eve_calls: float
    total_night_minutes: float
    total_night_calls: float
    total_intl_minutes: float
    total_intl_calls: float
    customer_service_calls: float

    @field_validator("international_plan", "voice_mail_plan")
    @classmethod
    def validate_yes_no(cls, v: str) -> str:
        if v.lower() not in ["yes", "no"]:
            raise ValueError('Doit être "yes" ou "no"')
        return v.lower()

    @field_validator(
        "account_length",
        "area_code",
        "number_vmail_messages",
        "total_day_minutes",
        "total_day_calls",
        "total_eve_minutes",
        "total_eve_calls",
        "total_night_minutes",
        "total_night_calls",
        "total_intl_minutes",
        "total_intl_calls",
        "customer_service_calls",
    )
    @classmethod
    def validate_non_negative(cls, v: float, info: ValidationInfo) -> float:
        if v < 0:
            raise ValueError(f"{info.field_name} doit être ≥ 0")
        return v


class PredictionResponse(BaseModel):
    prediction: str


class Hyperparameters(BaseModel):
    n_estimators: Optional[int] = 100
    max_depth: Optional[int] = None
    min_samples_split: Optional[int] = 2


def preprocess_input(data: InputData) -> pd.DataFrame:
    """Prépare les données d'entrée pour la prédiction"""
    input_dict = data.dict()

    # Conversion des valeurs catégorielles
    input_dict["international_plan"] = (
        1 if input_dict["international_plan"] == "yes" else 0
    )
    input_dict["voice_mail_plan"] = 1 if input_dict["voice_mail_plan"] == "yes" else 0

    # Encodage one-hot de l'état
    state_encoded = {state: 0 for state in all_states}
    state_feature_name = f"State_{input_dict['state']}"
    if state_feature_name in state_encoded:
        state_encoded[state_feature_name] = 1

    # Création du vecteur de caractéristiques
    numeric_features = [
        "account_length",
        "area_code",
        "international_plan",
        "voice_mail_plan",
        "number_vmail_messages",
        "total_day_minutes",
        "total_day_calls",
        "total_eve_minutes",
        "total_eve_calls",
        "total_night_minutes",
        "total_night_calls",
        "total_intl_minutes",
        "total_intl_calls",
        "customer_service_calls",
    ]

    return pd.DataFrame(
        [
            [input_dict[feat] for feat in numeric_features]
            + list(state_encoded.values())
        ],
        columns=feature_names,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: InputData):
    if model is None:
        raise HTTPException(503, "Modèle non chargé")

    try:  # <-- Début du bloc try BIEN ALIGNÉ
        X = preprocess_input(data)
        prediction = model.predict(X)[0]
        return {
            "prediction": (
                "Customer Will Churn" if prediction == 1 else "Customer Will Not Churn"
            )
        }
    except Exception as e:  # <-- Même niveau d'indentation que try
        raise HTTPException(500, f"Erreur de prédiction: {str(e)}")


@app.post("/retrain")
async def retrain(hparams: Hyperparameters):
    global model
    try:
        model = retrain_model(
            n_estimators=hparams.n_estimators,
            max_depth=hparams.max_depth,
            min_samples_split=hparams.min_samples_split,
        )
        return {"status": "success", "message": "Réentraînement réussi"}
    except Exception as e:
        raise HTTPException(500, f"Échec : {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
