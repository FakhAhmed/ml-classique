import os
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "src", "xgb_churn_model.json")

app = FastAPI(title="Telecom Churn API")

# 1. Chargement du modèle
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

# NOUVEAU : On extrait la liste exacte des 30 colonnes attendues par le modèle
EXPECTED_FEATURES = model.get_booster().feature_names

class CustomerData(BaseModel):
    features: dict

@app.get("/")
def read_root():
    return {"status": "Online"}

@app.post("/predict")
def predict_churn(data: CustomerData):
    try:
        # 1. Création du tableau avec les données reçues (ex: 19 colonnes)
        df = pd.DataFrame([data.features])
        
        # 2. LE GILET DE SAUVETAGE : On force le tableau à avoir les 30 colonnes. 
        # Celles qui manquent seront remplies par des 0 automatiquement.
        df = df.reindex(columns=EXPECTED_FEATURES, fill_value=0)
        
        # 3. Prédiction (On convertit explicitement en float pour éviter les bugs JSON)
        probability = float(model.predict_proba(df)[0][1])
        prediction = 1 if probability > 0.50 else 0
        
        return {
            "prediction": prediction,
            "probability_churn_percent": round(probability * 100, 2),
            "risk_level": "🔴 Haut" if probability > 0.65 else ("🟠 Moyen" if probability > 0.35 else "🟢 Bas")
        }
    except Exception as e:
        # S'il y a une erreur, on la renvoie proprement en texte au lieu d'un crash 500
        return {"erreur_interne": str(e)}