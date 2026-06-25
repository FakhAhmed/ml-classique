import os
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel
from google.cloud import storage, aiplatform # <-- NOUVEL IMPORT ICI

app = FastAPI(title="Telecom Churn API")

# --- NOUVELLE LOGIQUE : INTERROGATION DE VERTEX AI ---
PROJECT_ID = "ml-classique-churn"
REGION = "europe-west9"
MODEL_NAME = "churn-prediction-model"
LOCAL_MODEL_PATH = "/tmp/model.bst"

print("🔍 Interrogation de Vertex AI...")
aiplatform.init(project=PROJECT_ID, location=REGION)

# On demande à Vertex AI de trouver le modèle par son nom
models = aiplatform.Model.list(filter=f"display_name={MODEL_NAME}")
if not models:
    raise Exception(f"Modèle {MODEL_NAME} introuvable dans Vertex AI !")

latest_model = models[0]
model_uri = latest_model.uri  # ex: gs://modeles-ia-ml-classique-churn/v1/
print(f"📍 Version officielle trouvée dans le registre : {model_uri}")

# Extraction du nom du bucket et du chemin exact
bucket_name = model_uri.split("/")[2]
blob_prefix = model_uri.replace(f"gs://{bucket_name}/", "")
blob_path = f"{blob_prefix}model.bst"

print("⏳ Téléchargement depuis le coffre-fort...")
client = storage.Client()
bucket = client.bucket(bucket_name)
blob = bucket.blob(blob_path)
blob.download_to_filename(LOCAL_MODEL_PATH)
print("✅ Modèle téléchargé avec succès et prêt pour l'inférence !")

# -----------------------------------------------------

# On charge le modèle depuis le fichier temporaire
model = xgb.XGBClassifier()
model.load_model(LOCAL_MODEL_PATH)

# On extrait la liste exacte des 30 colonnes attendues par le modèle
EXPECTED_FEATURES = model.get_booster().feature_names

class CustomerData(BaseModel):
    features: dict

@app.get("/")
def read_root():
    return {"status": "Online"}

@app.post("/predict")
def predict_churn(data: CustomerData):
    try:
        df = pd.DataFrame([data.features])
        df = df.reindex(columns=EXPECTED_FEATURES, fill_value=0)
        
        probability = float(model.predict_proba(df)[0][1])
        prediction = 1 if probability > 0.50 else 0
        
        return {
            "prediction": prediction,
            "probability_churn_percent": round(probability * 100, 2),
            "risk_level": "🔴 Haut" if probability > 0.65 else ("🟠 Moyen" if probability > 0.35 else "🟢 Bas")
        }
    except Exception as e:
        return {"erreur_interne": str(e)}