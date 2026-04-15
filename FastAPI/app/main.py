from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import mlflow
import mlflow.tracking
import numpy as np
import logging
import pickle
import os
from pathlib import Path
from dotenv import load_dotenv

# Import our custom modules
from explainability import AMLExplainer
from shap_translator import translate_shap_for_llm
from llm_service import generate_investigation_summary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AML Triage API", version="1.0")

#global variables to hold loaded model artifacts
model_artifacts = None
explainer = None
feature_names = None

class TransactionData(BaseModel):
    features: Dict[str, Any]

class PredictionResponse(BaseModel):
    probability: float
    is_alert: bool
    alert_tier: Optional[str]=None
    threshold_used: float

class InvestigationResponse(BaseModel):
    shap_raw: List[Dict[str, Any]]
    evidence_list: str
    llm_summary: str

@app.on_event("startup")
def load_model_and_explainer():
    """
    fetches the production model from MLflow on startup.
    """
    global explainer, feature_names, model_artifacts
    
    # dagshub_token = os.getenv("DAGSHUB_TOKEN")
    # dagshub_username = os.getenv("DAGSHUB_USERNAME")
    # if not dagshub_token or not dagshub_username:
    #     raise ValueError("DAGSHUB_TOKEN OR DAGSHUB_USERNAME not set in .evn")
    
    # os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
    # os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    # # Set MLflow tracking URI to Dagshub
    # os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/virajdeshmukh080818/AML.mlflow"
    # mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    
    # #get run id from staging model
    # client = mlflow.tracking.MlflowClient(os.environ["MLFLOW_TRACKING_URI"])
    # mv = client.get_model_version_by_alias("AML_Laundering_Detector", "staging")
    # run_id = mv.run_id

    # # download artifacts
    # artifact_uri = f"runs:/{run_id}/model"
    # local_path = mlflow.artifacts.download_artifacts(artifact_uri)
    # pkl_file = list(Path(local_path).glob('*.pkl'))[0]
    pkl_file = Path(r"E:\AML\models\trained_model\LightGBM.pkl")

    if not pkl_file.exists():
        raise FileNotFoundError("model not found")
    with open(pkl_file, 'rb') as x:
        model_artifacts = pickle.load(x)
    
    # Extract feature names from model signature
    feature_names = model_artifacts["features"]
    #initialize explainer with the raw model
    explainer = AMLExplainer(model=model_artifacts['model'], feature_names=feature_names)
    logger.info("Model and SHAP Explainer loaded successfully.")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model_artifacts is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionData):
    """
    fast prediction endpoint. Does NOT call the LLM.
    """
    if not model_artifacts: raise HTTPException(status_code=503, detail="Model not loaded")

    # Format input
    X_input = np.array([[transaction.features.get(f, 0.0) for f in feature_names]], dtype=np.float32)

    raw_prob = model_artifacts["model"].predict_proba(X_input)[0][1]
    prob =model_artifacts['platt'].predict_proba([[raw_prob]])[0][1]
    
    #using your Best F1 threshold as the alert trigger
    threshold = model_artifacts["optimal_threshold"]
    emergency_thresh = 0.70

    if prob >= emergency_thresh:
        tier = "TIER_1_EMERGENCY"
        is_alert = True
    elif prob >= threshold:
        tier = "TIER_2_INVESTIGATE"
        is_alert = True
    else:
        tier = None
        is_alert = False

    return {
        "probability": float(prob),
        "is_alert": is_alert,
        "alert_tier": tier,
        "threshold_used": float(threshold),
        "emergency_threshold": float(emergency_thresh)
    }

@app.post("/investigate", response_model=InvestigationResponse)
def investigate_alert(transaction: TransactionData):
    """
    deep dive endpoint. Computes SHAP, translates it, and calls LLM.
    called by the frontend ONLY when an analyst clicks Investigate on an alert.
    """
    if not explainer: raise HTTPException(status_code=503, detail="Model not loaded yet")

    X_input = np.array([[transaction.features.get(f, 0.0) for f in feature_names]], dtype=np.float32)
    raw_prob = model_artifacts["model"].predict_proba(X_input)[0][1]
    prob = model_artifacts['platt'].predict_proba([[raw_prob]])[0][1]    

    tier_3_thesh = model_artifacts["optimal_threshold"]
    if prob < tier_3_thesh:
        raise HTTPException(
            status_code=400, detail=f"Transaction Probability ({prob:.3f}) is below the investigation thrshold ({tier_3_thesh})"
        )
    #get SHAP values
    shap_data = explainer.explain_transaction(transaction.features)
    human_readable_evidence = translate_shap_for_llm(shap_data, transaction.features)
    narrative = generate_investigation_summary(human_readable_evidence)
    
    return {
        "shap_raw": shap_data,             
        "evidence_list": human_readable_evidence, 
        "llm_summary": narrative  
    }