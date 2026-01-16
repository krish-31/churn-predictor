import pickle
import io
import os

import numpy as np
import pandas as pd

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from drift.drift_detector import DriftDetector

# ------------------------------------------------------------------
# FastAPI App
# ------------------------------------------------------------------
app = FastAPI(title="OTT Churn Batch Engine")

# ------------------------------------------------------------------
# CORS (Frontend Access)
# ------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Load Baseline Data for Drift Detection
# ------------------------------------------------------------------
BASELINE_DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "baseline.csv"
)

try:
    baseline_df = pd.read_csv(BASELINE_DATA_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load baseline dataset: {e}")

drift_detector = DriftDetector()

# ------------------------------------------------------------------
# Load Models
# ------------------------------------------------------------------
models = {}
MODEL_DIR = "models"

def load_local_models():
    try:
        models["xg"] = pickle.load(open(f"{MODEL_DIR}/xg_boosting_model.pkl", "rb"))
        models["rf"] = pickle.load(open(f"{MODEL_DIR}/random_forest_model.pkl", "rb"))
        models["lr"] = pickle.load(open(f"{MODEL_DIR}/logistic_regression_model.pkl", "rb"))
        print("✅ Models synchronized and loaded.")
    except Exception as e:
        print(f"❌ Error loading models: {e}")

load_local_models()

# ------------------------------------------------------------------
# Utility: Retention Strategy
# ------------------------------------------------------------------
def generate_retention_strategy(row):
    risk = row["churn_probability"]
    days_inactive = row["last_login_days"]

    if risk > 80:
        if days_inactive > 30:
            return "Critical: Reactivation Phone Call + 50% Win-back Offer"
        return "High Risk: Premium loyalty bundle (1 Month Free)"
    elif risk > 45:
        return f"Medium Risk: Targeted email for {row['favorite_genre']}"
    else:
        return "Healthy: Include in monthly newsletter"

# ------------------------------------------------------------------
# Health Check
# ------------------------------------------------------------------
@app.get("/")
def health():
    return {
        "status": "active",
        "loaded_models": list(models.keys())
    }

# ------------------------------------------------------------------
# Batch Prediction Endpoint
# ------------------------------------------------------------------
@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):

    if not (file.filename.endswith(".csv") or file.filename.endswith(".xlsx")):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid CSV or XLSX file."
        )

    REQUIRED_COLUMNS = [
        "customer_id", "age", "gender", "subscription_type", "watch_hours",
        "last_login_days", "region", "device", "monthly_fee",
        "number_of_profiles", "avg_watch_time_per_day",
        "favorite_genre", "payment_method"
    ]

    try:
        contents = await file.read()

        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))

        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns: {', '.join(missing_cols)}"
            )

        df = df.fillna({
            "avg_watch_time_per_day": 0,
            "watch_hours": 0,
            "last_login_days": 30,
            "monthly_fee": df["monthly_fee"].mean(),
            "number_of_profiles": 1
        })

        # Feature Engineering
        df["Cost_Per_Profile"] = df["monthly_fee"] / (df["number_of_profiles"] + 1e-6)
        df["Login_Frequency"] = 1 / (df["last_login_days"] + 1)
        df["Loyalty_Score"] = (
            df["watch_hours"] * df["avg_watch_time_per_day"]
        ) / (df["last_login_days"] + 1)
        df["Estimated_Tenure_Days"] = df["watch_hours"] / (df["avg_watch_time_per_day"] + 1e-6)
        df["Watch_Intensity"] = df["avg_watch_time_per_day"] / (df["monthly_fee"] + 1e-6)

        features_df = df.drop(columns=["customer_id", "churned"], errors="ignore")
        features_df = features_df.fillna(features_df.mean(numeric_only=True))

        probs = models["xg"].predict_proba(features_df)[:, 1]

        df["churn_probability"] = np.round(probs * 100, 2)
        df["risk_category"] = df["churn_probability"].apply(
            lambda x: "High" if x > 75 else "Medium" if x > 40 else "Low"
        )
        df["insight"] = df.apply(generate_retention_strategy, axis=1)

        summary = {
            "total_customers": len(df),
            "high_risk_count": int((df["risk_category"] == "High").sum()),
            "medium_risk_count": int((df["risk_category"] == "Medium").sum()),
            "low_risk_count": int((df["risk_category"] == "Low").sum()),
            "revenue_at_risk": round(
                float(df[df["risk_category"] == "High"]["monthly_fee"].sum()), 2
            ),
            "avg_risk_score": round(float(df["churn_probability"].mean()), 2)
        }

        results = df[[
            "customer_id", "churn_probability", "risk_category", "insight"
        ]].to_dict(orient="records")

        return {
            "summary": summary,
            "data": results
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

# ------------------------------------------------------------------
# Drift Detection Endpoint
# ------------------------------------------------------------------
@app.post("/drift/analyze")
async def analyze_drift(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        if file.filename.endswith(".csv"):
            recent_df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(".xlsx"):
            recent_df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(
                status_code=400,
                detail="Only CSV or XLSX files are supported for drift analysis"
            )

        drift_result = drift_detector.detect_drift(
            baseline_df=baseline_df,
            recent_df=recent_df
        )

        return drift_result

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Drift analysis failed: {str(e)}"
        )

# ------------------------------------------------------------------
# Run Server
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
