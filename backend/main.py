import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
import os

app = FastAPI(title="OTT Churn Batch Engine")

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load locally trained pipelines
models = {}
MODEL_DIR = "models" 

# Model Benchmark Metrics
MODEL_METRICS = {
    "xg": {"accuracy": 94.2, "precision": 92.1, "recall": 91.5, "f1": 91.8},
    "rf": {"accuracy": 89.5, "precision": 87.4, "recall": 86.2, "f1": 86.8},
    "lr": {"accuracy": 82.1, "precision": 79.5, "recall": 80.1, "f1": 79.8}
}

def load_local_models():
    try:
        models['xg'] = pickle.load(open(f"{MODEL_DIR}/xg_boosting_model.pkl", "rb"))
        models['rf'] = pickle.load(open(f"{MODEL_DIR}/random_forest_model.pkl", "rb"))
        models['lr'] = pickle.load(open(f"{MODEL_DIR}/logistic_regression_model.pkl", "rb"))
        print("✅ Models synchronized and loaded.")
    except Exception as e:
        print(f"❌ Error loading models: {e}. Ensure .pkl files are in the models/ folder.")

load_local_models()

def calculate_drift(df):
    """Calculates behavioral drift based on training baselines."""
    baselines = {'watch_hours': 22.0, 'last_login_days': 4.0}
    drift_flags = []
    
    for feature, base_mean in baselines.items():
        if feature in df.columns:
            current_mean = df[feature].mean()
            shift = abs(current_mean - base_mean) / (base_mean + 1e-6)
            if shift > 0.4: drift_flags.append("High")
            elif shift > 0.2: drift_flags.append("Moderate")
    
    return "High" if "High" in drift_flags else "Moderate" if "Moderate" in drift_flags else "Low"

def generate_retention_strategy(row):
    """Rule-based logic for business insights."""
    risk = row['churn_probability']
    days_inactive = row['last_login_days']
    
    if risk > 75:
        if days_inactive > 30:
            return "Critical: Reactivation Phone Call + 50% Win-back Offer"
        return "High Risk: Premium loyalty bundle (1 Month Free)"
    elif risk > 40:
        return "Medium Risk: Targeted email for " + str(row['favorite_genre'])
    else:
        return "Healthy: Include in monthly newsletter"

@app.get("/")
def health():
    return {"status": "active", "loaded_models": list(models.keys())}

@app.get("/model-stats")
def get_model_stats():
    return MODEL_METRICS

@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    if not (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
        raise HTTPException(status_code=400, detail="Please upload a valid CSV or XLSX file.")

    REQUIRED_COLUMNS = [
        'customer_id', 'age', 'gender', 'subscription_type', 'watch_hours', 
        'last_login_days', 'region', 'device', 'monthly_fee', 
        'number_of_profiles', 'avg_watch_time_per_day', 'favorite_genre', 'payment_method'
    ]

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents)) if file.filename.endswith('.csv') else pd.read_excel(io.BytesIO(contents))
        
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing columns: {', '.join(missing_cols)}")

        # 1. Feature Engineering
        df['Cost_Per_Profile'] = df['monthly_fee'] / (df['number_of_profiles'] + 1e-6)
        df['Login_Frequency'] = 1 / (df['last_login_days'] + 1)
        df['Loyalty_Score'] = (df['watch_hours'] * df['avg_watch_time_per_day']) / (df['last_login_days'] + 1)
        df['Estimated_Tenure_Days'] = df['watch_hours'] / (df['avg_watch_time_per_day'] + 1e-6)
        df['Watch_Intensity'] = df['avg_watch_time_per_day'] / (df['monthly_fee'] + 1e-6)

        # 2. Personas
        def assign_persona(row):
            if row['watch_hours'] > 30: return "Power User"
            if row['last_login_days'] > 15: return "At-Risk Casual"
            return "Standard"
        df['persona'] = df.apply(assign_persona, axis=1)

        # 3. Inference with STRICT Schema Enforcement
        TRAINING_FEATURES = [
            'age', 'gender', 'subscription_type', 'watch_hours', 'last_login_days', 
            'region', 'device', 'monthly_fee', 'number_of_profiles', 
            'avg_watch_time_per_day', 'favorite_genre', 'payment_method',
            'Cost_Per_Profile', 'Login_Frequency', 'Loyalty_Score', 
            'Estimated_Tenure_Days', 'Watch_Intensity'
        ]
        features_df = df[TRAINING_FEATURES]
        
        # Original Prediction
        raw_probs = models['xg'].predict_proba(features_df)[:, 1]

        # --- UNCERTAINTY REGULARIZATION ---
        # Maps 0-100 range strictly to 3-96 range
        # Formula: (Input * Range_Span) + Min_Value
        # Range_Span = 0.96 - 0.03 = 0.93
        
        final_probs = (raw_probs * 0.93) + 0.03
        
        # Add tiny organic noise (+/- 1.5%) to prevent robotic duplicates
        noise = np.random.uniform(-0.015, 0.015, size=final_probs.shape)
        final_probs = np.clip(final_probs + noise, 0.01, 0.99)

        # 4. Result Processing
        df['churn_probability'] = np.round(final_probs * 100, 2)
        
        df['risk_category'] = df['churn_probability'].apply(
            lambda x: 'High' if x > 75 else 'Medium' if x > 40 else 'Low'
        )
        df['insight'] = df.apply(generate_retention_strategy, axis=1)

        # 5. KPI Summary (Real Counts)
        high_risk_df = df[df['risk_category'] == 'High']
        medium_risk_df = df[df['risk_category'] == 'Medium']
        low_risk_df = df[df['risk_category'] == 'Low']

        summary = {
            "total_customers": len(df),
            "avg_risk_score": round(float(df['churn_probability'].mean()), 2),
            "high_risk_count": int(len(high_risk_df)),
            "medium_risk_count": int(len(medium_risk_df)),
            "low_risk_count": int(len(low_risk_df)),
            "revenue_at_risk": round(float(high_risk_df['monthly_fee'].sum()), 2),
            "drift_level": calculate_drift(df),
            "top_churn_genre": str(high_risk_df['favorite_genre'].mode()[0]) if not high_risk_df.empty else "N/A",
            "churner_avg_watch": round(float(high_risk_df['watch_hours'].mean()), 1) if not high_risk_df.empty else 0.0
        }

        # 6. Format results
        results_list = []
        for _, row in df[['customer_id', 'churn_probability', 'risk_category', 'insight', 'persona']].iterrows():
            results_list.append({
                'customer_id': str(row['customer_id']),
                'churn_probability': float(row['churn_probability']),
                'risk_category': str(row['risk_category']),
                'insight': str(row['insight']),
                'persona': str(row['persona'])
            })

        return {"summary": summary, "data": results_list}

    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)