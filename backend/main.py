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

def load_local_models():
    try:
        models['xg'] = pickle.load(open(f"{MODEL_DIR}/xg_boosting_model.pkl", "rb"))
        models['rf'] = pickle.load(open(f"{MODEL_DIR}/random_forest_model.pkl", "rb"))
        models['lr'] = pickle.load(open(f"{MODEL_DIR}/logistic_regression_model.pkl", "rb"))
        print("✅ Models synchronized and loaded.")
    except Exception as e:
        print(f"❌ Error loading models: {e}. Run scripts/train_local.py first.")

load_local_models()

def generate_retention_strategy(row):
    """Rule-based logic for business insights."""
    risk = row['churn_probability']
    days_inactive = row['last_login_days']
    
    if risk > 80:
        if days_inactive > 30:
            return "Critical: Reactivation Phone Call + 50% Win-back Offer"
        return "High Risk: Premium loyalty bundle (1 Month Free)"
    elif risk > 45:
        return "Medium Risk: Targeted email for " + str(row['favorite_genre'])
    else:
        return "Healthy: Include in monthly newsletter"

@app.get("/")
def health():
    return {"status": "active", "loaded_models": list(models.keys())}

@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    # Update validation to allow both .csv and .xlsx
    if not (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
        raise HTTPException(status_code=400, detail="Please upload a valid CSV or XLSX file.")

    REQUIRED_COLUMNS = [
        'customer_id', 'age', 'gender', 'subscription_type', 'watch_hours', 
        'last_login_days', 'region', 'device', 'monthly_fee', 
        'number_of_profiles', 'avg_watch_time_per_day', 'favorite_genre', 'payment_method'
    ]

    try:
        contents = await file.read()
        
        # Determine file type and read into DataFrame
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            # Requires 'openpyxl' to be installed
            df = pd.read_excel(io.BytesIO(contents))
        
        # --- SCHEMA VALIDATION ---
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            error_msg = f"Invalid Dataset! Missing required columns: {', '.join(missing_cols)}"
            print(f"⚠️ Validation Failed: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)

        # 1. Fill NaN values with sensible defaults
        df = df.fillna({
            'avg_watch_time_per_day': 0,
            'watch_hours': 0,
            'last_login_days': 30,
            'monthly_fee': df['monthly_fee'].mean(),
            'number_of_profiles': 1
        })

        # 2. Automated Feature Engineering (Python Safeguard)
        df['Cost_Per_Profile'] = df['monthly_fee'] / (df['number_of_profiles'] + 1e-6)
        df['Login_Frequency'] = 1 / (df['last_login_days'] + 1)
        df['Loyalty_Score'] = (df['watch_hours'] * df['avg_watch_time_per_day']) / (df['last_login_days'] + 1)
        df['Estimated_Tenure_Days'] = df['watch_hours'] / (df['avg_watch_time_per_day'] + 1e-6)
        df['Watch_Intensity'] = df['avg_watch_time_per_day'] / (df['monthly_fee'] + 1e-6)

        # 3. Prepare data for Inference
        features_df = df.drop(columns=['customer_id', 'churned'], errors='ignore')
        features_df = features_df.fillna(features_df.mean(numeric_only=True))

        # 4. Batch Inference (XGBoost Primary)
        probs = models['xg'].predict_proba(features_df)[:, 1]
        
        # 5. Process Results
        df['churn_probability'] = np.round(probs * 100, 2)
        df['risk_category'] = df['churn_probability'].apply(
            lambda x: 'High' if x > 75 else 'Medium' if x > 40 else 'Low'
        )
        df['insight'] = df.apply(generate_retention_strategy, axis=1)

        # --- ADVANCED DATA SCIENCE METRICS START ---
        high_risk_df = df[df['risk_category'] == 'High']
        risk_counts = df['risk_category'].value_counts().to_dict()
        
        # Determine top genre among high-risk users
        top_churn_genre = high_risk_df['favorite_genre'].mode()[0] if not high_risk_df.empty else "None"
        
        # Calculate average engagement for churners
        churner_avg_watch = float(high_risk_df['watch_hours'].mean()) if not high_risk_df.empty else 0.0
        
        # Summary calculations with NaN handling
        high_risk_revenue = high_risk_df['monthly_fee'].sum()
        avg_risk = df['churn_probability'].mean()
        
        if np.isnan(high_risk_revenue): high_risk_revenue = 0
        if np.isnan(avg_risk): avg_risk = 0
            
        summary = {
            "total_customers": len(df),
            "high_risk_count": int(risk_counts.get('High', 0)),
            "medium_risk_count": int(risk_counts.get('Medium', 0)),
            "low_risk_count": int(risk_counts.get('Low', 0)),
            "revenue_at_risk": round(float(high_risk_revenue), 2),
            "avg_risk_score": round(float(avg_risk), 2),
            "top_churn_genre": str(top_churn_genre),
            "churner_avg_watch": round(float(churner_avg_watch), 1)
        }
        # --- ADVANCED DATA SCIENCE METRICS END ---

        # Convert results to JSON-serializable format
        results_list = []
        for _, row in df[['customer_id', 'churn_probability', 'risk_category', 'insight']].iterrows():
            churn_prob = float(row['churn_probability'])
            if np.isnan(churn_prob):
                churn_prob = 0.0
            results_list.append({
                'customer_id': str(row['customer_id']),
                'churn_probability': churn_prob,
                'risk_category': str(row['risk_category']),
                'insight': str(row['insight'])
            })

        return {"summary": summary, "data": results_list}

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Processing Error: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error. Check file structure.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)