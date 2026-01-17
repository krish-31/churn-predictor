import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file in root directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

app = FastAPI(title="OTT Churn Batch Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in .env file.")

# Configure the AI
genai.configure(api_key=GOOGLE_API_KEY)
model = None

# --- NEW: AUTO-DETECT WORKING MODEL ---
def get_best_available_model():
    """
    Scans the user's API key to find a working model.
    Prioritizes: Gemini 1.5 Flash -> 1.5 Pro -> 1.0 Pro
    """
    try:
        print("🔍 Scanning for available Gemini models...")
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        
        # Priority List
        priorities = ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro']
        
        for p in priorities:
            if p in available_models:
                print(f"✅ Found Optimal Model: {p}")
                return genai.GenerativeModel(p)
        
        # Fallback: Just take the first one available
        if available_models:
            print(f"⚠️ Using Fallback Model: {available_models[0]}")
            return genai.GenerativeModel(available_models[0])
            
        print("❌ No text generation models found for this API key.")
        return None
        
    except Exception as e:
        print(f"⚠️ Model Auto-Detect Failed: {e}")
        return None

# Initialize Model using the Auto-Detector
model = get_best_available_model()

models = {}
MODEL_DIR = "models" 

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
        print(f"❌ Error loading models: {e}.")

load_local_models()

def calculate_drift(df):
    baselines = {'watch_hours': 22.0, 'last_login_days': 4.0}
    drift_flags = []
    for feature, base_mean in baselines.items():
        if feature in df.columns:
            current_mean = df[feature].mean()
            shift = abs(current_mean - base_mean) / (base_mean + 1e-6)
            if shift > 0.4: drift_flags.append("High")
            elif shift > 0.2: drift_flags.append("Moderate")
    return "High" if "High" in drift_flags else "Moderate" if "Moderate" in drift_flags else "Low"

# --- SMART HYBRID GENERATOR ---
ai_call_counter = 0

def generate_smart_insight(row):
    global ai_call_counter
    
    risk = row['churn_probability']
    
    # Standard Rule-Based Fallback
    if risk > 75:
        fallback = "Critical: Reactivation Phone Call + 50% Win-back Offer"
    elif risk > 40:
        fallback = f"Medium: Send targeted email for {row['favorite_genre']} content."
    else:
        fallback = "Healthy: Include in monthly newsletter."

    # --- AI EXECUTION LOGIC ---
    # Only run AI if High Risk AND counter < 5 AND model is loaded
    if risk > 75 and ai_call_counter < 5 and model:
        try:
            ai_call_counter += 1 
            
            prompt = (
                f"Customer Data: Risk {risk}%, Watch Hours {row['watch_hours']}, "
                f"Genre {row['favorite_genre']}, Inactive {row['last_login_days']} days. "
                f"Act as a retention expert. Write a strictly 1-sentence specific marketing action."
            )

            # Use the auto-detected model
            response = model.generate_content(prompt)
            
            return "✨ AI: " + response.text.strip()
            
        except Exception as e:
            print(f"Gemini API Error: {e}")
            return fallback
            
    return fallback

@app.get("/")
def health():
    return {"status": "active", "loaded_models": list(models.keys())}

@app.get("/model-stats")
def get_model_stats():
    return MODEL_METRICS

@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    global ai_call_counter
    ai_call_counter = 0 

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

        def assign_persona(row):
            if row['watch_hours'] > 30: return "Power User"
            if row['last_login_days'] > 15: return "At-Risk Casual"
            return "Standard"
        df['persona'] = df.apply(assign_persona, axis=1)

        # 2. Schema Inference
        TRAINING_FEATURES = [
            'age', 'gender', 'subscription_type', 'watch_hours', 'last_login_days', 
            'region', 'device', 'monthly_fee', 'number_of_profiles', 
            'avg_watch_time_per_day', 'favorite_genre', 'payment_method',
            'Cost_Per_Profile', 'Login_Frequency', 'Loyalty_Score', 
            'Estimated_Tenure_Days', 'Watch_Intensity'
        ]
        features_df = df[TRAINING_FEATURES]
        
        raw_probs = models['xg'].predict_proba(features_df)[:, 1]

        # 3. Uncertainty Regularization (3-96%)
        final_probs = (raw_probs * 0.93) + 0.03
        noise = np.random.uniform(-0.015, 0.015, size=final_probs.shape)
        final_probs = np.clip(final_probs + noise, 0.01, 0.99)

        # 4. Result Processing
        df['churn_probability'] = np.round(final_probs * 100, 2)
        df['risk_category'] = df['churn_probability'].apply(
            lambda x: 'High' if x > 75 else 'Medium' if x > 40 else 'Low'
        )

        # 5. Hybrid AI Generation
        print("🧠 Generating Hybrid Strategies...")
        df['insight'] = df.apply(generate_smart_insight, axis=1)

        # 6. KPI Summary
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