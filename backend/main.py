import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
import os
from dotenv import load_dotenv
import google.generativeai as genai
import time
import json
import plotly.express as px
import plotly.utils

# --- MERGE: Import Drift Detector ---
try:
    # Attempt to import from the local drift folder
    from drift.drift_detector import DriftDetector
except ImportError:
    print("⚠️ Warning: 'drift' module not found. Ensure 'drift' folder is in 'backend'.")
    DriftDetector = None

# --- CRITICAL FIX: Load .env from the CURRENT directory (backend) ---
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

app = FastAPI(title="OTT Churn Batch Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    # Only raise error if key is missing, but print help message first
    print("❌ ERROR: GOOGLE_API_KEY is missing. Check your .env file.")
    # We don't raise error here to allow server to start, but AI features will fail gracefully

# Configure AI
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
model = None

# --- AUTO-DETECT WORKING MODEL ---
def get_best_available_model():
    if not GOOGLE_API_KEY: return None
    try:
        print("🔍 Scanning for available Gemini models...")
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        
        priorities = ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro']
        for p in priorities:
            if p in available_models:
                print(f"✅ Found Optimal Model: {p}")
                return genai.GenerativeModel(p)
        
        if available_models:
            print(f"⚠️ Using Fallback Model: {available_models[0]}")
            return genai.GenerativeModel(available_models[0])
        return None
    except Exception as e:
        print(f"⚠️ Model Auto-Detect Failed: {e}")
        return None

model = get_best_available_model()

# --- MODEL LOADING ---
models = {}
MODEL_DIR = "models" 
MODEL_METRICS = {
    "xg": {"accuracy": 94.2, "precision": 92.1, "recall": 91.5, "f1": 91.8},
    "rf": {"accuracy": 89.5, "precision": 87.4, "recall": 86.2, "f1": 86.8},
    "lr": {"accuracy": 82.1, "precision": 79.5, "recall": 80.1, "f1": 79.8}
}

def load_local_models():
    try:
        # Ensure directory exists before loading
        if not os.path.exists(MODEL_DIR):
            print(f"⚠️ Warning: Model directory '{MODEL_DIR}' not found.")
            return

        models['xg'] = pickle.load(open(f"{MODEL_DIR}/xg_boosting_model.pkl", "rb"))
        models['rf'] = pickle.load(open(f"{MODEL_DIR}/random_forest_model.pkl", "rb"))
        models['lr'] = pickle.load(open(f"{MODEL_DIR}/logistic_regression_model.pkl", "rb"))
        print("✅ Models synchronized and loaded.")
    except Exception as e:
        print(f"❌ Error loading models: {e}.")

load_local_models()

# --- MERGE: DRIFT DETECTOR INITIALIZATION ---
drift_detector = None
baseline_df = None

try:
    base_path = os.path.join(os.path.dirname(__file__), "data", "baseline.csv")
    
    if os.path.exists(base_path):
        baseline_df = pd.read_csv(base_path)
        if DriftDetector:
            drift_detector = DriftDetector()
            print("✅ Drift Detector & Baseline Loaded Successfully.")
    else:
        # Create data directory if it doesn't exist
        os.makedirs(os.path.join(os.path.dirname(__file__), "data"), exist_ok=True)
        print(f"⚠️ Drift Warning: 'baseline.csv' not found at {base_path}")
        
except Exception as e:
    print(f"⚠️ Drift Initialization Failed: {e}")


# --- EXISTING FUNCTIONS ---

def calculate_drift_simple(df):
    """Simple drift calculation for the summary card"""
    baselines = {'watch_hours': 22.0, 'last_login_days': 4.0}
    drift_flags = []
    for feature, base_mean in baselines.items():
        if feature in df.columns:
            current_mean = df[feature].mean()
            shift = abs(current_mean - base_mean) / (base_mean + 1e-6)
            if shift > 0.4: drift_flags.append("High")
            elif shift > 0.2: drift_flags.append("Moderate")
    return "High" if "High" in drift_flags else "Moderate" if "Moderate" in drift_flags else "Low"

ai_call_counter = 0

def generate_smart_insight(row):
    global ai_call_counter
    risk = row['churn_probability']
    
    if risk > 75: fallback = "Critical: Reactivation Phone Call + 50% Win-back Offer"
    elif risk > 40: fallback = f"Medium: Send targeted email for {row['favorite_genre']} content."
    else: fallback = "Healthy: Include in monthly newsletter."

    # Safety check: Only run AI if High Risk, Counter < 5, and Model exists
    if risk > 75 and ai_call_counter < 5 and model:
        try:
            ai_call_counter += 1 
            prompt = (
                f"Customer Data: Risk {risk}%, Watch Hours {row['watch_hours']}, "
                f"Genre {row['favorite_genre']}, Inactive {row['last_login_days']} days. "
                f"Act as a retention expert. Write a strictly 1-sentence specific marketing action."
            )
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

# ==========================================
# 📊 NEW ENDPOINT: STATS / EDA (BLUE THEME)
# ==========================================
@app.post("/stats/analyze")
async def analyze_stats(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # Reset cursor just in case
        await file.seek(0)
        df = pd.read_csv(io.BytesIO(contents)) if file.filename.endswith('.csv') else pd.read_excel(io.BytesIO(contents))
        
        # 1. Dataset Overview
        summary = {
            'rows': df.shape[0],
            'cols': df.shape[1],
            'missing': int(df.isnull().sum().sum()),
            'duplicates': int(df.duplicated().sum()),
        }

        # 2. Identify Column Types
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        churn_col = next((c for c in df.columns if 'churn' in c.lower()), None)

        charts = {'categorical': [], 'numerical': [], 'correlation': None, 'bivariate': []}
        
        # Custom "Blue Shades" Palette for visual variety
        blue_gradients = ['#1E3A8A', '#1D4ED8', '#2563EB', '#3B82F6', '#60A5FA', '#93C5FD']

        # --- A. CATEGORICAL: Smart Selection (Donut vs Bar) ---
        for col in cat_cols:
            # SKIP ID COLUMNS
            if 'id' in col.lower() or 'email' in col.lower():
                continue

            unique_count = df[col].nunique()
            if unique_count < 15:
                counts = df[col].value_counts().reset_index()
                counts.columns = [col, 'Count']
                
                # If few categories, use a DONUT CHART (Cleaner look)
                if unique_count <= 5:
                    fig = px.pie(counts, names=col, values='Count', hole=0.6, 
                                 title=f"<b>{col.replace('_',' ').title()}</b> Composition",
                                 color_discrete_sequence=blue_gradients)
                    fig.update_traces(textinfo='percent+label', textfont_size=14)
                else:
                    # If many categories, use a BAR CHART
                    fig = px.bar(counts, x=col, y='Count', 
                                 title=f"<b>{col.replace('_',' ').title()}</b> Distribution",
                                 color='Count', # Gradient coloring based on count
                                 color_continuous_scale='Blues') 
                
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    font_color='#E0E7FF',
                    margin=dict(t=50, l=20, r=20, b=40),
                    showlegend=True,
                    coloraxis_showscale=False # Hide color bar for cleaner look
                )
                charts['categorical'].append(json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)))

        # --- B. NUMERICAL: Histograms with KDE look ---
        for i, col in enumerate(num_cols):
            # Skip ID columns or tiny variance columns
            if 'id' in col.lower() or df[col].nunique() < 5:
                continue

            # Alternate colors for visual separation
            chart_color = blue_gradients[i % len(blue_gradients)]

            fig = px.histogram(df, x=col, title=f"<b>{col.replace('_',' ').title()}</b> Spread",
                               color_discrete_sequence=[chart_color], nbins=40, opacity=0.8)
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                font_color='#E0E7FF',
                margin=dict(t=50, l=40, r=20, b=40),
                bargap=0.05
            )
            # Add a subtle grid
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
            
            charts['numerical'].append(json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)))

        # --- C. CORRELATION (HUGE MAP) ---
        if len(num_cols) > 1:
            # Filter IDs from correlation too
            valid_corr_cols = [c for c in num_cols if 'id' not in c.lower()]
            if len(valid_corr_cols) > 1:
                corr = df[valid_corr_cols].corr()
                fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto",
                                     color_continuous_scale='Blues',
                                     title="<b>Feature Correlation Heatmap</b>")
                fig_corr.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    font_color='#E0E7FF',
                    height=800,  # FORCE HUGE HEIGHT
                    margin=dict(t=60, l=40, r=40, b=40)
                )
                charts['correlation'] = json.loads(json.dumps(fig_corr, cls=plotly.utils.PlotlyJSONEncoder))

        # --- D. BIVARIATE (Box Plots for nice separation) ---
        if churn_col:
            for col in num_cols:
                if 'id' in col.lower() or col == churn_col: continue

                fig = px.box(df, x=churn_col, y=col, color=churn_col,
                             title=f"<b>{col.replace('_',' ').title()}</b> by Churn",
                             color_discrete_map={0: '#60A5FA', 1: '#1E40AF', 'No': '#60A5FA', 'Yes': '#1E40AF'})
                
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    font_color='#E0E7FF',
                    showlegend=False,
                    margin=dict(t=50, l=40, r=20, b=40)
                )
                charts['bivariate'].append(json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)))

        return {"summary": summary, "charts": charts}

    except Exception as e:
        print(f"Stats Analysis Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- MERGE: NEW DRIFT ENDPOINT ---
@app.post("/drift/analyze")
async def analyze_drift(file: UploadFile = File(...)):
    if not drift_detector or baseline_df is None:
        # Return a soft error so frontend doesn't crash, just shows empty drift
        print("Drift Analysis requested but detector is offline.")
        raise HTTPException(status_code=503, detail="Drift Detector not initialized.")
    
    try:
        contents = await file.read()
        recent_df = pd.read_csv(io.BytesIO(contents)) if file.filename.endswith('.csv') else pd.read_excel(io.BytesIO(contents))

        drift_result = drift_detector.detect_drift(
            baseline_df=baseline_df,
            recent_df=recent_df
        )
        return drift_result
        
    except Exception as e:
        print(f"Drift Analysis Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
        # Reset file cursor before reading again (important if file was read by drift endpoint)
        await file.seek(0)
        
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
        
        # Check if model is loaded before predicting
        if 'xg' not in models:
            raise HTTPException(status_code=500, detail="Prediction models not loaded. Check server logs.")

        raw_probs = models['xg'].predict_proba(features_df)[:, 1]

        # 3. Uncertainty Regularization
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
        for index, row in df.iterrows():
            if row['churn_probability'] > 75 and ai_call_counter < 5:
                df.at[index, 'insight'] = generate_smart_insight(row)
                time.sleep(2) # Safety delay for API limits
            else:
                df.at[index, 'insight'] = generate_smart_insight(row)

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
            "drift_level": calculate_drift_simple(df),
            "top_churn_genre": str(high_risk_df['favorite_genre'].mode()[0]) if not high_risk_df.empty else "N/A",
            "churner_avg_watch": round(float(high_risk_df['watch_hours'].mean()), 1) if not high_risk_df.empty else 0.0
        }

        results_list = []
        for _, row in df.iterrows():
            results_list.append({
                'customer_id': str(row['customer_id']),
                'churn_probability': float(row['churn_probability']),
                'risk_category': str(row['risk_category']),
                'insight': str(row.get('insight', 'No insight')),
                'persona': str(row['persona']),
                
                # Full details for Customer Modal
                'age': int(row.get('age', 0)),
                'gender': str(row.get('gender', 'N/A')),
                'location': str(row.get('region', 'Unknown')),
                'watch_hours': float(row.get('watch_hours', 0.0)),
                'last_login': int(row.get('last_login_days', 0)),
                'favorite_genre': str(row.get('favorite_genre', 'N/A')),
                'monthly_fee': float(row.get('monthly_fee', 0.0))
            })

        return {"summary": summary, "data": results_list}

    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)