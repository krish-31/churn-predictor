import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def train_and_save_models():
    # Setup paths relative to project root
    data_path = 'netflix_customer_churn.csv'
    model_dir = 'backend/models'
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found in the root directory.")
        return

    # Load Data
    print("📂 Loading dataset...")
    df = pd.read_csv(data_path)

    # Feature Engineering (Must match inference logic)
    print("🛠️ Engineering features...")
    df['Cost_Per_Profile'] = df['monthly_fee'] / df['number_of_profiles']
    df['Login_Frequency'] = 1 / (df['last_login_days'] + 1)
    df['Loyalty_Score'] = (df['watch_hours'] * df['avg_watch_time_per_day']) / (df['last_login_days'] + 1)
    df['Estimated_Tenure_Days'] = df['watch_hours'] / (df['avg_watch_time_per_day'] + 1e-6)
    df['Watch_Intensity'] = df['avg_watch_time_per_day'] / df['monthly_fee']

    # Define Feature Sets
    numeric_features = [
        'age', 'watch_hours', 'last_login_days', 'monthly_fee', 
        'number_of_profiles', 'avg_watch_time_per_day', 
        'Cost_Per_Profile', 'Login_Frequency', 'Loyalty_Score', 
        'Estimated_Tenure_Days', 'Watch_Intensity'
    ]
    categorical_features = ['gender', 'subscription_type', 'region', 'device', 'payment_method', 'favorite_genre']
    
    X = df[numeric_features + categorical_features]
    y = df['churned']

    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Model Dictionary with YOUR specific filenames
    models_to_train = {
        "logistic_regression_model": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest_model": RandomForestClassifier(n_estimators=100, random_state=42),
        "xg_boosting_model": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\n🚀 Training models locally...")
    
    for name, model_instance in models_to_train.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model_instance)
        ])
        
        pipeline.fit(X_train, y_train)
        accuracy = pipeline.score(X_test, y_test)
        
        save_path = os.path.join(model_dir, f"{name}.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(pipeline, f)
        print(f"✅ Saved: {save_path} (Acc: {accuracy:.2%})")

    print("\n✨ Local sync complete. Backend is ready to start.")

if __name__ == "__main__":
    train_and_save_models()