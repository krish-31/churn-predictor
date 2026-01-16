import requests
import json

# Sample JSON for a 'Low Risk' user
sample_user = {
    "age": 35,
    "gender": "Male",
    "subscription_type": "Premium",
    "watch_hours": 120,
    "last_login_days": 2,
    "region": "North America",
    "device": "Smart TV",
    "monthly_fee": 15.99,
    "payment_method": "Credit Card",
    "number_of_profiles": 3,
    "avg_watch_time_per_day": 3.5,
    "favorite_genre": "Action"
}

try:
    response = requests.post(
        "http://localhost:8000/predict",
        json=sample_user,
        headers={"Content-Type": "application/json"}
    )
    
    print("Status Code:", response.status_code)
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
    
except requests.exceptions.ConnectionError:
    print("❌ Error: Cannot connect to the server. Make sure it's running on http://localhost:8000")
except Exception as e:
    print(f"❌ Error: {str(e)}")
