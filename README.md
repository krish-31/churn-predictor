# 🎬 OTT Churn Prediction System

A production-ready machine learning solution that predicts customer churn for Over-The-Top (OTT) streaming platforms using XGBoost with real-time health monitoring and full-stack deployment.

![Status](https://img.shields.io/badge/status-production%20ready-brightgreen)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![React](https://img.shields.io/badge/React-18-blue)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Usage Examples](#usage-examples)
- [Deployment](#deployment)
- [Contributing](#contributing)

---

## 🎯 Overview

The OTT Churn Prediction System predicts which streaming service customers are likely to cancel their subscriptions. It leverages machine learning to identify at-risk customers, enabling proactive retention strategies.

**Key Capabilities:**
- **Batch Predictions**: Upload CSV/XLSX files with customer data
- **Real-time Health Monitoring**: Continuous backend availability tracking
- **Multi-format Support**: Seamless CSV and Excel file processing
- **Schema Validation**: Intelligent error detection with detailed feedback
- **Production Deployment**: Automated startup with concurrent backend/frontend execution

---

## ✨ Key Features

### 🤖 Machine Learning
- **XGBoost Primary Model** with Random Forest & Logistic Regression ensemble fallback
- **Feature Engineering**: 5 advanced features calculated on-the-fly
- **Intelligent Preprocessing**: NaN handling with domain-aware imputation strategies
- **Data Drift Detection**: Monitors model performance degradation over time

### 🌐 Frontend
- **Interactive Dashboard**: Real-time file upload and prediction results
- **Health Monitoring**: Live status indicator (Green/Red) for ML engine availability
- **Error Handling**: User-friendly alerts for validation failures
- **Responsive Design**: Tailwind CSS styling for seamless mobile/desktop experience

### ⚙️ Backend
- **FastAPI Framework**: High-performance async API
- **Multi-format Processing**: CSV and XLSX file support with pandas
- **Comprehensive Validation**: 13-column schema verification
- **Error Recovery**: Graceful handling with descriptive HTTP responses

### 🚀 DevOps
- **Automated Startup**: One-command initialization for entire stack
- **Port Management**: Automatic cleanup of port conflicts
- **Process Verification**: Health checks before completion
- **Concurrent Execution**: Parallel backend and frontend startup

---

## 💻 Tech Stack

### Backend
```
FastAPI 0.115.0        - Modern async web framework
Uvicorn 0.30.0         - ASGI server
XGBoost               - Primary ML model
Scikit-learn 1.6.1    - ML algorithms & preprocessing
Pandas 2.2.3          - Data manipulation
Pydantic 2.10.0       - Data validation
Python-dotenv 1.0.0   - Environment configuration
```

### Frontend
```
React 18              - UI framework
Vite                  - Build tool & dev server
Tailwind CSS          - Styling framework
JavaScript (JSX)      - Component development
```

### Infrastructure
```
Python 3.9+           - Backend runtime
Node.js 18+           - Frontend tooling
macOS/Linux/Windows   - Cross-platform support
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- npm or yarn

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/krish-31/churn-predictor.git
   cd churn-prediction-system
   ```

2. **Set up Python environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r backend/requirements.txt
   ```

3. **Set up frontend**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

### Run the Application

**Option 1: All-in-one startup (macOS/Linux)**
```bash
chmod +x run_app.sh
./run_app.sh
```

**Option 2: Manual startup**

Terminal 1 - Backend:
```bash
cd backend
uvicorn main:app --reload --port 8000
```

Terminal 2 - Frontend:
```bash
cd frontend
npm run dev
```

**Access the Application**
- 🌐 **Dashboard**: http://localhost:5173
- 📚 **API Docs**: http://localhost:8000/docs
- ❤️ **Health Check**: http://localhost:8000/

---

## 📁 Project Structure

```
churn-prediction-system/
├── backend/
│   ├── main.py                 # FastAPI application & routes
│   ├── requirements.txt         # Python dependencies
│   ├── start_backend.sh        # Backend startup script
│   ├── test_request.py         # API testing utilities
│   ├── data/
│   │   └── baseline.csv        # Reference dataset
│   ├── drift/
│   │   ├── drift_detector.py   # Data drift monitoring
│   │   └── drift_metrics.py    # Drift metrics calculations
│   └── models/                 # Trained ML models
├── frontend/
│   ├── src/
│   │   ├── App.jsx             # Main React component
│   │   ├── components/
│   │   │   ├── BackgroundWrapper.jsx
│   │   │   └── CustomerModal.jsx
│   │   ├── index.css           # Base styles
│   │   └── main.jsx            # React entry point
│   ├── index.html              # HTML template
│   ├── package.json            # npm dependencies
│   ├── vite.config.js          # Vite configuration
│   └── tailwind.config.js      # Tailwind CSS config
├── scripts/
│   └── train_local.py          # Model training script
├── Documents/
│   ├── DEPLOYMENT_GUIDE.md     # Detailed deployment instructions
│   ├── QUICK_START.md          # Quick reference guide
│   ├── PROJECT_COMPLETION_REPORT.md
│   └── SUBMISSION_GUIDE.md
├── run_app.sh                  # Master startup automation script
├── netflix_customer_churn.csv  # Sample dataset
└── README.md                   # This file
```

---

## 📚 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Health Check
```http
GET /
```
Returns backend status and system information.

**Response:**
```json
{
  "status": "ok",
  "message": "OTT Churn Prediction System is running",
  "timestamp": "2026-01-21T10:30:00"
}
```

#### 2. Predict Churn
```http
POST /predict
Content-Type: multipart/form-data
```

**Parameters:**
- `file`: CSV or XLSX file with customer data

**Required Columns:**
```
customer_id, age, gender, subscription_type, watch_hours,
last_login_days, region, device, monthly_fee,
number_of_profiles, avg_watch_time_per_day, favorite_genre, payment_method
```

**Response:**
```json
{
  "predictions": [
    {
      "customer_id": "C001",
      "churn_probability": 0.85,
      "churn_risk": "High",
      "recommendation": "Immediate intervention required"
    }
  ],
  "summary": {
    "total_customers": 100,
    "high_risk_count": 25,
    "medium_risk_count": 35,
    "low_risk_count": 40
  }
}
```

#### 3. Interactive API Docs
```
http://localhost:8000/docs  # Swagger UI
http://localhost:8000/redoc # ReDoc
```

---

## 💡 Usage Examples

### Example 1: Upload CSV File

1. Open Dashboard: http://localhost:5173
2. Click "Upload Dataset (CSV / XLSX)"
3. Select your CSV file with the 13 required columns
4. Click "Process Batch"
5. View predictions and risk analysis in real-time

### Example 2: Sample CSV Format

```csv
customer_id,age,gender,subscription_type,watch_hours,last_login_days,region,device,monthly_fee,number_of_profiles,avg_watch_time_per_day,favorite_genre,payment_method
C001,35,M,Premium,120,5,US,Smart TV,15.99,2,4.0,Action,Credit Card
C002,28,F,Basic,45,15,UK,Mobile,9.99,1,1.5,Romance,PayPal
C003,45,M,Standard,200,2,CA,Laptop,12.99,3,6.7,Documentary,Debit Card
```

### Example 3: Python API Client

```python
import requests

# Upload file for prediction
files = {'file': open('customers.csv', 'rb')}
response = requests.post('http://localhost:8000/predict', files=files)
predictions = response.json()

# Process results
for pred in predictions['predictions']:
    print(f"{pred['customer_id']}: {pred['churn_risk']} Risk")
```

---

## 🌐 Deployment

### Production Deployment (Render)

1. **Configure Environment Variables**
   ```
   PYTHON_VERSION=3.9
   NODE_VERSION=18
   ```

2. **Deploy Backend**
   - Push to GitHub
   - Connect Render to repository
   - Set build command: `pip install -r backend/requirements.txt`
   - Set start command: `uvicorn backend.main:app --host 0.0.0.0 --port 8000`

3. **Deploy Frontend**
   - Set build command: `npm run build` (in frontend directory)
   - Set start command: `npm run preview`

4. **Update API Endpoint**
   - Modify frontend to point to live backend URL

### Docker Deployment

```dockerfile
# Backend
FROM python:3.9
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt
COPY backend .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# Model Configuration
MODEL_PATH=./models/churn_model.pkl
CONFIDENCE_THRESHOLD=0.5

# Drift Detection
DRIFT_THRESHOLD=0.05
BASELINE_DATA=./data/baseline.csv

# API Configuration
API_DEBUG=false
CORS_ORIGINS=["http://localhost:5173", "https://yourdomain.com"]
```

---

## 📊 Model Details

### Input Features
- **Demographics**: age, gender, region
- **Subscription**: subscription_type, monthly_fee, number_of_profiles
- **Engagement**: watch_hours, avg_watch_time_per_day, device, favorite_genre
- **Activity**: last_login_days, payment_method

### Engineered Features
- `Cost_Per_Profile`: Monthly fee divided by number of profiles
- `Login_Frequency`: Inverse of days since last login
- `Loyalty_Score`: Watch hours × daily watch time ÷ last login days
- `Estimated_Tenure_Days`: Watch hours ÷ daily watch time
- `Watch_Intensity`: Daily watch time ÷ monthly fee

### Model Ensemble
- **Primary**: XGBoost Classifier
- **Fallback**: Random Forest + Logistic Regression
- **Output**: Churn probability (0-1) with risk classification (Low/Medium/High)

---

## 🧪 Testing

### Unit Tests
```bash
cd backend
pytest tests/ -v
```

### API Testing
```bash
cd backend
python test_request.py
```

### Manual API Testing
```bash
# Health check
curl http://localhost:8000/

# Prediction
curl -X POST -F "file=@customers.csv" http://localhost:8000/predict
```

---

## 📈 Monitoring & Maintenance

### Health Monitoring
- Frontend pings backend every 10 seconds
- Status indicator (Green/Red) shows availability
- Automatic reconnection on backend recovery

### Data Drift Detection
- Monitors feature distributions over time
- Alerts when drift exceeds configured threshold
- Stores metrics in `drift/drift_metrics.py`

### Log Files
```
Backend:  /tmp/backend.log
Frontend: /tmp/frontend.log
```

---

## 🚨 Troubleshooting

### Backend won't start
```bash
# Check port availability
lsof -i :8000

# Kill existing process
kill -9 <PID>
```

### Frontend connection issues
```bash
# Clear browser cache
# Check API endpoint in App.jsx
# Verify backend is running on port 8000
```

### Model not loading
```bash
# Ensure models/ directory exists
# Check model file permissions
# Verify model pickle format
```

---

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Support & Contact

- **Issues**: GitHub Issues
- **Documentation**: See `/Documents` folder
- **Quick Start**: Read [QUICK_START.md](Documents/QUICK_START.md)
- **Deployment**: Read [DEPLOYMENT_GUIDE.md](Documents/DEPLOYMENT_GUIDE.md)

---

## 🎉 Acknowledgments

Built with:
- FastAPI & Uvicorn for robust backend
- React & Vite for dynamic frontend
- XGBoost for powerful ML predictions
- Tailwind CSS for beautiful styling

---

**Last Updated:** January 21, 2026  
**Status:** ✅ Production Ready  
**Version:** 1.0.0

