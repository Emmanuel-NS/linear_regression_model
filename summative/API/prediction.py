
import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Define the app
app = FastAPI(title="Pharma Sales Prediction API", version="1.0.0")

# CORS configuration
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
    "http://127.0.0.1:8000",
    "*" # Ideally restrict this for production, but allows mobile apps to connect during dev
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Paths to artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Models are in ../linear_regression/
# Note: Ensure these files exist or update path
MODEL_PATH = os.path.join(BASE_DIR, '../linear_regression/best_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, '../linear_regression/scaler.pkl')
DATA_PATH = os.path.join(BASE_DIR, '../linear_regression/salesdaily.pkl')

# Global variables for model and scaler
model = None
scaler = None

# Input Schema
class SalesInput(BaseModel):
    year: int = Field(..., ge=2000, le=2100, description="Year of sales (e.g. 2023)")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    weekday: int = Field(..., ge=0, le=6, description="Weekday (0=Monday, 6=Sunday)")

    class Config:
        json_schema_extra = {
            "example": {
                "year": 2023,
                "month": 8,
                "weekday": 2
            }
        }

@app.on_event("startup")
def load_artifacts():
    global model, scaler
    try:
        # Check if files exist
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"Model loaded from {MODEL_PATH}")
        else:
            print(f"Warning: Model not found at {MODEL_PATH}")
            
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print(f"Scaler loaded from {SCALER_PATH}")
        else:
            print(f"Warning: Scaler not found at {SCALER_PATH}")
    except Exception as e:
        print(f"Error loading artifacts: {e}")

def preprocess_input(input_data: SalesInput):
    # This function must match the training preprocessing exactly!
    # Based on notebook analysis:
    # Features: Year, Month, Weekday
    # OHE: Month (drop_first=True), Weekday (drop_first=True)
    # Scaling: StandardScaler
    
    # Create DataFrame
    df = pd.DataFrame([input_data.dict()])
    
    # Rename columns to match training expected inputs (capitalized)
    # Pydantic uses lowercase by default. DataFrame columns will be 'year', 'month', 'weekday'
    df.columns = ['Year', 'Month', 'Weekday']
    
    # Manual One-Hot Encoding to ensure all columns exist
    # Month_2 to Month_12 (assuming 1 is dropped)
    for m in range(2, 13):
        df[f'Month_{m}'] = (df['Month'] == m).astype(int)
        
    # Weekday_1 to Weekday_6 (assuming 0 is dropped)
    # Weekday mapping: 0=Mon, 1=Tue... 6=Sun
    # If notebook standard was 0-6, and drop_first=True (dropping 0), we need 1-6
    for w in range(1, 7):
        df[f'Weekday_{w}'] = (df['Weekday'] == w).astype(int)
        
    # Drop original categorical columns
    df = df.drop(['Month', 'Weekday'], axis=1)
    
    input_vector = df.values
    
    if scaler:
        # If scaler has feature names, reorder df to match
        if hasattr(scaler, 'feature_names_in_'):
            # Add missing columns with 0 if any (robustness)
            for col in scaler.feature_names_in_:
                if col not in df.columns:
                    df[col] = 0
            # Reorder
            df = df[scaler.feature_names_in_]
            input_vector = df.values
            
        input_vector = scaler.transform(input_vector)
        
    return input_vector

@app.post("/predict", tags=["Prediction"])
def predict_sales(input_data: SalesInput):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not scaler:
        raise HTTPException(status_code=503, detail="Scaler not loaded")
    
    try:
        processed_data = preprocess_input(input_data)
        prediction = model.predict(processed_data)
        return {
            "predicted_sales": float(prediction[0]),
            "input": input_data.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

def retrain_model_task(new_data_path: str = None):
    # Retraining logic
    try:
        print("Starting retraining...")
        # Load dataset
        path = new_data_path if new_data_path else DATA_PATH
        if path.endswith('.pkl'):
            df = pd.read_pickle(path)
        else:
            # Fallback for csv
            try:
                df = pd.read_csv(path)
            except:
                print("Could not read data file.")
                return

            
        # Basic preprocessing (simplified from notebook)
        # Assuming df has 'Year', 'Month', 'Weekday', 'M01AB'
        if 'datum' in df.columns: 
            df['datum'] = pd.to_datetime(df['datum'])
            df['Year'] = df['datum'].dt.year
            df['Month'] = df['datum'].dt.month
            df['Weekday'] = df['datum'].dt.weekday
            
        target = 'M01AB'
        features = ['Year', 'Month', 'Weekday']
        
        # Verify columns exist
        if not all(col in df.columns for col in features + [target]):
            print(f"Missing columns in new data. Required: {features + [target]}")
            return

        X = df[features]
        y = df[target]
        
        # OHE
        X = pd.get_dummies(X, columns=['Month', 'Weekday'], drop_first=True)
        
        # Split (minimal split for quick retrain check)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale
        new_scaler = StandardScaler()
        X_train_scaled = new_scaler.fit_transform(X_train)
        
        # Train (RandomForest as it was the best)
        # Using fewer estimators for speed in demo
        new_model = RandomForestRegressor(n_estimators=50, random_state=42)
        new_model.fit(X_train_scaled, y_train)
        
        # Save
        global model, scaler
        joblib.dump(new_model, MODEL_PATH)
        joblib.dump(new_scaler, SCALER_PATH)
        
        # Reload
        model = new_model
        scaler = new_scaler
        print("Retraining complete. Model updated.")
        
    except Exception as e:
        print(f"Retraining failed: {e}")

@app.post("/retrain", tags=["Training"])
async def trigger_retraining(background_tasks: BackgroundTasks):
    # Trigger retraining in background
    background_tasks.add_task(retrain_model_task)
    return {"message": "Retraining started in background"}

@app.get("/")
def root():
    return {"message": "Pharma Sales Prediction API is running. Go to /docs for Swagger UI."}
