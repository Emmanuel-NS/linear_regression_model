# Pharma Sales Prediction (M01AB)

## Mission & Problem
**Mission:** Optimize pharmaceutical inventory by predicting daily sales volume of Anti-inflammatory drugs (M01AB) using Multivariate Linear Regression.
**Problem:** Pharmacies face stockouts and overstock due to irregular demand patterns driven by seasonality and specific days of the week, leading to financial loss and poor patient service.

## Dataset
**Source:** [Pharma Sales Data (Kaggle)](https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data)
**Description:** The dataset contains 6 years of sales data. We utilized the `salesdaily.csv` file, specifically the `M01AB` column (Anti-inflammatory and Antirheumatic non-steroids), analyzing 2,106 daily records to engineer temporal features (Year, Month, Weekday).

## API Deployment
The regression model is deployed using FastAPI on Render.
- **Base URL:** https://linear-regression-model-q2c6.onrender.com
- **Swagger UI (Test Prediction):** https://linear-regression-model-q2c6.onrender.com/docs

## Video Demo
[Insert YouTube Video Link Here]

## How to Run the Mobile App
1. **Prerequisites:** Ensure Flutter SDK is installed and a device/emulator is connected.
2. **Navigate:** Open a terminal and move to the app directory:
   ```bash
   cd summative/FlutterApp
   ```
3. **Install Dependencies:**
   ```bash
   flutter pub get
   ```
4. **Run App:**
   ```bash
   flutter run
   ```
