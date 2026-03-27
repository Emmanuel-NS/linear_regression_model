# Pharma Sales Prediction (M01AB)

## Mission And Problem
Mission: optimize pharmaceutical inventory by predicting daily M01AB sales.

Problem: pharmacies face stockouts and overstock due to seasonal and weekday demand variation. This project predicts expected demand to support better replenishment planning.

## Dataset
Source: [Pharma Sales Data (Kaggle)](https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data)

Used file: `salesdaily.csv`.

Target: `M01AB` (anti-inflammatory and antirheumatic non-steroids).

Feature set used for prediction: `Year`, `Month`, `Weekday`.

## Deployed API
FastAPI deployment on Render:

- Base URL: https://linear-regression-model-q2c6.onrender.com
- Swagger UI: https://linear-regression-model-q2c6.onrender.com/docs

## Repository Structure
- `summative/API/` - FastAPI backend and model-serving endpoint.
- `summative/FlutterApp/` - Flutter mobile app client.
- `summative/linear_regression/` - training notebook and model assets.

## Run The Mobile App
1. Install Flutter SDK and connect a device/emulator.
2. Open terminal and run:

```bash
cd summative/FlutterApp
flutter pub get
flutter run
```

## Video Demo
Add your final demo link here:

- YouTube: https://youtu.be/NelnlThafII
