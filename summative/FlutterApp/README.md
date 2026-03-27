# Pharma Sales Predictor App

Flutter mobile client for the Pharma Sales Prediction project.

## What This App Does
- Takes user inputs: `Year`, `Month`, and `Weekday`.
- Calls the deployed FastAPI prediction endpoint.
- Displays predicted M01AB demand (rounded and exact value).

## API Endpoint
- Base URL: `https://linear-regression-model-q2c6.onrender.com`
- Swagger (manual testing): `https://linear-regression-model-q2c6.onrender.com/docs`

## Run Locally
```bash
flutter pub get
flutter run
```

## Build Release
```bash
flutter build apk --release
```
