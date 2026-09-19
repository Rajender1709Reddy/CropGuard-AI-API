from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from pathlib import Path


# CropGuard AI - Crop Yield Prediction API

app = FastAPI(
    title="CropGuard AI - Crop Yield Prediction API",
    description="FastAPI service for pre-cultivation crop yield prediction.",
    version="1.0.0"
)


# File paths

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "cropguard_yield_model.pkl"
DATA_PATH = BASE_DIR / "Final_dataset_clean.csv"


# Load trained model

model = None

try:
    model = joblib.load(MODEL_PATH)
    print("Yield model loaded successfully.")
except Exception as e:
    print(f"Error loading yield model: {e}")


# Load historical agriculture dataset

df = None

try:
    df = pd.read_csv(DATA_PATH)

    # Create year_start from agri_year if needed
    if "year_start" not in df.columns:
        if "agri_year" not in df.columns:
            raise ValueError(
                "Dataset must contain either 'year_start' or 'agri_year'."
            )

        df["year_start"] = (
            df["agri_year"]
            .astype(str)
            .str.extract(r"(\d{4})")[0]
        )

        df["year_start"] = pd.to_numeric(
            df["year_start"],
            errors="coerce"
        )

    # Normalize categorical columns to match model training
    required_text_columns = [
        "district_name",
        "crop_name",
        "crop_type",
        "season"
    ]

    for col in required_text_columns:
        if col not in df.columns:
            raise ValueError(
                f"Required dataset column '{col}' is missing."
            )

        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    # Check required model/backend columns
    required_columns = [
        "actual_rainfall",
        "normal_rainfall",
        "rainfall_deviation",
        "total_irrigated_area"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required dataset columns: {missing_columns}"
        )

    print("Agriculture dataset loaded successfully.")
    print("Dataset shape:", df.shape)

except Exception as e:
    print(f"Error loading agriculture dataset: {e}")


# Input schema

class YieldPredictionInput(BaseModel):
    district_name: str = Field(
        ...,
        description="District name, e.g. Nalgonda"
    )

    crop_name: str = Field(
        ...,
        description="Crop name, e.g. Rice"
    )

    season: str = Field(
        ...,
        description="Season, e.g. Kharif"
    )

    area: float = Field(
        ...,
        gt=0,
        description="Farmer cultivated area in hectares"
    )


# Home endpoint

@app.get("/")
def home():
    return {
        "message": "CropGuard AI Yield Prediction API is running",
        "status": "success",
        "model": "Random Forest Regression",
        "docs": "/docs"
    }


# Health endpoint

@app.get("/health")
def health():

    model_loaded = model is not None
    dataset_loaded = df is not None

    if not model_loaded or not dataset_loaded:
        return {
            "status": "unhealthy",
            "model_loaded": model_loaded,
            "dataset_loaded": dataset_loaded
        }

    return {
        "status": "healthy",
        "model_loaded": True,
        "dataset_loaded": True
    }


# Prediction endpoint

@app.post("/predict")
def predict_yield(data: YieldPredictionInput):

    # Check model
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Yield prediction model is not loaded."
        )

    # Check dataset
    if df is None:
        raise HTTPException(
            status_code=500,
            detail="Agriculture dataset is not loaded."
        )

    # Clean user inputs

    district = data.district_name.strip().lower()
    crop = data.crop_name.strip().lower()
    season = data.season.strip().lower()

    farmer_area = float(data.area)

    # Find historical records for selected combination

    matching_data = df[
        (df["district_name"] == district)
        & (df["crop_name"] == crop)
        & (df["season"] == season)
    ].copy()

    if matching_data.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                "No historical data is available for the selected "
                "District, Crop and Season."
            )
        )

    # Select latest available historical record

    matching_data = matching_data.sort_values(
        "year_start",
        na_position="first"
    )

    selected_data = matching_data.iloc[-1]

    # Get backend features from latest historical record

    crop_type = selected_data["crop_type"]
    actual_rainfall = selected_data["actual_rainfall"]
    normal_rainfall = selected_data["normal_rainfall"]
    rainfall_deviation = selected_data["rainfall_deviation"]
    total_irrigated_area = selected_data["total_irrigated_area"]

    # Create model input

    sample_input = pd.DataFrame({
        "district_name": [district],
        "crop_name": [crop],
        "crop_type": [crop_type],
        "season": [season],
        "actual_rainfall": [actual_rainfall],
        "normal_rainfall": [normal_rainfall],
        "rainfall_deviation": [rainfall_deviation],
        "total_irrigated_area": [total_irrigated_area]
    })

    # Predict yield

    try:
        predicted_yield = model.predict(sample_input)[0]
        predicted_yield = max(float(predicted_yield), 0.0)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    # Estimate total production

    estimated_total_production = (
        predicted_yield * farmer_area
    )

    # Historical reference year

    historical_year = selected_data.get(
        "agri_year",
        selected_data.get("year_start", "Unknown")
    )

    # Return prediction

    return {
        "prediction": round(predicted_yield, 2),
        "unit": "tonnes/hectare",

        "estimated_total_production": round(
            estimated_total_production,
            2
        ),
        "production_unit": "tonnes",

        "district": district.title(),
        "crop": crop.title(),
        "season": season.title(),

        "farmer_area": round(
            farmer_area,
            2
        ),
        "area_unit": "hectares",

        "historical_reference_year": str(
            historical_year
        )
    }
