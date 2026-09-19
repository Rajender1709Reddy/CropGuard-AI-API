from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity



app = FastAPI(
    title="CropGuard AI - Agricultural Risk Classification API",
    description=(
        "Agricultural risk classification API based strictly "
        "on the ML_Task_Classification notebook."
    ),
    version="1.0.0"
)



BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "crop_risk_classification_model.pkl"
DATA_PATH = BASE_DIR / "Final_dataset_clean.csv"


# LOAD DATASET


try:
    df = pd.read_csv(DATA_PATH)

except Exception as e:
    raise RuntimeError(
        f"Could not load Final_dataset_clean.csv: {e}"
    )



try:
    loaded_model = joblib.load(MODEL_PATH)

except Exception as e:
    raise RuntimeError(
        f"Could not load crop_risk_classification_model.pkl: {e}"
    )



for col in [
    "district_name",
    "crop_name",
    "crop_type",
    "season"
]:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.lower()
    )


df = df.sort_values(
    [
        "district_name",
        "crop_name",
        "season",
        "year"
    ]
).copy()


group_cols = [
    "district_name",
    "crop_name",
    "season"
]



df["previous_year_yield"] = (
    df.groupby(group_cols)["yield"].shift(1)
)

df["previous_year_production"] = (
    df.groupby(group_cols)["production"].shift(1)
)

df["previous_year_area"] = (
    df.groupby(group_cols)["area"].shift(1)
)

df["previous_year_irrigation"] = (
    df.groupby(group_cols)["total_irrigated_area"].shift(1)
)


df["historical_avg_yield"] = (
    df.groupby(group_cols)["yield"]
    .transform(
        lambda x: x.shift(1).expanding().mean()
    )
)

df["historical_avg_production"] = (
    df.groupby(group_cols)["production"]
    .transform(
        lambda x: x.shift(1).expanding().mean()
    )
)

df["historical_avg_irrigation"] = (
    df.groupby(group_cols)["total_irrigated_area"]
    .transform(
        lambda x: x.shift(1).expanding().mean()
    )
)


df["yield_trend"] = (
    df["previous_year_yield"]
    - df["historical_avg_yield"]
)

df["rainfall_anomaly"] = (
    df["actual_rainfall"]
    - df["normal_rainfall"]
)

df["irrigation_change"] = (
    df["total_irrigated_area"]
    - df["previous_year_irrigation"]
)



class ClassificationInput(BaseModel):

    district_name: str = Field(
        ...,
        description="District, e.g. karimnagar"
    )

    crop_name: str = Field(
        ...,
        description="Crop, e.g. rice"
    )

    season: str = Field(
        ...,
        description="Season, e.g. kharif"
    )

    area: float = Field(
        ...,
        gt=0,
        description="Area in hectares"
    )


def clean_text(value):
    return (
        str(value)
        .strip()
        .lower()
    )



@app.get("/")
def home():

    return {

        "message":
            "CropGuard AI Agricultural Risk Classification API is running",

        "status":
            "success",

        "user_inputs": [
            "district_name",
            "crop_name",
            "season",
            "area"
        ],

        "outputs": [
            "Predicted Risk",
            "Risk Probability",
            "Main Contributing Factors",
            "Recommended Crop"
        ],

        "docs":
            "/docs"
    }



@app.get("/health")
def health():

    return {

        "status":
            "healthy",

        "model_loaded":
            loaded_model is not None,

        "dataset_loaded":
            df is not None
    }

@app.post("/predict")
def predict_risk(
    data: ClassificationInput
):

    district_name = clean_text(
        data.district_name
    )

    crop_name = clean_text(
        data.crop_name
    )

    season = clean_text(
        data.season
    )

    area = data.area




    matching_data = df[
        (df["district_name"] == district_name) &
        (df["crop_name"] == crop_name) &
        (df["season"] == season)
    ]


    if matching_data.empty:

        raise HTTPException(
            status_code=404,
            detail=(
                "No historical data available. "
                "Please select another combination."
            )
        )



    matching_data = matching_data.sort_values(
        "year"
    )

    selected_data = matching_data.iloc[-1]


    # Real-Time Features

    area_change = (
        area
        - selected_data["previous_year_area"]
    )


    irrigation_coverage = (
        selected_data["total_irrigated_area"] / area
        if area > 0
        else 0
    )


    # Model Input

    sample_input = pd.DataFrame({

        "previous_year_yield": [
            selected_data["previous_year_yield"]
        ],

        "historical_avg_yield": [
            selected_data["historical_avg_yield"]
        ],

        "yield_trend": [
            selected_data["yield_trend"]
        ],

        "previous_year_production": [
            selected_data["previous_year_production"]
        ],

        "historical_avg_production": [
            selected_data["historical_avg_production"]
        ],

        "previous_year_area": [
            selected_data["previous_year_area"]
        ],

        "area_change": [
            area_change
        ],

        "normal_rainfall": [
            selected_data["normal_rainfall"]
        ],

        "rainfall_anomaly": [
            selected_data["rainfall_anomaly"]
        ],

        "rainfall_deviation": [
            selected_data["rainfall_deviation"]
        ],

        "previous_year_irrigation": [
            selected_data["previous_year_irrigation"]
        ],

        "historical_avg_irrigation": [
            selected_data["historical_avg_irrigation"]
        ],

        "irrigation_change": [
            selected_data["irrigation_change"]
        ],

        "irrigation_coverage": [
            irrigation_coverage
        ],

        "district_name": [
            district_name
        ],

        "crop_name": [
            crop_name
        ],

        "crop_type": [
            selected_data["crop_type"]
        ],

        "season": [
            season
        ]
    })



    predicted_risk = loaded_model.predict(
        sample_input
    )[0]



    previous_yield = selected_data[
        "previous_year_yield"
    ]

    historical_yield = selected_data[
        "historical_avg_yield"
    ]

    rainfall_deviation = selected_data[
        "rainfall_deviation"
    ]

    irrigation_change = selected_data[
        "irrigation_change"
    ]



    if (
        pd.notna(previous_yield)
        and pd.notna(historical_yield)
        and historical_yield > 0
    ):

        yield_gap = (
            (
                historical_yield
                - previous_yield
            )
            / historical_yield
        ) * 100

        yield_risk = max(
            0,
            min(yield_gap, 100)
        )

    else:

        yield_risk = 0


    if pd.notna(rainfall_deviation):

        rainfall_risk = max(
            0,
            min(
                (-rainfall_deviation / 100) * 100,
                100
            )
        )

    else:

        rainfall_risk = 0


    previous_irrigation = selected_data[
        "previous_year_irrigation"
    ]

    if (
        pd.notna(irrigation_change)
        and pd.notna(previous_irrigation)
        and previous_irrigation > 0
    ):

        irrigation_risk = (
            -irrigation_change
            / previous_irrigation
        ) * 100

        irrigation_risk = max(
            0,
            min(irrigation_risk, 100)
        )

    else:

        irrigation_risk = 0



    previous_area = selected_data[
        "previous_year_area"
    ]

    if (
        pd.notna(previous_area)
        and previous_area > 0
    ):

        area_change_percentage = (
            abs(
                area
                - previous_area
            )
            / previous_area
        ) * 100

        area_risk = max(
            0,
            min(area_change_percentage, 100)
        )

    else:

        area_risk = 0




    estimated_risk_probability = (
        0.50 * yield_risk
        + 0.20 * rainfall_risk
        + 0.20 * irrigation_risk
        + 0.10 * area_risk
    )

    estimated_risk_probability = max(
        0,
        min(
            estimated_risk_probability,
            100
        )
    )


    # Rainfall
    if rainfall_deviation < -20:

        rainfall_factor = (
            "High contribution"
        )

    elif rainfall_deviation < 0:

        rainfall_factor = (
            "Medium contribution"
        )

    else:

        rainfall_factor = (
            "Low contribution"
        )


    # Historical instability
    if yield_risk > 30:

        historical_factor = (
            "High contribution"
        )

    elif yield_risk > 15:

        historical_factor = (
            "Medium contribution"
        )

    else:

        historical_factor = (
            "Low contribution"
        )


    # Irrigation
    if irrigation_change < 0:

        irrigation_factor = (
            "Medium contribution"
        )

    else:

        irrigation_factor = (
            "Low contribution"
        )


    # Yield trend
    if selected_data["yield_trend"] < 0:

        trend_factor = (
            "High contribution"
        )

    elif selected_data["yield_trend"] == 0:

        trend_factor = (
            "Medium contribution"
        )

    else:

        trend_factor = (
            "Low contribution"
        )



    candidate_crops = df[
        (df["district_name"] == district_name) &
        (df["season"] == season)
    ]["crop_name"].dropna().unique()


    recommendation_features = [
        "historical_avg_yield",
        "normal_rainfall",
        "rainfall_deviation",
        "historical_avg_irrigation",
        "previous_year_area"
    ]


    crop_profiles = []


    for candidate_crop in candidate_crops:

        # Do not recommend selected crop
        if candidate_crop == crop_name:
            continue


        candidate_data = df[
            (df["district_name"] == district_name) &
            (df["crop_name"] == candidate_crop) &
            (df["season"] == season)
        ].sort_values("year")


        if candidate_data.empty:
            continue


        candidate_data = candidate_data.iloc[-1]


        profile = {
            "crop": candidate_crop
        }


        for feature in recommendation_features:

            profile[feature] = candidate_data[
                feature
            ]


        crop_profiles.append(profile)


    crop_profiles_df = pd.DataFrame(
        crop_profiles
    )



    if crop_profiles_df.empty:

        recommended_crop_name = "Not Available"

        recommendation_score = 0

        similarity_score = 0

        recommended_crop_risk = 0

        recommended_crop_level = "N/A"

    else:


        crop_profiles_df[
            recommendation_features
        ] = crop_profiles_df[
            recommendation_features
        ].fillna(
            crop_profiles_df[
                recommendation_features
            ].median()
        )



        user_profile = selected_data[
            recommendation_features
        ].copy()


        user_profile = pd.DataFrame(
            [user_profile]
        )


        # Use current user-entered area
        user_profile[
            "previous_year_area"
        ] = area


        # Fill missing values
        user_profile[
            recommendation_features
        ] = user_profile[
            recommendation_features
        ].fillna(
            crop_profiles_df[
                recommendation_features
            ].median()
        )



        scaler = MinMaxScaler()


        candidate_scaled = scaler.fit_transform(
            crop_profiles_df[
                recommendation_features
            ]
        )


        user_scaled = scaler.transform(
            user_profile[
                recommendation_features
            ]
        )


  

        similarity_scores = cosine_similarity(
            user_scaled,
            candidate_scaled
        )[0]


        crop_profiles_df[
            "similarity_score"
        ] = similarity_scores * 100



        recommendation_results = []


        for index, row in crop_profiles_df.iterrows():

            candidate_crop = row["crop"]


            candidate_data = df[
                (df["district_name"] == district_name) &
                (df["crop_name"] == candidate_crop) &
                (df["season"] == season)
            ].sort_values("year")


            if candidate_data.empty:
                continue


            candidate_data = candidate_data.iloc[-1]


            # Candidate Yield Risk
            candidate_previous_yield = (
                candidate_data[
                    "previous_year_yield"
                ]
            )

            candidate_historical_yield = (
                candidate_data[
                    "historical_avg_yield"
                ]
            )


            if (
                pd.notna(candidate_previous_yield)
                and
                pd.notna(candidate_historical_yield)
                and
                candidate_historical_yield > 0
            ):

                candidate_yield_risk = (
                    (
                        candidate_historical_yield
                        - candidate_previous_yield
                    )
                    /
                    candidate_historical_yield
                ) * 100

                candidate_yield_risk = max(
                    0,
                    min(
                        candidate_yield_risk,
                        100
                    )
                )

            else:

                candidate_yield_risk = 0


            # Candidate Rainfall Risk
            candidate_rainfall_deviation = (
                candidate_data[
                    "rainfall_deviation"
                ]
            )


            if pd.notna(
                candidate_rainfall_deviation
            ):

                candidate_rainfall_risk = max(
                    0,
                    min(
                        -candidate_rainfall_deviation,
                        100
                    )
                )

            else:

                candidate_rainfall_risk = 0


            # Candidate Irrigation Risk
            candidate_irrigation_change = (
                candidate_data[
                    "irrigation_change"
                ]
            )

            candidate_previous_irrigation = (
                candidate_data[
                    "previous_year_irrigation"
                ]
            )


            if (
                pd.notna(candidate_irrigation_change)
                and
                pd.notna(candidate_previous_irrigation)
                and
                candidate_previous_irrigation > 0
            ):

                candidate_irrigation_risk = (
                    -candidate_irrigation_change
                    /
                    candidate_previous_irrigation
                ) * 100

                candidate_irrigation_risk = max(
                    0,
                    min(
                        candidate_irrigation_risk,
                        100
                    )
                )

            else:

                candidate_irrigation_risk = 0


            # Candidate Area Risk
            candidate_previous_area = (
                candidate_data[
                    "previous_year_area"
                ]
            )


            if (
                pd.notna(candidate_previous_area)
                and
                candidate_previous_area > 0
            ):

                candidate_area_risk = (
                    abs(
                        area
                        - candidate_previous_area
                    )
                    /
                    candidate_previous_area
                ) * 100

                candidate_area_risk = max(
                    0,
                    min(
                        candidate_area_risk,
                        100
                    )
                )

            else:

                candidate_area_risk = 0


            # Overall Agricultural Risk
            candidate_risk_score = (
                0.50 * candidate_yield_risk
                + 0.20 * candidate_rainfall_risk
                + 0.20 * candidate_irrigation_risk
                + 0.10 * candidate_area_risk
            )


            candidate_risk_score = max(
                0,
                min(
                    candidate_risk_score,
                    100
                )
            )


            # Risk Level
            if candidate_risk_score < 33:

                candidate_risk_level = (
                    "Low Risk"
                )

            elif candidate_risk_score < 66:

                candidate_risk_level = (
                    "Moderate Risk"
                )

            else:

                candidate_risk_level = (
                    "High Risk"
                )


            # Content-Based Recommendation Score
            similarity_score = (
                row["similarity_score"]
            )


            recommendation_score = (
                0.70 * similarity_score
                +
                0.30 * (
                    100
                    - candidate_risk_score
                )
            )


            recommendation_results.append({

                "crop": candidate_crop,

                "similarity_score":
                    similarity_score,

                "risk_score":
                    candidate_risk_score,

                "risk_level":
                    candidate_risk_level,

                "recommendation_score":
                    recommendation_score
            })



        if recommendation_results:

            recommendation_results = sorted(
                recommendation_results,
                key=lambda x:
                    x["recommendation_score"],
                reverse=True
            )


            recommended_crop = (
                recommendation_results[0]
            )


            recommended_crop_name = (
                recommended_crop["crop"]
            )


            recommendation_score = (
                recommended_crop[
                    "recommendation_score"
                ]
            )


            similarity_score = (
                recommended_crop[
                    "similarity_score"
                ]
            )


            recommended_crop_risk = (
                recommended_crop[
                    "risk_score"
                ]
            )


            recommended_crop_level = (
                recommended_crop[
                    "risk_level"
                ]
            )


        else:

            recommended_crop_name = (
                "Not Available"
            )

            recommendation_score = 0

            similarity_score = 0

            recommended_crop_risk = 0

            recommended_crop_level = "N/A"




    return {

        "status": "success",

        "district":
            district_name.title(),

        "crop":
            crop_name.title(),

        "season":
            season.title(),

        "area":
            area,

        "area_unit":
            "hectares",

        "predicted_risk":
            str(predicted_risk),

        "risk_probability":
            round(
                float(
                    estimated_risk_probability
                ),
                0
            ),

        "risk_probability_unit":
            "percent",

        "main_contributing_factors": {

            "Rainfall condition":
                rainfall_factor,

            "Historical instability":
                historical_factor,

            "Irrigation condition":
                irrigation_factor,

            "Historical yield trend":
                trend_factor
        },

        "recommended_crop": {

            "crop":
                recommended_crop_name.title(),

            "risk":
                round(
                    float(
                        recommended_crop_risk
                    ),
                    1
                ),

            "risk_level":
                recommended_crop_level.upper(),

            "recommendation_score":
                round(
                    float(
                        recommendation_score
                    ),
                    2
                )
        }
    }


