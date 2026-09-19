#!/usr/bin/env python
# coding: utf-8

# #### Project: CropGuard AI
# 
# The project uses two supervised ML tasks.
# 
# 1. 🌾 Pre-Cultivation Crop Yield Prediction
# ML Task: Regression
# Target (Y): yield — existing numerical column
# Purpose: Predict the expected crop yield in tonnes/hectare for a selected district, crop, season, and agricultural year using historical crop performance, rainfall, irrigation, and contextual information.
# Output: Predicted Yield
# Example: 4.82 tonnes/hectare
# 2. ⚠️ Agricultural Risk Classification
# ML Task: Classification
# Target (Y): risk_level — derived target
# Classes: Low Risk / Moderate Risk / High Risk
# Purpose: Identify the agricultural risk associated with a particular district–crop–season combination using historical crop performance, rainfall conditions, irrigation conditions, and other validated features.
# Output: Risk Level + Risk Probability
# Example: HIGH RISK — 78% probability

# In[1]:


### coding 


# In[2]:


## importing libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns



# In[3]:


import os 
path = r"C:\Users\91934\OneDrive\Desktop\ML Project\ML Project\data\ml\Final_dataset_clean.csv"
print(os.path.exists(path))


# In[4]:


df = pd.read_csv(path)
print("Shape:", df.shape)


# ### 0. Data Loading (from db load tables data as pandas df)

# In[5]:


from sqlalchemy import create_engine
from urllib.parse import quote_plus

username = "root"
password = input("Enter MySQL password: ")

engine = create_engine(
    f"mysql+pymysql://{username}:{quote_plus(password)}@localhost:3306/cropguard_ai"
)

with engine.connect() as connection:
    print("Database connection successful!")


# In[6]:


# Load validated data from database
query = "SELECT * FROM validated_agriculture_data"

df = pd.read_sql(query, engine)

print("Dataset Shape:", df.shape)


# In[7]:


df.head()


# In[8]:


df = pd.read_sql(
    """
    SELECT *
    FROM validated_agriculture_data
    """,
    engine
)

print("Data loaded from MySQL successfully!")
print("Shape:", df.shape)
df.head()


# ### CLASSIFICATION TASK — Agrimcultural Risk Classification 

# #### 1. Modeling 

#  ### 1.1 Pre-Requisites

# #### 1.1 Prerequisites – Creating Target Variable (Y)
# 
# Before building the classification model, the target variable `risk_level` must be created.
# 
# The dataset does not contain a predefined `risk_level` column. Therefore, the target variable is derived from historical agricultural performance, rainfall conditions, and irrigation conditions.
# 
# The risk levels are classified into three categories:
# 
# - **Low Risk** – agricultural conditions are relatively stable.
# - **Moderate Risk** – agricultural conditions show some level of stress.
# - **High Risk** – agricultural conditions show significant agricultural stress.
# 
# The target variable is created using historical information so that current-year outcome information is not directly used as an input feature. This helps reduce data leakage and makes the risk classification more suitable for prediction.

# In[9]:


## Sort Data
df = df.sort_values(
    by=["district_name", "crop_name", "season", "year"]
).reset_index(drop=True)

## Define Group
group_cols = [
    "district_name",
    "crop_name",
    "season"
]


# In[10]:


## Previous-year features

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


# In[11]:


## Fill missing previous-year values

df["previous_year_yield"] = df["previous_year_yield"].fillna(
    df["yield"]
)

df["previous_year_production"] = df["previous_year_production"].fillna(
    df["production"]
)

df["previous_year_area"] = df["previous_year_area"].fillna(
    df["area"]
)

df["previous_year_irrigation"] = df["previous_year_irrigation"].fillna(
    df["total_irrigated_area"]
)


# In[12]:


## Historical averages

df["historical_avg_yield"] = (
    df.groupby(group_cols)["yield"]
    .transform(lambda x: x.shift(1).expanding().mean())
)

df["historical_avg_production"] = (
    df.groupby(group_cols)["production"]
    .transform(lambda x: x.shift(1).expanding().mean())
)

df["historical_avg_irrigation"] = (
    df.groupby(group_cols)["total_irrigated_area"]
    .transform(lambda x: x.shift(1).expanding().mean())
)


# In[13]:


df["historical_avg_yield"] = df["historical_avg_yield"].fillna(
    df["previous_year_yield"]
)

df["historical_avg_production"] = df["historical_avg_production"].fillna(
    df["previous_year_production"]
)

df["historical_avg_irrigation"] = df["historical_avg_irrigation"].fillna(
    df["previous_year_irrigation"]
)


# In[14]:


## Yield trend
df["yield_trend"] = (
    df["previous_year_yield"]
    - df["historical_avg_yield"]
)


# In[15]:


## Area change
df["area_change"] = (
    df["area"]
    - df["previous_year_area"]
)


# In[16]:


## Historical rainfall
df["historical_rainfall"] = (
    df.groupby(group_cols)["actual_rainfall"]
    .transform(
        lambda x: x.shift(1).expanding().mean()
    )
)


# In[17]:


## Rainfall anomaly
df["rainfall_anomaly"] = (
    df["actual_rainfall"]
    - df["normal_rainfall"]
)


# In[18]:


## Irrigation change
df["irrigation_change"] = (
    df["total_irrigated_area"]
    - df["previous_year_irrigation"]
)


# In[19]:


## Irrigation coverage
df["irrigation_coverage"] = (
    df["total_irrigated_area"]
    / df["area"]
)


# #### Create Risk Level

# In[20]:


# Yield stress
df["yield_stress"] = (
    df["historical_avg_yield"]
    - df["previous_year_yield"]
)


# In[21]:


# Rainfall stress
df["rainfall_stress"] = (
    -df["rainfall_deviation"]
)


# In[22]:


# Irrigation stress
df["irrigation_stress"] = (
    -df["irrigation_change"]
)


# In[23]:


# Convert stress into percentile-based scores
df["yield_stress_score"] = (
    df["yield_stress"].rank(pct=True)
)

df["rainfall_stress_score"] = (
    df["rainfall_stress"].rank(pct=True)
)

df["irrigation_stress_score"] = (
    df["irrigation_stress"].rank(pct=True)
)


# In[24]:


# Combined risk score
df["risk_score"] = (
    0.5 * df["yield_stress_score"]
    + 0.3 * df["rainfall_stress_score"]
    + 0.2 * df["irrigation_stress_score"]
)


# #### Create risk level

# In[25]:


df["risk_level"] = pd.cut(
    df["risk_score"],
    bins=[-np.inf, 0.33, 0.66, np.inf],
    labels=[
        "Low Risk",
        "Moderate Risk",
        "High Risk"
    ]
)


# In[26]:


## Check target
print(df["risk_level"].value_counts())


# #### 1.1.1 X & y:
# - Picking y output col for Predictive Modelling & considering remaining cols are X

# In[27]:


#### 1.1.1 X & y

# Select input features (X) and target variable (y)

features = [
    "previous_year_yield",
    "historical_avg_yield",
    "yield_trend",
    "previous_year_production",
    "historical_avg_production",
    "previous_year_area",
    "area_change",
    "normal_rainfall",
    "rainfall_anomaly",
    "rainfall_deviation",
    "previous_year_irrigation",
    "historical_avg_irrigation",
    "irrigation_change",
    "irrigation_coverage",
    "district_name",
    "crop_name",
    "crop_type",
    "season"
]

# Define X and y
X = df[features]
y = df["risk_level"]

print("X shape:", X.shape)
print("y shape:", y.shape)


# #### 1.1.2 Train-Test Split:
# 
# The dataset is divided into training and testing data based on the `year` column.
# 
# Since the agricultural risk classification problem uses historical agricultural data, a **time-based train-test split** is used instead of a random split.
# 
# - **Training data:** 2014 to 2021
# - **Testing data:** 2022 to 2024
# 
# The data from 2014–2021 is used for model learning, while the data from 2022–2024 is kept as unseen future data for testing the trained classification model.
# 
# This approach is suitable for the project because it simulates a real-world situation where historical agricultural information is used to predict risk for future years.

# In[28]:


## Starting year
df["year_start"] = (
    df["agri_year"]
    .astype(str)
    .str.extract(r"(\d{4})")[0]
    .astype(int)
)


# In[29]:


df = df.sort_values("year_start").reset_index(drop=True)
years = sorted(df["year_start"].unique())
print("Available agricultural years:")
print(years)


# In[30]:


year_counts = df["year_start"].value_counts().sort_index()

print("\nRecords available for each year:")
display(year_counts)


# In[31]:


split_index = int(len(years) * 0.80)

train_years = years[:split_index]
test_years = years[split_index:]

print("\nTraining years:")
print(train_years)

print("\nTesting years:")
print(test_years)


# In[32]:


# Create train-test masks
train_mask = df["year_start"].isin(train_years)
test_mask = df["year_start"].isin(test_years)


# In[33]:


# Splitting X and y
X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_test = y.loc[test_mask].copy()


# Check shapes
print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# In[34]:


# Check missing values in target
print("\nMissing values in y_train:", y_train.isna().sum())
print("Missing values in y_test:", y_test.isna().sum())


# Check target distribution
print("\nTraining target:")
print(y_train.value_counts())

print("\nTesting target:")
print(y_test.value_counts())


# In[35]:


# Preview target
print("\ny_train head:")
display(y_train.head())


# #### 1.1.3 Missing Values & Outliers Handling:
# 
# Missing values are checked separately for the input variables and the target variable.
# 
# Missing values in the target variable `risk_level` are removed because the classification model cannot learn from a record without a known risk category.
# 
# Missing values in the input features will be handled later through the preprocessing pipeline.
# 
# Outliers in numerical features such as yield, production, area, rainfall, and irrigation are investigated rather than automatically removed, because extreme values may represent genuine differences between districts, crops, seasons, or agricultural conditions.

# In[36]:


# Check missing values in target
print("Missing values in risk_level:")
print(df["risk_level"].isnull().sum())


# In[37]:


## check numerical outliers
numeric_columns = [
    "area",
    "production",
    "yield",
    "actual_rainfall",
    "normal_rainfall",
    "rainfall_deviation",
    "total_irrigated_area"
]

for col in numeric_columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = (
        (df[col] < lower_limit) |
        (df[col] > upper_limit)
    ).sum()

    print(f"{col}: {outliers} potential outliers")


# ### Outlier Decision
# 
# The IQR method identified potential outliers in area (2,016), production (2,052), yield (1,184), and rainfall deviation (49). No potential outliers were detected in actual rainfall, normal rainfall, or total irrigated area.
# 
# These observations are retained because extreme values may represent genuine differences across crops, districts, and seasons rather than data errors. Therefore, no observations are removed solely based on the IQR method. The potential impact of extreme values will instead be considered during model training and evaluation.

# ### 1.1.4 Feature Engineering of X for Y
# 
# Feature engineering is performed to prepare the input variables for predicting
# the agricultural risk level.
# 
# The available dataset contains agricultural, rainfall and irrigation information.
# Additional historical features are generated from the existing data to capture
# previous performance, trends, variability and changes over time.

# #### 1.1.4.1 Feature Generation
# 
# Historical features are generated using the `year`, district, crop and season
# information.
# 
# Features such as previous-year yield, historical average yield, yield trend,
# yield variability, rainfall anomaly, rainfall variability, previous-year
# irrigation, irrigation change and irrigation coverage are created to represent
# agricultural stress and historical conditions.

# #### 1.1.4.2 Feature Selection
# 
# The following features are selected based on their relevance to agricultural
# risk identification and their availability in the prepared dataset.
# 
# The selected features include historical crop performance, rainfall conditions,
# irrigation conditions and agricultural context such as district, crop, crop type
# and season.

# In[38]:


features = [
    "previous_year_yield",
    "historical_avg_yield",
    "yield_trend",
    "previous_year_production",
    "historical_avg_production",
    "previous_year_area",
    "area_change",
    "normal_rainfall",
    "rainfall_anomaly",
    "rainfall_deviation",
    "previous_year_irrigation",
    "historical_avg_irrigation",
    "irrigation_change",
    "irrigation_coverage",
    "district_name",
    "crop_name",
    "crop_type",
    "season"
]

X = df[features]
y = df["risk_level"]

print("X shape:", X.shape)
print("y shape:", y.shape)


# #### 1.1.4.3 Feature Type Identification
# 
# The selected input features contain both **categorical and numerical variables**.
# 
# Categorical features such as `district_name`, `crop_name`, `crop_type`, and `season` will be converted into numerical form using **One-Hot Encoding**.
# 
# Numerical features such as historical yield, production, area, rainfall and irrigation features will be used as numerical inputs. Scaling will be applied for distance-based algorithms such as **KNN and SVM**.

# In[39]:


categorical_features = [
    "district_name",
    "crop_name",
    "crop_type",
    "season"
]

numerical_features = [
    "previous_year_yield",
    "historical_avg_yield",
    "yield_trend",
    "previous_year_production",
    "historical_avg_production",
    "previous_year_area",
    "area_change",
    "normal_rainfall",
    "rainfall_anomaly",
    "rainfall_deviation",
    "previous_year_irrigation",
    "historical_avg_irrigation",
    "irrigation_change",
    "irrigation_coverage"
]

print("Categorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)


# #### 1.1.4.4 Missing Value Handling
# 
# Missing numerical values will be replaced using the **median** calculated from
# the training data.
# 
# Missing categorical values will be replaced using the **most frequent value**
# from the training data.
# 
# This approach prevents missing values from affecting model training while
# ensuring that information from the test dataset does not influence the
# preprocessing process.

# #### 1.1.4.5 Encoding
# 
# Categorical features such as `district_name`, `crop_name`, `crop_type`, and
# `season` are converted into numerical values using **One-Hot Encoding**.
# 
# One-hot encoding creates separate binary columns for each category, allowing
# machine learning classification algorithms to process the categorical data.

# In[40]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]
)

numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# In[41]:


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


print("Preprocessing pipeline created successfully.")
print(preprocessor)


# #### 1.1.4.6 Scaling
# For numerical features

# In[42]:


from sklearn.preprocessing import StandardScaler

numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("Preprocessing pipeline created successfully.")

print(preprocessor)


# ### 1.2 Model + Define, Train & Study
# 
# For the Agricultural Risk Classification problem, classification algorithms are
# selected based on the categorical target variable `risk_level`.
# 
# The following classification algorithms are considered:
# 
# - **Logistic Regression**
# - **K-Nearest Neighbors (KNN)**
# - **Naive Bayes**
# - **Support Vector Machine (SVM)**
# - **Decision Tree Classifier**
# - **Random Forest Classifier**
# 
# XGBoost is not used in this project due to installation and resource constraints.
# 
# The selected models are loaded using the Python `sklearn` library and trained
# using the training data. Their performance will later be evaluated using the
# test data.

# In[43]:


## Import Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[44]:


## Define Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

print("Classification models defined successfully.")


# #### 1.2.4 Train Models

# In[45]:


trained_models = {}

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    trained_models[name] = pipeline

    print(f"{name} trained successfully.")


# ### 1.3 Trained Model Predictions & Evaluations
# 
# The trained classification models are used to predict the `risk_level` for the
# unseen test data.
# 
# The model performance is evaluated using classification metrics such as
# **Accuracy, Precision, Recall and F1-Score** along with the **Confusion Matrix**.
# 
# The training and testing performance of each model is compared to identify
# underfitting and overfitting using the **Bias-Variance Trade-off**.
# 
# Finally, the best classification model is selected based on its overall
# performance on the test dataset, with greater importance given to Precision,
# Recall and F1-Score along with Accuracy.

# #### 1.3.1 Predictions on Train and Test Data
# 
# Predictions are generated using all trained classification models on both
# training and test datasets.
# 
# The training predictions are used to understand model fitting, while
# test predictions are used to evaluate how well the models perform on
# unseen data.

# In[46]:


predictions = {}

for name, model in trained_models.items():

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    predictions[name] = {
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred
    }

    print(f"{name} predictions completed.")


# #### 1.3.2 Classification Evaluation Metrics

# In[47]:


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

results = []

for name in predictions:

    y_train_pred = predictions[name]["y_train_pred"]
    y_test_pred = predictions[name]["y_test_pred"]

    # Training metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)

    train_precision = precision_score(
        y_train,
        y_train_pred,
        average="weighted",
        zero_division=0
    )

    train_recall = recall_score(
        y_train,
        y_train_pred,
        average="weighted",
        zero_division=0
    )

    train_f1 = f1_score(
        y_train,
        y_train_pred,
        average="weighted",
        zero_division=0
    )

    # Testing metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)

    test_precision = precision_score(
        y_test,
        y_test_pred,
        average="weighted",
        zero_division=0
    )

    test_recall = recall_score(
        y_test,
        y_test_pred,
        average="weighted",
        zero_division=0
    )

    test_f1 = f1_score(
        y_test,
        y_test_pred,
        average="weighted",
        zero_division=0
    )

    results.append({
        "Models": name,
        "Train Accuracy": train_accuracy,
        "Test Accuracy": test_accuracy,
        "Train Precision": train_precision,
        "Test Precision": test_precision,
        "Train Recall": train_recall,
        "Test Recall": test_recall,
        "Train F1": train_f1,
        "Test F1": test_f1
    })


results_df = pd.DataFrame(results)

results_df.round(3)


# #### 1.3.3 Bias-Variance Trade-off

# In[48]:


def fit_type(train_score, test_score):

    difference = train_score - test_score

    if train_score < 0.70 and test_score < 0.70:
        return "Underfit"

    elif difference > 0.10:
        return "Overfit"

    else:
        return "Good Fit"


# In[49]:


results_df["Fit Assessment"] = results_df.apply(
    lambda row: fit_type(
        row["Train F1"],
        row["Test F1"]
    ),
    axis=1
)

results_df.round(3)


# #### 1.3.4 Final Classification Evaluation Table

# In[50]:


evaluation_table = results_df[
    [
        "Models",
        "Train F1",
        "Test F1",
        "Test Accuracy",
        "Test Precision",
        "Test Recall",
        "Fit Assessment"
    ]
].copy()

evaluation_table = evaluation_table.round({
    "Train F1": 2,
    "Test F1": 2,
    "Test Accuracy": 2,
    "Test Precision": 2,
    "Test Recall": 2
})

display(evaluation_table)


# #### 1.3.5 Pick the best Model

# In[51]:


# Select the model with the highest Test F1-Score
best_model_name = evaluation_table.loc[
    evaluation_table["Test F1"].idxmax(),
    "Models"
]

print("Best Classification Model:", best_model_name)


# In[52]:


best_model_result = evaluation_table[
    evaluation_table["Models"] == best_model_name
]

display(best_model_result)


# Based on the evaluation results, the Decision Tree Classifier achieved the highest Test F1-Score of 0.97, along with a Test Accuracy of 0.97, Test Precision of 0.97, and Test Recall of 0.97.
# 
# The model achieved a Train F1-Score of 1.00 and a Test F1-Score of 0.97, resulting in a small train-test performance gap of 0.03. Based on the defined fit assessment, the Decision Tree is classified as Good Fit.
# 
# Therefore, the Decision Tree Classifier is selected as the final classification model because it provides the strongest overall performance on unseen test data while maintaining a relatively small difference between training and testing performance.

# #### 1.4 Hyperparameter Tuning
# Hyperparameter tuning is performed to improve the performance of the selected classification model and reduce overfitting.
# 
# The hyperparameters of the selected model are adjusted and evaluated using cross-validation on the training data, with weighted F1-Score used as the optimization metric. The tuned model is then evaluated on the unseen test dataset and compared with the original model to determine whether its performance and generalization have improved.
# 
# If tuning does not provide a meaningful improvement on the unseen test data, the original model is retained as the final model.

# In[ ]:


from sklearn.model_selection import GridSearchCV

# Hyperparameter grid for Decision Tree
param_grid = {
    "model__max_depth": [3, 5, 7, 10, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 5, 10]
}

# Get the trained Decision Tree pipeline
dt_model = trained_models["Decision Tree"]

# Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=dt_model,
    param_grid=param_grid,
    cv=5,
    scoring="f1_weighted",
    n_jobs=-1
)

# Tune using training data only
grid_search.fit(X_train, y_train)

print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation F1-Score:")
print(round(grid_search.best_score_, 3))


# In[ ]:


tuned_dt = grid_search.best_estimator_

print("Tuned Decision Tree model created successfully.")


# In[ ]:


from sklearn.metrics import f1_score, accuracy_score

# Predictions
y_train_pred_tuned = tuned_dt.predict(X_train)
y_test_pred_tuned = tuned_dt.predict(X_test)

# F1-Score
train_f1_tuned = f1_score(
    y_train,
    y_train_pred_tuned,
    average="weighted",
    zero_division=0
)

test_f1_tuned = f1_score(
    y_test,
    y_test_pred_tuned,
    average="weighted",
    zero_division=0
)

# Accuracy
train_accuracy_tuned = accuracy_score(
    y_train,
    y_train_pred_tuned
)

test_accuracy_tuned = accuracy_score(
    y_test,
    y_test_pred_tuned
)

# Display results
print("Tuned Decision Tree")
print("--------------------------")
print("Train F1:", round(train_f1_tuned, 3))
print("Test F1:", round(test_f1_tuned, 3))
print("Train Accuracy:", round(train_accuracy_tuned, 3))
print("Test Accuracy:", round(test_accuracy_tuned, 3))


# ### 1.5 Saving Model & Real-Time Prediction

# #### 1.5.1 Select Final Model
# 
# Based on the model evaluation and hyperparameter tuning results, the Decision Tree Classifier was selected as the final classification model for agricultural risk prediction.
# 
# The original Decision Tree achieved a Test F1-Score of 0.97. After hyperparameter tuning, the tuned model was evaluated on the unseen test dataset and compared with the original model.

# In[ ]:


final_model = tuned_dt

print(
    "Final model:",
    type(final_model.named_steps["model"]).__name__
)


# #### 1.5.2 Save the Best Model

# In[ ]:


import joblib
joblib.dump(
    final_model,
    "crop_risk_classification_model.pkl"
)

print("Best classification model saved successfully.")


# #### 1.5.3 Load the Saved Model

# In[ ]:


loaded_model = joblib.load(
    "crop_risk_classification_model.pkl"
)

print("Saved classification model loaded successfully.")


# #### 1.5.4 Real-Time Prediction

# In[ ]:


import pandas as pd
import joblib
import ipywidgets as widgets
from IPython.display import display, clear_output

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

data_path = r"C:\Users\91934\OneDrive\Desktop\ML Project\ML Project\data\ml\Final_dataset_clean.csv"

df = pd.read_csv(data_path)

loaded_model = joblib.load(
    "crop_risk_classification_model.pkl"
)


# In[ ]:


## Prepare text columns
for col in ["district_name", "crop_name", "crop_type", "season"]:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.lower()
    )


# In[ ]:


## sort data
df = df.sort_values(
    ["district_name", "crop_name", "season", "year"]
).copy()

group_cols = [
    "district_name",
    "crop_name",
    "season"
]


# In[ ]:


## historical features
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


# In[ ]:


## dropdown
districts = sorted(
    df["district_name"].dropna().unique()
)

district_dropdown = widgets.Dropdown(
    options=districts,
    description="District:",
    style={"description_width": "initial"}
)

crop_dropdown = widgets.Dropdown(
    options=[],
    description="Crop:",
    style={"description_width": "initial"}
)

season_dropdown = widgets.Dropdown(
    options=[],
    description="Season:",
    style={"description_width": "initial"}
)

area_input = widgets.FloatText(
    value=1.0,
    min=0.01,
    description="Area:",
    style={"description_width": "initial"}
)

predict_button = widgets.Button(
    description="Predict Risk",
    button_style="success"
)

output = widgets.Output()


# In[ ]:


## district->crop
def update_crops(change):

    selected_district = change["new"]

    crop_dropdown.options = []
    season_dropdown.options = []

    if selected_district is None:
        return

    available_crops = sorted(
        df.loc[
            df["district_name"] == selected_district,
            "crop_name"
        ]
        .dropna()
        .unique()
    )

    crop_dropdown.options = available_crops

    if len(available_crops) > 0:
        crop_dropdown.value = available_crops[0]



# In[ ]:


## district + crop -> season
def update_seasons(change):

    selected_district = district_dropdown.value
    selected_crop = change["new"]

    season_dropdown.options = []

    if selected_district is None or selected_crop is None:
        return

    available_seasons = sorted(
        df.loc[
            (df["district_name"] == selected_district) &
            (df["crop_name"] == selected_crop),
            "season"
        ]
        .dropna()
        .unique()
    )

    season_dropdown.options = available_seasons

    if len(available_seasons) > 0:
        season_dropdown.value = available_seasons[0]


district_dropdown.observe(
    update_crops,
    names="value"
)

crop_dropdown.observe(
    update_seasons,
    names="value"
)


# In[ ]:


## Prediction

def predict_risk(button):

    with output:

        clear_output()

        district_name = district_dropdown.value
        crop_name = crop_dropdown.value
        season = season_dropdown.value
        area = area_input.value

        matching_data = df[
            (df["district_name"] == district_name) &
            (df["crop_name"] == crop_name) &
            (df["season"] == season)
        ]

        if matching_data.empty:

            print("\nNo historical data available.")
            print("Please select another combination.")

            return


        ## Latest Historical Record


        matching_data = matching_data.sort_values("year")

        selected_data = matching_data.iloc[-1]



        ## Real-Time Features

        area_change = (
            area -
            selected_data["previous_year_area"]
        )

        irrigation_coverage = (
            selected_data["total_irrigated_area"] / area
            if area > 0 else 0
        )

        ## Model Input
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

        ## ML Model Prediction

        predicted_risk = loaded_model.predict(
            sample_input
        )[0]


        ## Historical Risk Variables

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


        ## Yield Risk

        if (
            pd.notna(previous_yield)
            and pd.notna(historical_yield)
            and historical_yield > 0
        ):

            yield_gap = (
                (historical_yield - previous_yield)
                / historical_yield
            ) * 100

            yield_risk = max(
                0,
                min(yield_gap, 100)
            )

        else:

            yield_risk = 0


        ## Rainfall Risk

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


        ## Irrigation Risk

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


        ## Area Change Risk

        previous_area = selected_data[
            "previous_year_area"
        ]

        if (
            pd.notna(previous_area)
            and previous_area > 0
        ):

            area_change_percentage = (
                abs(area - previous_area)
                / previous_area
            ) * 100

            area_risk = max(
                0,
                min(area_change_percentage, 100)
            )

        else:

            area_risk = 0


        ## Final Estimated Risk Probability

        estimated_risk_probability = (
            0.50 * yield_risk
            + 0.20 * rainfall_risk
            + 0.20 * irrigation_risk
            + 0.10 * area_risk
        )

        estimated_risk_probability = max(
            0,
            min(estimated_risk_probability, 100)
        )


        ## Contributing Factors

        # Rainfall
        if rainfall_deviation < -20:

            rainfall_factor = "High contribution"

        elif rainfall_deviation < 0:

            rainfall_factor = "Medium contribution"

        else:

            rainfall_factor = "Low contribution"


        # Historical instability
        if yield_risk > 30:

            historical_factor = "High contribution"

        elif yield_risk > 15:

            historical_factor = "Medium contribution"

        else:

            historical_factor = "Low contribution"


        # Irrigation
        if irrigation_change < 0:

            irrigation_factor = "Medium contribution"

        else:

            irrigation_factor = "Low contribution"


        # Yield trend
        if selected_data["yield_trend"] < 0:

            trend_factor = "High contribution"

        elif selected_data["yield_trend"] == 0:

            trend_factor = "Medium contribution"

        else:

            trend_factor = "Low contribution"


        ## CONTENT-BASED CROP RECOMMENDATION

        # Candidate crops from the same district and season
        candidate_crops = df[
            (df["district_name"] == district_name) &
            (df["season"] == season)
        ]["crop_name"].dropna().unique()


        # Features representing crop characteristics
        recommendation_features = [
            "historical_avg_yield",
            "normal_rainfall",
            "rainfall_deviation",
            "historical_avg_irrigation",
            "previous_year_area"
        ]


        # Create Crop Profiles

        crop_profiles = []

        for candidate_crop in candidate_crops:

            # Do not recommend the crop already selected
            if candidate_crop == crop_name:
                continue


            candidate_data = df[
                (df["district_name"] == district_name) &
                (df["crop_name"] == candidate_crop) &
                (df["season"] == season)
            ].sort_values("year")


            if candidate_data.empty:
                continue


            # Latest historical record
            candidate_data = candidate_data.iloc[-1]


            profile = {
                "crop": candidate_crop
            }


            for feature in recommendation_features:

                profile[feature] = candidate_data[feature]


            crop_profiles.append(profile)


        crop_profiles_df = pd.DataFrame(
            crop_profiles
        )


        # Check Recommendation Availability

        if crop_profiles_df.empty:

            recommended_crop_name = "Not Available"

            recommendation_score = 0

            similarity_score = 0

            recommended_crop_risk = 0

            recommended_crop_level = "N/A"

        else:

            # Handle Missing Values

            crop_profiles_df[
                recommendation_features
            ] = crop_profiles_df[
                recommendation_features
            ].fillna(
                crop_profiles_df[
                    recommendation_features
                ].median()
            )


            # Create User Profile

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


            # Normalize Features

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

            # Calculate Cosine Similarity

            similarity_scores = cosine_similarity(
                user_scaled,
                candidate_scaled
            )[0]


            crop_profiles_df[
                "similarity_score"
            ] = similarity_scores * 100


            # Calculate Recommendation Results

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
                        min(candidate_yield_risk, 100)
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
                        min(candidate_irrigation_risk, 100)
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
                            area -
                            candidate_previous_area
                        )
                        /
                        candidate_previous_area
                    ) * 100

                    candidate_area_risk = max(
                        0,
                        min(candidate_area_risk, 100)
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
                    min(candidate_risk_score, 100)
                )

                # Risk Level

                if candidate_risk_score < 33:

                    candidate_risk_level = "Low Risk"

                elif candidate_risk_score < 66:

                    candidate_risk_level = "Moderate Risk"

                else:

                    candidate_risk_level = "High Risk"


                # Content-Based Recommendation Score

                similarity_score = (
                    row["similarity_score"]
                )


                # Higher similarity is better.
                # Lower agricultural risk is better.

                recommendation_score = (
                    0.70 * similarity_score
                    +
                    0.30 * (
                        100 -
                        candidate_risk_score
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


            # Select Best Recommended Crop

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

                recommended_crop_name = "Not Available"

                recommendation_score = 0

                similarity_score = 0

                recommended_crop_risk = 0

                recommended_crop_level = "N/A"


        # Final Output

        print(
            "District:",
            district_name.title()
        )

        print(
            "Crop:",
            crop_name.title()
        )

        print(
            "Season:",
            season.title()
        )

        print(
            "Area:",
            area,
            "hectares"
        )

        print()


        # Classification Output

        print(
            "Predicted Risk:",
            predicted_risk.upper()
        )

        print()

        print(
            "Risk Probability:",
            round(
                estimated_risk_probability,
                0
            ),
            "%"
        )

        print()

        print(
            "Main contributing factors:"
        )

        print(
            "    Rainfall condition       →",
            rainfall_factor
        )

        print(
            "    Historical instability   →",
            historical_factor
        )

        print(
            "    Irrigation condition     →",
            irrigation_factor
        )

        print(
            "    Historical yield trend   →",
            trend_factor
        )

        print()


        # Content-Based Recommendation Output

        print(
            "🏆 Recommended Crop:"
        )

        print(
            "    Crop →",
            recommended_crop_name.title()
        )

        '''print(
            "    Recommendation Score →",
            f"{recommendation_score:.1f}%"
        )

        print(
            "    Similarity Score →",
            f"{similarity_score:.1f}%"
        )
'''
        print(
            "    Risk →",
            f"{recommended_crop_risk:.1f}%"
        )

        print(
            "    Risk Level →",
            recommended_crop_level.upper()
        )


# In[ ]:


predict_button.on_click(
    predict_risk
)


# In[ ]:


if len(districts) > 0:

    district_dropdown.value = districts[0]

    update_crops({
        "new": districts[0]
    })


# In[ ]:


display(
    district_dropdown,
    crop_dropdown,
    season_dropdown,
    area_input,
    predict_button,
    output
)


# In[ ]:




