# CropGuard AI API

CropGuard AI is a FastAPI project with two machine-learning services for
agriculture:

1. Crop yield prediction using a regression model.
2. Agricultural risk classification using a classification model.

Both services use historical agriculture data to find the latest matching
record for the selected district, crop, and season.

## Deployed Backend URLs

- [CropGuard AI backend](https://cropguard-ai-vurl.onrender.com/)
- [CropGuard AI API backend](https://cropguard-ai-api.onrender.com/)

FastAPI Swagger documentation is available at `/docs` on each deployment:

- [Backend 1 API docs](https://cropguard-ai-vurl.onrender.com/docs)
- [Backend 2 API docs](https://cropguard-ai-api.onrender.com/docs)

## Model URLs

- [Crop yield prediction model](https://github.com/Rajender1709Reddy/CropGuard-AI-API/blob/master/CropGuard_Reg/cropguard_yield_model.pkl)
- [Crop risk classification model](https://github.com/Rajender1709Reddy/CropGuard-AI-API/blob/master/CroupGuard_Classification/crop_risk_classification_model.pkl)

## Repository Structure

```text
CropGuard_Reg/
	app.py                         Yield prediction FastAPI service
	cropguard_yield_model.pkl      Trained yield regression model
	Final_dataset_clean.csv        Historical agriculture dataset
	requirements.txt               Python dependencies

CroupGuard_Classification/
	app.py                         Agricultural risk FastAPI service
	ML_Task_Classification.py      Classification model workflow
	crop_risk_classification_model.pkl
																 Trained risk classification model
```

The classification folder is named `CroupGuard_Classification` in the
repository. The spelling is retained so existing deployment paths continue to
work.

## API Endpoints

### Yield Prediction API

Source: [`CropGuard_Reg/app.py`](./CropGuard_Reg/app.py)

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Confirms that the yield service is running. |
| `GET` | `/health` | Reports whether the model and dataset loaded successfully. |
| `POST` | `/predict` | Predicts yield and estimates total production. |

Request body for `POST /predict`:

```json
{
	"district_name": "Nalgonda",
	"crop_name": "Rice",
	"season": "Kharif",
	"area": 2.5
}
```

The `area` value is measured in hectares and must be greater than zero. The
service uses the latest historical record matching the district, crop, and
season, then returns the predicted yield in tonnes per hectare and estimated
total production in tonnes.

Example response:

```json
{
	"prediction": 4.25,
	"unit": "tonnes/hectare",
	"estimated_total_production": 10.63,
	"production_unit": "tonnes",
	"district": "Nalgonda",
	"crop": "Rice",
	"season": "Kharif",
	"farmer_area": 2.5,
	"area_unit": "hectares",
	"historical_reference_year": "2022-23"
}
```

### Agricultural Risk Classification API

Source: [`CroupGuard_Classification/app.py`](./CroupGuard_Classification/app.py)

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Confirms that the risk service is running. |
| `GET` | `/health` | Reports whether the model and dataset loaded successfully. |
| `POST` | `/predict` | Classifies agricultural risk and gives recommendations. |

Request body for `POST /predict`:

```json
{
	"district_name": "Karimnagar",
	"crop_name": "Rice",
	"season": "Kharif",
	"area": 2.5
}
```

The classification service calculates features such as area change,
irrigation coverage, rainfall conditions, yield trend, and historical
stability. It returns the predicted risk, risk probability, contributing
factors, and a recommended crop when one is available.

Important response fields include:

- `predicted_risk`: The model's risk class.
- `risk_probability`: Estimated risk probability as a percentage.
- `main_contributing_factors`: Rainfall, historical, irrigation, and trend factors.
- `recommended_crop`: Suggested crop, risk level, and recommendation score.

## Running Locally

Use Python 3.10 or later and create a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r CropGuard_Reg/requirements.txt
```

Start the yield service:

```powershell
cd CropGuard_Reg
uvicorn app:app --reload --port 8000
```

Start the classification service in a second terminal:

```powershell
cd CroupGuard_Classification
uvicorn app:app --reload --port 8001
```

After startup, open `http://127.0.0.1:8000/docs` or
`http://127.0.0.1:8001/docs` to use Swagger UI.

## Example Request With cURL

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
	-H "Content-Type: application/json" \
	-d '{
		"district_name": "Nalgonda",
		"crop_name": "Rice",
		"season": "Kharif",
		"area": 2.5
	}'
```

Replace port `8000` with port `8001` to call the classification service.

## Error Handling

- `404`: No historical record matches the selected district, crop, and season.
- `500`: A required model or dataset is unavailable, or prediction fails.
- `422`: The request body does not satisfy the API schema, such as a missing
	field or an `area` value less than or equal to zero.
