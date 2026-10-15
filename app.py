from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

app = FastAPI(title="Student Placement Prediction API")

# Model aur columns wali files ka path
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "placement_model.joblib"
COLUMNS_PATH = BASE_DIR / "columns.joblib"

# Model load karna
model = joblib.load(MODEL_PATH)
saved_columns = joblib.load(COLUMNS_PATH)

# Model ke original feature names ko priority dena
if hasattr(model, "feature_names_in_"):
    columns = list(model.feature_names_in_)
else:
    columns = list(saved_columns)

print("App directory:", BASE_DIR)
print("Model path:", MODEL_PATH)
print("Columns path:", COLUMNS_PATH)
print("Total features:", len(columns))


# Input data ka structure
class StudentInput(BaseModel):
    gender: str
    study_hours: float
    attendance: float
    sleep_hours: float
    internet_usage: float
    assignments_completed: int
    previous_score: float
    extracurricular: str
    exam_score: float
    branch: str


@app.get("/")
def home():
    return {
        "message": "Student Placement Prediction API is running"
    }


@app.post("/predict")
def predict_placement(data: StudentInput):
    try:
        # Numerical aur categorical input features
        input_data = {
            "gender": (
                1 if data.gender.strip().lower() in ["male", "m"]
                else 0
            ),
            "study_hours": data.study_hours,
            "attendance": data.attendance,
            "sleep_hours": data.sleep_hours,
            "internet_usage": data.internet_usage,
            "assignments_completed": data.assignments_completed,
            "previous_score": data.previous_score,
            "extracurricular": (
                1 if data.extracurricular.strip().lower() == "yes"
                else 0
            ),
            "exam_score": data.exam_score,
        }

        # Sabhi model features ko zero se initialize karna
        row = {column: 0 for column in columns}

        # Numerical features set karna
        for feature, value in input_data.items():
            if feature in row:
                row[feature] = value

        # Branch category handle karna
        branch = data.branch.strip().lower()

        branch_columns = [
            column for column in columns
            if column.lower().startswith("branch_")
        ]

        matching_columns = [
            column for column in branch_columns
            if column.split("_", 1)[1].strip().lower() == branch
        ]

        if matching_columns:
            # Sirf ek matching branch feature ko 1 karna
            row[matching_columns[0]] = 1

        # Model ke exact feature order mein DataFrame banana
        input_df = pd.DataFrame([row])
        input_df = input_df.reindex(columns=columns, fill_value=0)

        # Prediction
        prediction = model.predict(input_df)[0]

        # Response
        if int(prediction) == 1:
            result = "Placed"
        else:
            result = "Not Placed"

        return {
            "prediction": result,
            "input_branch": data.branch
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )