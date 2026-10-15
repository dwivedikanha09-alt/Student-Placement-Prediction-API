
import joblib
import pandas as pd

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="Student Placement Prediction API")

# Get project directory
BASE_DIR = Path(__file__).resolve().parent

# Load trained model and feature columns
model = joblib.load(BASE_DIR / "placement_model.joblib")
columns = joblib.load(BASE_DIR / "columns.joblib")

# Normalize feature names while preserving the model's expected names
columns = list(columns)

class StudentInput(BaseModel):
    gender: str
    study_hours: float
    attendance: float
    sleep_hours: float
    internet_usage: float
    assignments_completed: float
    previous_score: float
    extracurricular: str
    exam_score: float
    branch: str


@app.get("/")
def home():
    index_file = BASE_DIR / "templates" / "index.html"

    if not index_file.exists():
        raise HTTPException(
            status_code=404,
            detail="templates/index.html not found"
        )

    return FileResponse(index_file)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: StudentInput):

    try:
        # Encode categorical values
        gender_value = (
            1 if data.gender.strip().lower() in ["male", "m"]
            else 0
        )

        extra_value = (
            1 if data.extracurricular.strip().lower() == "yes"
            else 0
        )

        branch_value = data.branch.strip().upper()

        # Create input features
        row = {
            "gender": gender_value,
            "study_hours": data.study_hours,
            "attendance": data.attendance,
            "sleep_hours": data.sleep_hours,
            "internet_usage": data.internet_usage,
            "assignments_completed": data.assignments_completed,
            "previous_score": data.previous_score,
            "extracurricular": extra_value,
            "exam_score": data.exam_score,
        }

        # Handle branch columns, including spaces in their names
        branch_columns = [
            col for col in columns
            if str(col).strip().lower().startswith("branch_")
        ]

        matched_branch = False

        for col in branch_columns:
            # Remove spaces around the feature name and category
            category = str(col).strip()[len("branch_"):].strip().upper()

            value = 1 if category == branch_value else 0
            row[col] = value

            if value == 1:
                matched_branch = True

        if branch_columns and not matched_branch:
            valid_branches = [
                str(col).strip()[len("branch_"):].strip()
                for col in branch_columns
            ]

            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Unknown branch. Check the branch value.",
                    "provided_branch": data.branch,
                    "valid_branches": valid_branches
                }
            )

        # Create DataFrame using the EXACT original model columns
        df_row = pd.DataFrame([row])

        df_row = df_row.reindex(
            columns=columns,
            fill_value=0
        )

        # Predict placement
        pred = model.predict(df_row)[0]

        return {
            "prediction": "Placed" if int(pred) == 1 else "Not Placed"
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
