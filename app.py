
import joblib
import pandas as pd

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

# --------------------------------------------------
# Application setup
# --------------------------------------------------

app = FastAPI(
    title="Student Placement Prediction API",
    description="Predict student placement using a trained Decision Tree model.",
    version="1.0.0"
)

BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------
# Load model and training feature columns
# --------------------------------------------------

try:
    model = joblib.load(BASE_DIR / "placement_model.joblib")
    columns = joblib.load(BASE_DIR / "columns.joblib")
    columns = list(columns)

except Exception as e:
    raise RuntimeError(f"Failed to load model or columns: {e}")


# --------------------------------------------------
# Input schema
# --------------------------------------------------

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


# --------------------------------------------------
# Home route
# --------------------------------------------------

@app.get("/")
def home():
    index_file = BASE_DIR / "templates" / "index.html"

    if not index_file.exists():
        return {
            "message": "Student Placement Prediction API is running",
            "docs": "/docs",
            "health": "/health"
        }

    return FileResponse(index_file)


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------
# Prediction route
# --------------------------------------------------

@app.post("/predict")
def predict(data: StudentInput):

    try:
        # Encode categorical inputs
        gender = (
            1
            if data.gender.strip().lower() in ["male", "m"]
            else 0
        )

        extracurricular = (
            1
            if data.extracurricular.strip().lower() == "yes"
            else 0
        )

        branch = data.branch.strip().upper()

        # Build the original input features
        row = {
            "gender": gender,
            "study_hours": data.study_hours,
            "attendance": data.attendance,
            "sleep_hours": data.sleep_hours,
            "internet_usage": data.internet_usage,
            "assignments_completed": data.assignments_completed,
            "previous_score": data.previous_score,
            "extracurricular": extracurricular,
            "exam_score": data.exam_score,
        }

        # Identify branch columns while ignoring extra spaces
        branch_columns = [
            col for col in columns
            if str(col).strip().lower().startswith("branch_")
        ]

        # Set one-hot encoded branch values
        available_branches = []

        for col in branch_columns:
            category = (
                str(col).strip()[len("branch_"):]
                .strip().upper()
            )

            available_branches.append(category)
            row[col] = 1 if category == branch else 0

        # Reject unknown branches rather than silently
        # predicting with an all-zero branch encoding
        if branch_columns and branch not in available_branches:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Unknown branch",
                    "provided_branch": data.branch,
                    "available_branches": sorted(
                        set(available_branches)
                    )
                }
            )

        # Create DataFrame and preserve EXACT training names
        df_row = pd.DataFrame([row])

        df_row = df_row.reindex(
            columns=columns,
            fill_value=0
        )

        # Critical: keep the original model feature names.
        # Never strip or rename the trained model's columns.
        if hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)

            # Verify the saved columns and model agree
            if columns != expected_features:
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "columns.joblib does not match the "
                        "feature names stored in the model. "
                        "Regenerate columns.joblib from the "
                        "original training DataFrame."
                    )
                )

            df_row = df_row.reindex(
                columns=expected_features,
                fill_value=0
            )

        # Predict
        prediction = model.predict(df_row)[0]

        # Return prediction
        return {
            "prediction": (
                "Placed" if int(prediction) == 1
                else "Not Placed"
            )
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
