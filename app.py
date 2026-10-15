import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

model   = joblib.load("placement_model.joblib")
columns = joblib.load("columns.joblib")

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

from fastapi.responses import FileResponse
@app.get("/")
def home():
    return FileResponse("templates/index.html")

@app.post("/predict")
def predict(data: StudentInput):
    gender = 1 if data.gender.lower() in ["male", "m"] else 0
    extra  = 1 if data.extracurricular.lower() == "yes" else 0
    branch = data.branch.strip().upper()

    row = {
        "gender":                gender,
        "study_hours":           data.study_hours,
        "attendance":            data.attendance,
        "sleep_hours":           data.sleep_hours,
        "internet_usage":        data.internet_usage,
        "assignments_completed": data.assignments_completed,
        "previous_score":        data.previous_score,
        "extracurricular":       extra,
        "exam_score":            data.exam_score,
    }
    for col in columns:
        if col.startswith("branch_"):
            row[col] = 1 if col == f"branch_{branch}" else 0

    df_row = pd.DataFrame([row]).reindex(columns=columns, fill_value=0)
    pred   = model.predict(df_row)[0]

    return {"prediction": "Placed" if pred == 1 else "Not Placed"}


