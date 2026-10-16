from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import joblib

app = FastAPI(title="Student Placement Prediction")

# Load the trained model and feature columns
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "placement_model.joblib")
saved_columns = joblib.load(BASE_DIR / "columns.joblib")

if hasattr(model, "feature_names_in_"):
    columns = list(model.feature_names_in_)
else:
    columns = list(saved_columns)


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


# Website homepage
@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Placement Prediction</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f1f5f9;
            color: #172033;
        }

        header {
            background: linear-gradient(135deg, #172554, #2563eb);
            color: white;
            text-align: center;
            padding: 38px 16px;
        }

        header h1 {
            margin: 0 0 12px;
            font-size: 30px;
        }

        header p {
            margin: 0;
            color: #dbeafe;
        }

        .container {
            max-width: 850px;
            margin: 30px auto;
            padding: 0 16px;
        }

        .card {
            background: white;
            padding: 26px;
            border-radius: 16px;
            box-shadow: 0 8px 28px rgba(15, 23, 42, 0.08);
        }

        .card h2 {
            margin-top: 0;
            margin-bottom: 8px;
        }

        .description {
            color: #64748b;
            margin-bottom: 24px;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 18px;
        }

        .field {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        label {
            font-size: 14px;
            font-weight: bold;
        }

        input, select {
            width: 100%;
            padding: 12px;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            font-size: 15px;
            background: white;
        }

        input:focus, select:focus {
            outline: 2px solid #93c5fd;
            border-color: #2563eb;
        }

        .full-width {
            grid-column: 1 / -1;
        }

        button {
            width: 100%;
            margin-top: 24px;
            padding: 14px;
            border: none;
            border-radius: 9px;
            background: #2563eb;
            color: white;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
        }

        button:hover {
            background: #1d4ed8;
        }

        button:disabled {
            opacity: 0.6;
            cursor: wait;
        }

        #result {
            display: none;
            margin-top: 22px;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }

        #result h2 {
            margin: 0 0 8px;
        }

        #result p {
            margin-bottom: 0;
        }

        .success {
            display: block !important;
            background: #dcfce7;
            color: #166534;
        }

        .not-placed {
            display: block !important;
            background: #ffedd5;
            color: #9a3412;
        }

        .error {
            display: block !important;
            background: #fee2e2;
            color: #991b1b;
        }

        footer {
            text-align: center;
            padding: 24px 12px;
            color: #64748b;
            font-size: 13px;
        }

        @media (max-width: 600px) {
            .form-grid {
                grid-template-columns: 1fr;
            }

            .full-width {
                grid-column: auto;
            }

            header h1 {
                font-size: 25px;
            }

            .card {
                padding: 20px;
            }
        }
    </style>
</head>

<body>
    <header>
        <h1>Student Placement Prediction</h1>
        <p>Enter your details to get a model-based placement prediction.</p>
    </header>

    <main class="container">
        <section class="card">
            <h2>Student Details</h2>
            <p class="description">
                Fill in all the fields below and click Predict Placement.
            </p>

            <form id="predictionForm">
                <div class="form-grid">

                    <div class="field">
                        <label for="gender">Gender</label>
                        <select id="gender" required>
                            <option value="Female">Female</option>
                            <option value="Male">Male</option>
                        </select>
                    </div>

                    <div class="field">
                        <label for="branch">Branch</label>
                        <select id="branch" required>
                            <option value="CSE">CSE</option>
                            <option value="CIVIL">Civil</option>
                            <option value="COMMERCE">Commerce</option>
                            <option value="ECE">ECE</option>
                            <option value="IT">IT</option>
                            <option value="MECHANICAL">Mechanical</option>
                        </select>
                    </div>

                    <div class="field">
                        <label for="study_hours">Daily Study Hours</label>
                        <input id="study_hours" type="number"
                               min="0" max="24" step="any" required>
                    </div>

                    <div class="field">
                        <label for="attendance">Attendance (%)</label>
                        <input id="attendance" type="number"
                               min="0" max="100" step="any" required>
                    </div>

                    <div class="field">
                        <label for="sleep_hours">Daily Sleep Hours</label>
                        <input id="sleep_hours" type="number"
                               min="0" max="24" step="any" required>
                    </div>

                    <div class="field">
                        <label for="internet_usage">Internet Usage</label>
                        <input id="internet_usage" type="number"
                               min="0" step="any" required>
                    </div>

                    <div class="field">
                        <label for="assignments_completed">
                            Assignments Completed
                        </label>
                        <input id="assignments_completed" type="number"
                               min="0" step="1" required>
                    </div>

                    <div class="field">
                        <label for="previous_score">Previous Score</label>
                        <input id="previous_score" type="number"
                               min="0" max="100" step="any" required>
                    </div>

                    <div class="field">
                        <label for="exam_score">Exam Score</label>
                        <input id="exam_score" type="number"
                               min="0" max="100" step="any" required>
                    </div>

                    <div class="field">
                        <label for="extracurricular">
                            Extracurricular Activities
                        </label>
                        <select id="extracurricular" required>
                            <option value="Yes">Yes</option>
                            <option value="No">No</option>
                        </select>
                    </div>

                </div>

                <button id="predictButton" type="submit">
                    Predict Placement
                </button>
            </form>

            <div id="result" role="status" aria-live="polite"></div>
        </section>
    </main>

    <footer>
        Student Placement Prediction | Machine Learning Project
        <br>
        This prediction is an estimate, not a guarantee of employment.
    </footer>

    <script>
        const form = document.getElementById("predictionForm");
        const button = document.getElementById("predictButton");
        const resultBox = document.getElementById("result");

        form.addEventListener("submit", async function(event) {
            event.preventDefault();

            const data = {
                gender: document.getElementById("gender").value,
                branch: document.getElementById("branch").value,
                study_hours: Number(
                    document.getElementById("study_hours").value
                ),
                attendance: Number(
                    document.getElementById("attendance").value
                ),
                sleep_hours: Number(
                    document.getElementById("sleep_hours").value
                ),
                internet_usage: Number(
                    document.getElementById("internet_usage").value
                ),
                assignments_completed: Number(
                    document.getElementById("assignments_completed").value
                ),
                previous_score: Number(
                    document.getElementById("previous_score").value
                ),
                exam_score: Number(
                    document.getElementById("exam_score").value
                ),
                extracurricular:
                    document.getElementById("extracurricular").value
            };

            button.disabled = true;
            button.textContent = "Predicting...";
            resultBox.className = "";
            resultBox.style.display = "block";
            resultBox.textContent = "Please wait while the model predicts.";

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(data)
                });

                const output = await response.json();

                if (!response.ok) {
                    throw new Error(
                        output.detail || "Prediction failed."
                    );
                }

                if (output.prediction === "Placed") {
                    resultBox.className = "success";
                    resultBox.innerHTML =
                        "<h2>Prediction: Placed</h2>" +
                        "<p>The model predicts a placement outcome of Placed.</p>";
                } else {
                    resultBox.className = "not-placed";
                    resultBox.innerHTML =
                        "<h2>Prediction: Not Placed</h2>" +
                        "<p>The model predicts a placement outcome of Not Placed.</p>";
                }

            } catch (error) {
                resultBox.className = "error";
                resultBox.textContent =
                    "Error: " + error.message +
                    " Please try again or check the server logs.";
            } finally {
                button.disabled = false;
                button.textContent = "Predict Placement";
            }
        });
    </script>
</body>
</html>
""")


# Prediction API
@app.post("/predict")
def predict_placement(data: StudentInput):
    try:
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

        # Initialize every model feature
        row = {column: 0 for column in columns}

        for feature, value in input_data.items():
            if feature in row:
                row[feature] = value

        # Match the submitted branch with the saved feature columns
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
            row[matching_columns[0]] = 1

        input_df = pd.DataFrame([row])
        input_df = input_df.reindex(columns=columns, fill_value=0)

        prediction = model.predict(input_df)[0]

        result = "Placed" if int(prediction) == 1 else "Not Placed"

        return {
            "prediction": result,
            "input_branch": data.branch
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))