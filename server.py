import numpy as np
import pandas as pd
from joblib import load
from flask import Flask, request, jsonify
from flask_expects_json import expects_json
from flask_cors import CORS

model = load('random_forest_model.pkl')

app = Flask(__name__)
CORS(app) 

schema = {
    "type": "object",
    "properties": {
        "age": {"type": "number"},
        "gender": {"type": "string", "enum": ["Male", "Female"]},
        "highBloodPressure": {"type": "boolean"},
        "heartDisease": {"type": "boolean"},
        "married": {"type": "boolean"},
        "jobType": {"type": "string", "enum": ["PRIVATE", "SELF_EMPLOYED", "GOVT_JOB", "CHILDREN", "NEVER_WORKED"]},
        "residenceType": {"type": "string", "enum": ["URBAN", "RURAL"]},
        "bloodSugarLevel": {"type": "number"},
        "bmi": {"type": "number"},
        "smokeType": {"type": "string", "enum": ["FORMERLY", "NEVER", "SMOKES", "UNKNOWN"]}
    },
    "required": ["age", "gender", "highBloodPressure", "heartDisease", "married", "jobType", "residenceType", "bloodSugarLevel", "bmi", "smokeType"]
}


def preprocess_responses(data):
    responses = {}
    responses['age'] = float(data.get("age"))
    responses['gender_Female'] = 1 if data.get("gender") == 'Female' else 0
    responses['gender_Male'] = 1 if data.get("gender") == 'Male' else 0
    responses['hypertension'] = int(data.get("highBloodPressure"))
    responses['heart_disease'] = int(data.get("heartDisease"))
    married = 'Yes' if data.get("married") else 'No'
    responses['ever_married_Yes'] = 1 if married == 'Yes' else 0
    responses['ever_married_No'] = 1 if married == 'No' else 0
    work_type = data.get("jobType").replace('_', ' ').title()
    for wt in ['Govt_job', 'Never_worked', 'Private', 'Self-employed', 'Children']:
        responses[f'work_type_{wt}'] = 1 if work_type == wt else 0
    residence_type = data.get("residenceType").title()
    responses['Residence_type_Urban'] = 1 if residence_type == 'Urban' else 0
    responses['Residence_type_Rural'] = 1 if residence_type == 'Rural' else 0
    responses['avg_glucose_level'] = float(data.get("bloodSugarLevel"))
    responses['bmi'] = float(data.get("bmi"))
    smoking_status = data.get("smokeType").replace(' ', '_').lower()
    for ss in ['formerly_smoked', 'never_smoked', 'smokes', 'unknown']:
        responses[f'smoking_status_{ss}'] = 1 if smoking_status == ss else 0
    return responses


def predict_stroke_probability(responses):
    columns_used = [
        'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
        'gender_Female', 'gender_Male', 'ever_married_No', 'ever_married_Yes',
        'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private',
        'work_type_Self-employed', 'work_type_children',
        'Residence_type_Rural', 'Residence_type_Urban',
        'smoking_status_Unknown', 'smoking_status_formerly smoked',
        'smoking_status_never smoked', 'smoking_status_smokes'
    ]
    df = pd.DataFrame([responses], columns=columns_used).fillna(0)
    probability = model.predict_proba(df)[0][1]
    return probability

@app.route('/predict', methods=['POST'])
@expects_json(schema)
def predict():
    data = request.json
    user_responses = preprocess_responses(data)
    stroke_probability = predict_stroke_probability(user_responses)
    return jsonify({"stroke_probability": f"{stroke_probability:.2%}"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
