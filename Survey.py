import numpy as np
import pandas as pd
from joblib import load

# Load the trained model
model = load('random_forest_model.pkl')

# Function to collect user responses and preprocess them
def collect_responses():
    responses = {}

    print("뇌졸중 진단 설문조사를 시작합니다.")
    print("질문에 답변을 순서대로 입력해주세요.")

    responses['age'] = float(input("1. 나이는 몇 살인가요? (숫자 입력): "))

    gender = input("2. 성별을 선택해주세요 (Male/Female): ").strip().title()
    responses['gender_Female'] = 1 if gender == 'Female' else 0
    responses['gender_Male'] = 1 if gender == 'Male' else 0

    responses['hypertension'] = int(input("3. 고혈압이 있나요? (1: 예, 0: 아니요): "))
    responses['heart_disease'] = int(input("4. 심장 질환 이력이 있나요? (1: 예, 0: 아니요): "))

    married = input("5. 결혼 여부를 선택해주세요 (Yes/No): ").strip().title()
    responses['ever_married_Yes'] = 1 if married == 'Yes' else 0
    responses['ever_married_No'] = 1 if married == 'No' else 0

    work_type = input("6. 현재 직업 유형을 선택해주세요 (Private/Self-employed/Govt_job/Children/Never_worked): ").strip().title()
    for wt in ['Govt_job', 'Never_worked', 'Private', 'Self-employed', 'Children']:
        responses[f'work_type_{wt}'] = 1 if work_type == wt else 0

    residence_type = input("7. 거주 유형을 선택해주세요 (Urban/Rural): ").strip().title()
    responses['Residence_type_Urban'] = 1 if residence_type == 'Urban' else 0
    responses['Residence_type_Rural'] = 1 if residence_type == 'Rural' else 0

    responses['avg_glucose_level'] = float(input("8. 평균 혈당 수치는 얼마인가요? (숫자 입력): "))
    responses['bmi'] = float(input("9. BMI 지수는 얼마인가요? (숫자 입력): "))

    smoking_status = input("10. 흡연 상태를 선택해주세요 (Formerly smoked/Never smoked/Smokes/Unknown): ").strip().title()
    for ss in ['Formerly smoked', 'Never smoked', 'Smokes', 'Unknown']:
        responses[f'smoking_status_{ss}'] = 1 if smoking_status == ss else 0

    return responses

# Function to predict stroke probability
def predict_stroke_probability(responses):
    # Load the columns used in training the model
    columns_used = [
        'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
        'gender_Female', 'gender_Male', 'ever_married_No', 'ever_married_Yes',
        'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private',
        'work_type_Self-employed', 'work_type_children',
        'Residence_type_Rural', 'Residence_type_Urban',
        'smoking_status_Unknown', 'smoking_status_formerly smoked',
        'smoking_status_never smoked', 'smoking_status_smokes'
    ]

    # Create a DataFrame with user responses
    df = pd.DataFrame([responses], columns=columns_used).fillna(0)

    # Make predictions
    probability = model.predict_proba(df)[0][1]
    return probability

# Main function to run the program
def main():
    # Collect user responses
    user_responses = collect_responses()

    # Predict stroke probability
    stroke_probability = predict_stroke_probability(user_responses)

    print(f"뇌졸중 발병 가능성: {stroke_probability:.2%}")

if __name__ == "__main__":
    main()
