from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load saved pipeline and label encoder
pipeline = joblib.load("best_xgb_pipeline.pkl")
le = joblib.load("label_encoder.pkl")

@app.route('/')
def home():
    return render_template("index.html", prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input from form
        data = {col: request.form[col] for col in request.form.keys()}
        
        # Convert numeric fields to int/float
        numeric_cols = [
            'Age', 'DistanceFromHome', 'EmpEducationLevel', 'EmpEnvironmentSatisfaction',
            'EmpHourlyRate', 'EmpJobInvolvement', 'EmpJobLevel', 'EmpJobSatisfaction',
            'NumCompaniesWorked', 'EmpLastSalaryHikePercent', 'EmpRelationshipSatisfaction',
            'TotalWorkExperienceInYears', 'TrainingTimesLastYear', 'EmpWorkLifeBalance',
            'ExperienceYearsAtThisCompany', 'ExperienceYearsInCurrentRole',
            'YearsSinceLastPromotion', 'YearsWithCurrManager'
        ]
        for col in numeric_cols:
            data[col] = int(data[col])

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Make prediction
        prediction_encoded = pipeline.predict(input_df)[0]
        prediction = le.inverse_transform([prediction_encoded])[0]

        return render_template("index.html", prediction=int(prediction))

    except Exception as e:
        return f"⚠️ Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
# To run the app, use the command: python app.py