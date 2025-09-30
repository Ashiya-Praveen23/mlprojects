from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

## Route for a home page
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Match field names with HTML form
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parentEdu'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('testPrep'),
                reading_score=float(request.form.get('reading')),
                writing_score=float(request.form.get('writing'))
            )

            # Convert to DataFrame
            pred_df = data.get_data_as_data_frame()
            print("Input DataFrame:\n", pred_df)

            # Run prediction
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            # Return prediction on the same page
            return render_template('home.html', results=results[0])

        except Exception as e:
            print("Error during prediction:", e)
            return render_template('home.html', results="Error in prediction")

if __name__ == "__main__":
    app.run(host="0.0.0.0")
     