from flask import Flask, render_template, request
from joblib import load
import numpy as np

app = Flask(__name__)

# Load machine learning models
clf_m = load('rf_m.joblib')
clf_f = load('rf_f.joblib')

# Load label encoders
label_encoders = load('label_encoders.joblib')

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input data from the form
        state_code = int(request.form['state_code'])
        age_group = label_encoders['Age-group'].transform([request.form['age_group']])[0]
        disability = label_encoders['Disability'].transform([request.form['disability']])[0]

        # Create a feature vector
        feature_vector = np.array([[state_code, disability, age_group]])

        # Use machine learning models for prediction
        prediction_m = clf_m.predict(feature_vector)[0]
        prediction_f = clf_f.predict(feature_vector)[0]

        # Render the result template with predictions
        return render_template('result.html', prediction_m=prediction_m, prediction_f=prediction_f)

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return render_template('result.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
