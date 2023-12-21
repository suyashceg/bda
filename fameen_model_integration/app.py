from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
app = Flask(__name__)

# Load the model and transformers
model = joblib.load('disability_model.joblib')
cat_coder = joblib.load('cat_coder.joblib')
label_transformer = joblib.load('label_transformer.joblib')
df = pd.read_csv('processed.csv')
#inputs to take
# attr = [['TAMIL NADU', 'Rural', '0-14']]
#filtering for respective population 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        area_name = request.form.get('area_name')
        rural_urban = request.form.get('rural_urban')
        age_group = request.form.get('age_group')

        # Transform input data using the categorical encoder
        input_data = cat_coder.transform([[area_name, rural_urban, age_group]]).toarray()
        filtered_df = df[
        (df['Area Name'] == area_name) &
        (df['Total/ Rural/Urban'] == rural_urban) &
        (df['Age-group'] == age_group)
        ]
        # summing up the respective population
        population = sum(filtered_df['Total disabled population - Persons'])
        input_data = np.append(input_data,population)
        # Make a prediction using the model
        prediction = model.predict([input_data])[0]

        # Transform the predicted label back to the original format
        predicted_label = label_transformer.inverse_transform([prediction])[0]

        return render_template('result.html', prediction=predicted_label)

    except Exception as e:
        return render_template('error.html', error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True)
