from flask import Flask, render_template, request, jsonify
import cv2
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'models/logisticRegression.joblib')
model = joblib.load('logisticRegression.joblib')

# Function to preprocess the input image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (416, 416))
    img = img.reshape(1, -1)
    return img

# Mapping numerical predictions to class labels
class_labels = {
    0: 'Wheelchair',
    1: 'Crutch'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image from the request
        image = request.files['image']
        image_path = 'uploaded_image.jpg'
        image.save(image_path)

        # Preprocess the image
        processed_image = preprocess_image(image_path)

        # Make predictions
        prediction = int(model.predict(processed_image)[0])

        # Map numerical prediction to class label
        predicted_class = class_labels[prediction]

        # Return the prediction as JSON
        return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
