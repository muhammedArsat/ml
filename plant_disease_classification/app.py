from flask import Flask, request, render_template
import joblib
import numpy as np
import cv2
import os

# Load the trained model
model = joblib.load('svm.pkl')

# Initialize Flask app
app = Flask(__name__)

# Define a function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  # Resize to match the training size
    img = img.reshape(1, -1)  # Flatten the image
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded.'
    
    file = request.files['file']
    if file.filename == '':
        return 'No file selected.'
    
    # Save the uploaded file to a temporary location
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Preprocess the image
    processed_image = preprocess_image(file_path)

    # Make prediction
    prediction = model.predict(processed_image)

    # Delete the uploaded file after prediction
    os.remove(file_path)

    return 'Healthy' if prediction[0] == 0 else 'Diseased'

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    app.run(debug=True)
