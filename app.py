from flask import Flask, request, render_template, redirect, url_for, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
import base64
from io import BytesIO
from PIL import Image
from werkzeug.utils import secure_filename

# Load your pre-trained model
model = tf.keras.models.load_model('train.h5')

# Define class names
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
               'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
               'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del',
               'nothing', 'space']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def preprocess_image(image_data):
    # Decode the image
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image = image.convert('RGB')
    image = np.array(image)
    
    # Preprocess the image for the model
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    
    return image

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/detect')
def detect():
    return render_template('camera.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contatc():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = data['image'].split(",")[1]
    
    # Preprocess the image
    image = preprocess_image(image_data)
    
    # Predict the class
    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]

    return jsonify({'predicted_class': predicted_class})

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image
        image = cv2.imread(filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = tf.keras.applications.mobilenet.preprocess_input(image)
        image = np.expand_dims(image, axis=0)

        # Predict the class
        predictions = model.predict(image)
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_names[predicted_class_index]

        # Remove the file after processing
        os.remove(filepath)

        return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
