from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('brain_tumor_model.h5')

# Define the image size expected by the model
IMG_SIZE = (128, 128)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    # Get the uploaded image
    file = request.files['file']
    
    # Open the image and resize to 128x128
    img = Image.open(file.stream)
    img = img.resize(IMG_SIZE)  # Resize to the expected shape (128x128)
    
    # Convert to RGB (in case the image is grayscale)
    img = img.convert('RGB')
    
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0  # Normalizing pixel values to [0, 1]
    
    # Add an extra dimension to match the input shape (1, 128, 128, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Get prediction from the model
    prediction = model.predict(img_array)
    
    # Handle the result (e.g., display the prediction)
    if prediction[0] > 0.5:
        result = "Brain Tumor Detected"
    else:
        result = "No Brain Tumor"
    
    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
