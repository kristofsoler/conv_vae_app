from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import preprocess  # Optional preprocessing module

# Initialize the Flask app
app = Flask(__name__)

# Load the model
MODEL_PATH = '../model/model.h5'  # Adjust the path if needed
model = load_model(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.get_json()

        # Extract input and preprocess if needed
        input_data = np.array(data['input'])  # Example input format: [[...]]
        
        # Optional: preprocess input_data if required by the model
        # input_data = preprocess.your_function(input_data)

        # Make prediction
        predictions = model.predict(input_data).tolist()

        # Return the predictions as JSON
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)