from flask import Flask, render_template, request, jsonify  # Add jsonify
from flask_cors import CORS
import numpy as np
import pickle

# Loading model
model = pickle.load(open('./notebook and dataset/model1.pkl', 'rb'))

# Flask app
app = Flask(__name__)

CORS(app)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from the form
        features = request.form['feature']
        print("Raw Input:", features)  # Debugging: Print raw input

        # Split the input string into a list of strings
        features = features.split(',')
        print("Split Features:", features)  # Debugging: Print split features

        # Convert to float and handle errors
        np_features = np.array([float(x.strip()) for x in features])
        print("Numeric Features:", np_features)  # Debugging: Print numeric features

        # Check the number of features
        if len(np_features) != 31:  # Replace 30 with the expected number of features
            print("Incorrect length input")
            return jsonify({'prediction': f'Invalid input! Expected 30 features, but got {len(np_features)}.'})

        # Prediction
        pred = model.predict(np_features.reshape(1, -1))
        print("Prediction:", pred)  # Debugging: Print prediction

        # Prepare the message
        # prediction_result = 'Cancerous' if pred[0] == 1 else 'Not Cancerous'
        # return jsonify({'prediction': prediction_result})  # Return JSON response
        return jsonify({'prediction': 'Cancerous' if pred[0] == 1 else 'Not Cancerous'})
    except ValueError as e:
        # Debugging: Print error
        print("Error:", e)
        return jsonify({'prediction': 'Invalid input! Please enter only numerical values.'})  # Return JSON response
    
if __name__ == '__main__':
    app.run(debug=True)