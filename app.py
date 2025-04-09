from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

# Load the Random Forest model
MODEL_PATH = 'C:/Users/chava/OneDrive/Desktop/ADIDAS SALES/rf_model.pkl'

def load_model():
    try:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        retailer_id = int(request.form['retailer_id'])
        price_per_unit = float(request.form['price_per_unit'])
        units_sold = int(request.form['units_sold'])
        operating_profit = float(request.form['operating_profit'])
        operating_margin = float(request.form['operating_margin'])
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])
        
        # Create features array for prediction
        # Adjust this according to your model's expected input
        features = np.array([[retailer_id, price_per_unit, units_sold, 
                             operating_profit, operating_margin, year, month, day]])
        
        # Load model
        model = load_model()
        if model is None:
            return jsonify({'error': 'Failed to load model'}), 500
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Return prediction
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'input_data': {
                'retailer_id': retailer_id,
                'price_per_unit': price_per_unit,
                'units_sold': units_sold,
                'operating_profit': operating_profit,
                'operating_margin': operating_margin,
                'date': f"{year}-{month:02d}-{day:02d}"
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    app.run(debug=True)