from flask import Flask, render_template, request, jsonify, send_file
import pickle
import numpy as np
import os
from datetime import datetime
import json
import csv
import io

app = Flask(__name__)

# Load the HMM model
MODEL_PATH = 'C:/Users/chava/OneDrive/Desktop/ADIDAS SALES/models/hmm_model.pkl'
# Path to store prediction history
HISTORY_PATH = 'C:/Users/chava/OneDrive/Desktop/ADIDAS SALES/history/predictions_history.json'

def load_model():
    try:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def save_prediction(prediction_data):
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
        
        # Add timestamp to the prediction data
        prediction_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Load existing history or create new
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, 'r') as file:
                history = json.load(file)
        else:
            history = []
        
        # Add new prediction to history
        history.append(prediction_data)
        
        # Save updated history
        with open(HISTORY_PATH, 'w') as file:
            json.dump(history, file, indent=4)
            
    except Exception as e:
        print(f"Error saving prediction history: {e}")

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
        features = np.array([[retailer_id, price_per_unit, units_sold, 
                             operating_profit, operating_margin, year, month, day]])
        
        # Load model
        model = load_model()
        if model is None:
            return jsonify({'error': 'Failed to load model'}), 500
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Prepare response data
        response_data = {
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
        }
        
        # Save prediction to history
        save_prediction(response_data)
        
        # Return prediction
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/history')
def get_history():
    try:
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, 'r') as file:
                history = json.load(file)
            return jsonify(history)
        else:
            return jsonify([])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download-history')
def download_history():
    try:
        if not os.path.exists(HISTORY_PATH):
            return jsonify({'error': 'No history available'}), 404
            
        # Get format parameter (default to csv)
        format_type = request.args.get('format', 'csv')
        
        # Load history
        with open(HISTORY_PATH, 'r') as file:
            history = json.load(file)
        
        if format_type == 'json':
            # Create a memory file for JSON
            output = io.StringIO()
            json.dump(history, output, indent=4)
            output.seek(0)
            
            return send_file(
                io.BytesIO(output.getvalue().encode('utf-8')),
                mimetype='application/json',
                as_attachment=True,
                download_name='adidas_sales_predictions_history.json'
            )
        else:
            # Default: CSV format
            output = io.StringIO()
            csv_writer = csv.writer(output)
            
            # Write headers
            if history:
                headers = ['Timestamp', 'Retailer ID', 'Price per Unit', 'Units Sold', 
                          'Operating Profit', 'Operating Margin', 'Date', 'Prediction']
                csv_writer.writerow(headers)
                
                # Write data rows
                for entry in history:
                    row = [
                        entry.get('timestamp', ''),
                        entry['input_data']['retailer_id'],
                        entry['input_data']['price_per_unit'],
                        entry['input_data']['units_sold'],
                        entry['input_data']['operating_profit'],
                        entry['input_data']['operating_margin'],
                        entry['input_data']['date'],
                        entry['prediction']
                    ]
                    csv_writer.writerow(row)
            
            output.seek(0)
            
            return send_file(
                io.BytesIO(output.getvalue().encode('utf-8')),
                mimetype='text/csv',
                as_attachment=True,
                download_name='adidas_sales_predictions_history.csv'
            )
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
   app.run(host='0.0.0.0', port=5000,debug=True)
