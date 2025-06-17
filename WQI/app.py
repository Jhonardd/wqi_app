from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
from flask_cors import CORS

# Import your model classes
from cnn_model import WQICNN, calculate_wqi
from lstm_model import WQILSTM
from cnnlstm_model import CNNLSTM

app = Flask(__name__)
CORS(app)

# Configuration
DATA_PATHS = {
    'water': os.path.join("edited", "Water_Parameters_2013-2025.xlsx"),
    'climate': os.path.join("edited", "Climatological_Parameters_2013-2025.xlsx"),
    'volcano': os.path.join("edited", "Volcanic_Parameters_2013-2024.xlsx")
}

MODEL_PATHS = {
    'cnn': os.path.join("models", "wqi_cnn_model.pth"),
    'lstm': os.path.join("models", "wqi_lstm_model.pth"),
    'cnn_lstm': os.path.join("models", "wqi_cnnlstm_model.pth")
}

# Initialize predictor
predictor = None

def initialize_predictor():
    global predictor
    if predictor is None:
        predictor = WQIPredictor(DATA_PATHS)
        # Load models
        for model_type, model_path in MODEL_PATHS.items():
            predictor.load_model(model_type, model_path)
    return predictor

@app.route('/')
def home():
    try:
        predictor = initialize_predictor()
        max_date = predictor.get_max_date().strftime('%Y-%m-%d')
        return render_template('index.html', max_date=max_date)
    except Exception as e:
        return render_template('error.html', message="Initialization error")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    target_date = data.get('date')
    model_type = data.get('model', 'cnn_lstm')
    
    try:
        predictor = initialize_predictor()
        prediction = predictor.predict_wqi(target_date, model_type=model_type)
        
        # Determine water quality level
        if prediction > 90:
            level = "Excellent"
        elif 70 <= prediction <= 90:
            level = "Good"
        elif 50 <= prediction < 70:
            level = "Fair"
        else:
            level = "Poor"
        
        return jsonify({
            'status': 'success',
            'prediction': round(prediction, 2),
            'level': level,
            'date': target_date,
            'model': model_type
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/api/water-quality-data')
def get_water_quality_data():
    try:
        # Load and merge data files
        water_data = pd.read_excel(DATA_PATHS['water'])
        climate_data = pd.read_excel(DATA_PATHS['climate'])
        volcano_data = pd.read_excel(DATA_PATHS['volcano'])
        
        # Convert dates to datetime
        water_data['Date'] = pd.to_datetime(water_data['Date'])
        climate_data['Date'] = pd.to_datetime(climate_data['Date'])
        volcano_data['Date'] = pd.to_datetime(volcano_data['Date'])
        
        # Merge datasets
        merged_data = pd.merge(water_data, climate_data, on='Date', how='outer')
        merged_data = pd.merge(merged_data, volcano_data, on='Date', how='outer')
        
        # Select relevant columns
        result_data = merged_data[[
            'Date',
            'Ammonia (mg/L)',
            'Phosphate (mg/L)',
            'Dissolved Oxygen (mg/L)',
            'Nitrate-N/Nitrite-N  (mg/L)',
            'pH Level',
            'Surface Water Temp (°C)'
        ]].copy()
        
        # Rename columns to match frontend
        result_data.rename(columns={
            'Nitrate-N/Nitrite-N  (mg/L)': 'Nitrate (mg/L)',
            'Surface Water Temp (°C)': 'Temperature (°C)'
        }, inplace=True)
        
        # Fill missing dates with interpolation
        result_data.set_index('Date', inplace=True)
        result_data = result_data.resample('D').asfreq()
        result_data = result_data.interpolate(method='time', limit_direction='both')
        result_data.reset_index(inplace=True)
        
        # Convert to list of dictionaries
        result = result_data.to_dict(orient='records')
        
        # Format dates
        for item in result:
            item['Date'] = item['Date'].strftime('%Y-%m-%d')
                
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f"Error loading data: {str(e)}"
        }), 500

@app.route('/api/max-date')
def api_max_date():
    try:
        predictor = initialize_predictor()
        max_date = predictor.get_max_date()
        return jsonify({
            'status': 'success',
            'max_date': max_date.strftime('%Y-%m-%d')
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

class WQIPredictor:
    def __init__(self, data_paths):
        self.data = self.load_and_preprocess_data(data_paths)
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()

        # Calculate WQI
        self.data['WQI'] = self.data.apply(calculate_wqi, axis=1)

        self.features = [
            'Surface Water Temp (°C)', 'Middle Water Temp (°C)', 'Bottom Water Temp (°C)', 
            'pH Level', 'Ammonia (mg/L)', 'Nitrate-N/Nitrite-N  (mg/L)', 
            'Phosphate (mg/L)', 'Dissolved Oxygen (mg/L)', 'Rainfall', 'Env_Temperature', 'CO2', 'SO2'
        ]
        target = 'WQI'
        self.seq_length = 6

        # Apply scaling
        self.scaled_features = self.feature_scaler.fit_transform(self.data[self.features])
        self.scaled_target = self.target_scaler.fit_transform(self.data[[target]])
        
        # Initialize models
        self.models = {
            'cnn': None,
            'lstm': None,
            'cnn_lstm': None
        }
    
    def load_and_preprocess_data(self, data_paths):
        # Load data files
        water_data = pd.read_excel(data_paths['water'])
        climate_data = pd.read_excel(data_paths['climate'])
        volcano_data = pd.read_excel(data_paths['volcano'])
        
        # Preprocessing
        climate_data = climate_data.drop(columns=["RH", "WIND_SPEED", "WIND_DIRECTION"])
        climate_data["T_AVE"] = (climate_data["TMIN"] + climate_data["TMAX"]) / 2
        climate_data = climate_data.drop(columns=["TMIN", "TMAX"])
        
        # Augment data
        water_data = water_data.merge(
            climate_data[['Date', 'RAINFALL', 'T_AVE']], 
            on='Date', 
            how='left'
        )
        water_data = water_data.merge(
            volcano_data[['Date', 'CO2 Flux (t/d)', 'SO2 Flux (t/d)']], 
            on='Date', 
            how='left'
        )
        water_data.rename(columns={
            'RAINFALL': 'Rainfall',
            'T_AVE': 'Env_Temperature',
            'CO2 Flux (t/d)': 'CO2',
            'SO2 Flux (t/d)': 'SO2'
        }, inplace=True)
        
        # Fill missing values with column means
        for col in water_data.columns:
            if col != 'Date' and water_data[col].dtype in ['float64', 'int64']:
                water_data[col] = water_data[col].fillna(water_data[col].mean())
        
        # Ensure date column is datetime
        water_data['Date'] = pd.to_datetime(water_data['Date'])
        water_data = water_data.set_index('Date')
        
        return water_data
    
    def load_model(self, model_type, model_path):
        if model_type == 'cnn':
            model = WQICNN()
        elif model_type == 'lstm':
            model = WQILSTM(input_size=len(self.features))
        elif model_type == 'cnn_lstm':
            model = CNNLSTM(
                num_features=len(self.features),
                sequence_length=self.seq_length
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.load_state_dict(torch.load(model_path))
        model.eval()
        self.models[model_type] = model
        return model
    
    def generate_features_for_date(self, target_date):
        """Generate features for missing dates using monthly patterns"""
        if target_date in self.data.index:
            return self.data.loc[target_date, self.features].values
        
        # Find similar months in historical data
        month = target_date.month
        seasonal_data = self.data[self.data.index.month == month]
        
        if not seasonal_data.empty:
            return seasonal_data[self.features].mean().values
        return self.data[self.features].mean().values
    
    def prepare_input_sequence(self, target_date):
        """Prepare input sequence of seq_length days"""
        # Create sequence of 6 days ending at target date
        sequence_dates = [
            target_date - timedelta(days=i) 
            for i in range(self.seq_length - 1, -1, -1)
        ]
        
        sequence_data = []
        for date in sequence_dates:
            sequence_data.append(self.generate_features_for_date(date))
        
        # Convert to numpy array and scale
        scaled_sequence = self.feature_scaler.transform(np.array(sequence_data))
        return scaled_sequence
    
    def predict_wqi(self, target_date, model_type='cnn_lstm'):
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        
        # Handle future dates by generating synthetic data
        max_training_date = self.data.index.max()
        if target_date > max_training_date:
            # Generate synthetic data for future dates
            current_date = max_training_date + timedelta(days=1)
            while current_date <= target_date:
                if current_date not in self.data.index:
                    synthetic_features = self.generate_features_for_date(current_date)
                    new_row = pd.DataFrame([synthetic_features], 
                                        index=[current_date], 
                                        columns=self.features)
                    self.data = pd.concat([self.data, new_row])
                current_date += timedelta(days=1)  # Changed from months to days
        
        model = self.models.get(model_type)
        if model is None:
            raise ValueError(f"Model {model_type} not loaded")
        
        sequence = self.prepare_input_sequence(target_date)
        
        # Prepare input tensor based on model type
        if model_type == 'cnn':
            input_tensor = torch.FloatTensor(sequence).unsqueeze(0).unsqueeze(1)
        else:  # lstm or cnn_lstm
            input_tensor = torch.FloatTensor(sequence).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor).numpy().flatten()[0]
        
        # Inverse transform prediction
        prediction_orig = self.target_scaler.inverse_transform([[prediction]])[0][0]
        
        # Ensure prediction is within valid range
        return max(0, min(100, prediction_orig))
    
    def get_max_date(self):
        """Get a practical maximum date for predictions"""
        # Return date 5 years from now
        future_date = datetime.now() + relativedelta(years=5)
        return future_date

if __name__ == '__main__':
    app.run(debug=True, port=5000)