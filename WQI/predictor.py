import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from cnn_model import WQICNN, calculate_wqi
from lstm_model import WQILSTM
from cnnlstm_model import CNNLSTM


class WQIPredictor:
    def __init__(self, data_paths):
        # Load data
        self.data = self.load_and_preprocess_data(data_paths)
        # Save scalers for future use
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()

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
        
        # Initialize models dictionary
        self.models = {
            'cnn': None,
            'lstm': None,
            'cnn_lstm': None
        }
    
    def load_and_preprocess_data(self, data_paths):
        # Load data files
        with open(data_paths['water'], "rb") as f1, \
             open(data_paths['climate'], "rb") as f2, \
             open(data_paths['volcano'], "rb") as f3:
            data = pd.read_excel(f1)
            ex_data1 = pd.read_excel(f2)
            ex_data2 = pd.read_excel(f3)
        
        # Preprocessing
        ex_data1 = ex_data1.drop(columns=["RH", "WIND_SPEED", "WIND_DIRECTION"])
        ex_data1["T_AVE"] = (ex_data1["TMIN"] + ex_data1["TMAX"]) / 2
        ex_data1 = ex_data1.drop(columns=["TMIN", "TMAX"])
        
        # Augment data
        data["Rainfall"] = ex_data1["RAINFALL"]
        data["Env_Temperature"] = ex_data1["T_AVE"]
        data["CO2"] = ex_data2["CO2 Flux (t/d)"]
        data["SO2"] = ex_data2["SO2 Flux (t/d)"]
        
        # Fill missing values
        for col in data.columns[1:]:
            data[col] = data[col].fillna(data[col].mean())
        
        # Ensure date column is datetime
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.set_index('Date')
        
        return data
    
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
        # If date already exists, return existing features
        if target_date in self.data.index:
            return self.data.loc[target_date, self.features]
        
        # Generate new features
        month = target_date.month
        
        seasonal_data = self.data[self.data.index.month == month]
        if not seasonal_data.empty:
            return seasonal_data[self.features].mean()
        return self.data[self.features].mean()  # Fallback to overall mean
    
    def prepare_input_sequence(self, target_date):
        """
        Prepare input sequence for prediction, generating any missing dates
        """
        # Calculate the first date needed in the sequence
        start_date = target_date - relativedelta(months=self.seq_length - 1)
        
        # Generate missing dates in the sequence range
        current_date = start_date
        while current_date <= target_date:
            if current_date not in self.data.index:
                # Generate features for this missing date
                new_features = self.generate_features_for_date(current_date)
                new_row = pd.DataFrame([new_features], index=[current_date], columns=self.features)
                self.data = pd.concat([self.data, new_row])
            current_date += relativedelta(months=1)
        
        # Sort index after adding new dates
        self.data = self.data.sort_index()
        
        # Get the sequence data
        sequence_data = self.data.loc[start_date:target_date, self.features]
        
        # Verify we have enough data
        if len(sequence_data) < self.seq_length:
            raise ValueError(f"Not enough data points. Need {self.seq_length}, got {len(sequence_data)}")
        
        # Scale the features
        scaled_sequence = self.feature_scaler.transform(sequence_data.tail(self.seq_length))
        return scaled_sequence
    
    def predict_wqi(self, target_date, model_type='cnn_lstm'):
        """Predict WQI for a specific future date"""
        # Convert to datetime if needed
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        
        # Get model
        model = self.models.get(model_type)
        if model is None:
            raise ValueError(f"Model {model_type} not loaded. Call load_model() first.")
        
        # Prepare input sequence (generates any missing dates)
        sequence = self.prepare_input_sequence(target_date)
        
        # Convert to tensor based on model type
        if model_type == 'cnn':
            input_tensor = torch.FloatTensor(sequence).unsqueeze(0).unsqueeze(1)
        else:  # lstm or cnn_lstm
            input_tensor = torch.FloatTensor(sequence).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor).numpy().flatten()[0]
        
        # Inverse transform prediction
        prediction_orig = self.target_scaler.inverse_transform([[prediction]])[0][0]
        return prediction_orig

# Example usage
if __name__ == '__main__':
    # Configuration
    DATA_PATHS = {
        'water': "edited/Water_Parameters_2013-2025.xlsx",
        'climate': "edited/Climatological_Parameters_2013-2025.xlsx",
        'volcano': "edited/Volcanic_Parameters_2013-2024.xlsx"
    }
    MODEL_PATHS = {
        'cnn': 'models/wqi_cnn_model.pth',
        'lstm': 'models/wqi_lstm_model.pth',
        'cnn_lstm': 'models/wqi_cnnlstm_model.pth'
    }
    
    # Create predictor
    predictor = WQIPredictor(DATA_PATHS)
    
    # Load models
    for model_type, model_path in MODEL_PATHS.items():
        predictor.load_model(model_type, model_path)
    
    # Predict for a specific future date
    while True:
        try:
            future_date = input(str("Enter a date (YYYY-MM-DD) or 'q' to quit: "))
            if future_date == 'q':
                break

            model = input(str("\nChoose a model (CNN, LSTM, CNN_LSTM): ")).lower()
            if model not in ['cnn', 'lstm', 'cnn_lstm']:
                print(f"{model} is not one of the models.")
                break

            prediction = predictor.predict_wqi(future_date, model_type=model)
            
            pollutant_level = ""
            if prediction > 90:
                pollutant_level = "Excellent"
            elif 70 <= prediction <= 90:
                pollutant_level = "Good"
            elif 50 <= prediction < 70:
                pollutant_level = "Fair"
            else:
                pollutant_level = "Poor"
            
            print(f"Predicted WQI for {future_date}: {prediction:.2f}\n"
                  f"Pollutant Level: {pollutant_level}"
            )

        except ValueError as e:
            print(f"{future_date} and {model} are not strings")