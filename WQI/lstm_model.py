import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Load the data
with open("edited/Water_Parameters_2013-2025.xlsx", "rb") as f1, \
     open("edited/Climatological_Parameters_2013-2025.xlsx", "rb") as f2, \
     open("edited/Volcanic_Parameters_2013-2024.xlsx", "rb") as f3:
    data = pd.read_excel(f1)
    ex_data1 = pd.read_excel(f2)
    ex_data2 = pd.read_excel(f3)

# DATA PROCESSING
# # Drop rows with too many missing values
# data = data.dropna(thresh=len(data.columns) - 3)

# Drop unnecessary columns
ex_data1 = ex_data1.drop(columns=["RH", "WIND_SPEED", "WIND_DIRECTION"])
ex_data1["T_AVE"] = (ex_data1["TMIN"] + ex_data1["TMAX"]) / 2
ex_data1 = ex_data1.drop(columns=["TMIN", "TMAX"])

# Augment external factors to data
data["Rainfall"] = ex_data1["RAINFALL"]
data["Env_Temperature"] = ex_data1["T_AVE"]
data["CO2"] = ex_data2["CO2 Flux (t/d)"]
data["SO2"] = ex_data2["SO2 Flux (t/d)"]

# Fill missing values with column means
for col in data.columns[1:]:
    data[col] = data[col].fillna(data[col].mean())

# Calculate Water Quality Index (WQI) - using standard formula
# We'll use a simplified version of the WQI calculation
def calculate_wqi(row):
    # Weight factors for different parameters (can be adjusted based on importance)
    weights = {
        'Surface Water Temp (°C)': 0.03,
        'Middle Water Temp (°C)': 0.03,
        'Bottom Water Temp (°C)': 0.03,
        'pH Level': 0.2,
        'Ammonia (mg/L)': 0.2,
        'Nitrate-N/Nitrite-N  (mg/L)': 0.15,
        'Phosphate (mg/L)': 0.15,
        'Dissolved Oxygen (mg/L)': 0.2
    }
    
    # Normalize each parameter (0-100 scale, higher is better for most)
    # Note: Normalization depends on ideal ranges for each parameter
    normalized = {
        'Surface Water Temp (°C)': min(max(0, 100 - 5*abs(row['Surface Water Temp (°C)'] - 25)), 100),  # Ideal around 25°C
        'Middle Water Temp (°C)': min(max(0, 100 - 5*abs(row['Middle Water Temp (°C)'] - 25)), 100) if pd.notna(row['Middle Water Temp (°C)']) else 75,
        'Bottom Water Temp (°C)': min(max(0, 100 - 5*abs(row['Bottom Water Temp (°C)'] - 25)), 100) if pd.notna(row['Bottom Water Temp (°C)']) else 75,
        'pH Level': min(max(0, 100 - 20*abs(row['pH Level'] - 7.5)), 100),  # Ideal around 7.5
        'Ammonia (mg/L)': min(max(0, 100 - 500*row['Ammonia (mg/L)']), 100),  # Lower is better
        'Nitrate-N/Nitrite-N  (mg/L)': min(max(0, 100 - 500*row['Nitrate-N/Nitrite-N  (mg/L)']), 100),  # Lower is better
        'Phosphate (mg/L)': min(max(0, 100 - 1000*row['Phosphate (mg/L)']), 100) if pd.notna(row['Phosphate (mg/L)']) else 75,  # Lower is better
        'Dissolved Oxygen (mg/L)': min(max(0, 20*row['Dissolved Oxygen (mg/L)']), 100)  # Higher is better
    }
    
    # Calculate WQI
    wqi = 0
    total_weight = 0
    for param, weight in weights.items():
        if param in normalized:
            wqi += normalized[param] * weight
            total_weight += weight
    
    return wqi / total_weight

# Add WQI column to dataframe
data['WQI'] = data.apply(calculate_wqi, axis=1)

# Feature selection - use all available parameters
features = ['Surface Water Temp (°C)', 'Middle Water Temp (°C)', 'Bottom Water Temp (°C)', 
            'pH Level', 'Ammonia (mg/L)', 'Nitrate-N/Nitrite-N  (mg/L)', 
            'Phosphate (mg/L)', 'Dissolved Oxygen (mg/L)', 'Rainfall', 'Env_Temperature', 'CO2', 'SO2']
target = 'WQI'

# Prepare data for LSTM (we'll treat the time series as 1D "images")
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Save scalers for future use
feature_scaler = StandardScaler()
target_scaler = MinMaxScaler()

# Apply scaling
scaled_features = feature_scaler.fit_transform(data[features])
scaled_target = target_scaler.fit_transform(data[[target]])

# Create sequences
seq_length = 6  # using 6 months of data to predict next month's WQI
X, y = create_sequences(scaled_features, seq_length)
X_target, y_target = create_sequences(scaled_target, seq_length)

# Split data into train and test sets
X_train, X_train_val, y_train, y_train_val = train_test_split(X, y_target, test_size=0.4, random_state=42)

X_test, X_val, y_test, y_val = train_test_split(X_train_val, y_train_val, test_size=0.5, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
X_val = torch.FloatTensor(X_val)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)
y_val = torch.FloatTensor(y_val)


# Create PyTorch Dataset
class WaterQualityDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

train_dataset = WaterQualityDataset(X_train, y_train)
test_dataset = WaterQualityDataset(X_test, y_test)
val_dataset = WaterQualityDataset(X_val, y_val)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# LSTM Model
# LSTM Model Definition
class WQILSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(WQILSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5 if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))  # out shape: (batch_size, seq_len, hidden_size)
        
        # Take the output from the last timestep
        out = out[:, -1, :]  # shape: (batch_size, hidden_size)
        
        # Fully connected layers
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out
    
    
if __name__ == '__main__':
    # Initialize LSTM model
    num_features = len(features)
    model = WQILSTM(input_size=num_features)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Early stop parameters
    patience = 20
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Training loop
    num_epochs = 150
    train_losses = []
    test_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        # features, labels = next(iter(train_loader))
        # for _ in range(100):    # Overfitting the training set
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

         # Test loss
        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                test_loss += criterion(outputs, labels).item()
        
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}: '
                  f'Train Loss: {avg_train_loss:.6f}, '
                  f'Val Loss: {avg_val_loss:.6f}, '
                  f'Test Loss: {avg_test_loss:.6f}')
            
         # Early stopping check
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     torch.save(model.state_dict(), 'best_model.pth')
        #     epochs_no_improve = 0
        # else:
        #     epochs_no_improve += 1
        #     if epochs_no_improve >= patience:
        #         print(f'\nEarly stopping at epoch {epoch+1}')
        #         break


 # Evaluate the model on test set
    model.eval()
    test_preds = []
    test_actual = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            test_preds.extend(outputs.numpy())
            test_actual.extend(labels.numpy())
    
    test_preds = np.array(test_preds)
    test_actual = np.array(test_actual)
    
    # Inverse transform predictions
    test_preds_orig = target_scaler.inverse_transform(test_preds.reshape(-1, 1))
    test_actual_orig = target_scaler.inverse_transform(test_actual.reshape(-1, 1))
    
    # Calculate quantitative metrics
    rmse = np.sqrt(mean_squared_error(test_actual_orig, test_preds_orig))
    r2 = r2_score(test_actual_orig, test_preds_orig)
    mae = np.mean(np.abs(test_actual_orig - test_preds_orig))
    
    print(f"\nQuantitative Analysis on Test Set:\n"
          f"RMSE: {rmse:.4f}\n"
          f"MAE: {mae:.4f}\n"
          f"R²: {r2:.4f}\n"
    )

    # Plot training history
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    plt.show()

    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.plot(test_actual, label='Actual WQI')
    plt.plot(test_preds, label='Predicted WQI')
    plt.xlabel('Sample')
    plt.ylabel('WQI (Normalized)')
    plt.legend()
    plt.title('Actual vs Predicted WQI')
    plt.show()

    # Save the model
    torch.save(model.state_dict(), 'wqi_lstm_model.pth')
