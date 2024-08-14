# Required Libraries
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import rasterio
from datetime import datetime, timedelta
import shap
import seaborn as sns

# Hyperparameters and configuration
absence_ratios = [0.2]
kfold_patience = 10
final_model_patience = 100
batch_size = 32
learning_rate = 0.001
k_folds = 5
sequence_length = 365  # Change this value to adjust the number of days used for input features

# Default selected features (modify this list as needed)
selected_features = [
    # "mean_2m_air_temperature", 
    "range_2m_air_temperature", 
    # "relative_humidity", 
    # "surface_pressure", 
    # "total_precipitation", 
    # "wind_10m"
]

# File paths and folders
data_file = 'datasets/aedes_aegypti_dataset.csv'
era5_directory = 'era5'  # Replace with the actual path
output_folder = 'predictions'  # Replace with the desired output folder
auc_csv_file = os.path.join(output_folder, 'auc_values.csv')
auc_plot_file = os.path.join(output_folder, 'auc_vs_absence_ratio.png')
shap_image_file = os.path.join(output_folder, 'shap_feature_importance.png')
roc_data_file = os.path.join(output_folder, 'roc_data.csv')
roc_plot_file = os.path.join(output_folder, 'roc_curve.png')
test_predictions_file = os.path.join(output_folder, 'test_predictions.csv')
test_statistics_file = os.path.join(output_folder, 'test_statistics.csv')
shap_values_file = os.path.join(output_folder, 'shap_values.csv')
final_model_file = os.path.join(output_folder, 'final_model_state_dict.pth')

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Date range for predictions (string format)
start_date_str = "2021-01-01"  # Replace with your desired start date
end_date_str = "2021-12-31"  # Replace with your desired end date
start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

# Define the neural network
class FeedforwardNN(nn.Module):
    def __init__(self, input_size):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Load the data
data = pd.read_csv(data_file)

# Identify the columns that represent the selected weather variables
weather_columns = [col for col in data.columns[4:] if any(var in col for var in selected_features)]
weather_var_types = sorted(set([col.split('.')[1] for col in weather_columns]))  # Extract selected weather variable types

# Initialize scalers for each selected weather variable type
scalers = {var: StandardScaler() for var in weather_var_types}

# Batch standardize the selected weather variables
for var in weather_var_types:
    # Select all columns corresponding to this weather variable type
    cols_to_standardize = [col for col in weather_columns if var in col]
    
    # Fit the scaler on the selected columns and transform the data
    scalers[var].fit(data[cols_to_standardize])
    data[cols_to_standardize] = scalers[var].transform(data[cols_to_standardize])

# Separate the data into presence (1) and absence (0) observations
presence_data = data[data['real_observation'] == 1]
absence_data = data[data['real_observation'] == 0]

# Create test set with 20% of presence observations and equal number of absences
presence_train, presence_test = train_test_split(presence_data, test_size=0.2, random_state=42)
absence_test = absence_data.sample(n=len(presence_test), random_state=42)
test_set = pd.concat([presence_test, absence_test]).reset_index(drop=True)

# Remaining data for training and validation
remaining_presence = presence_train
remaining_absence = absence_data.drop(absence_test.index)

# Function to create balanced dataset for validation
def create_balanced_dataset(presence, absence, n_samples):
    balanced_presence = presence.sample(n=n_samples, replace=False, random_state=42)
    balanced_absence = absence.sample(n=n_samples, replace=False, random_state=42)
    return pd.concat([balanced_presence, balanced_absence]).reset_index(drop=True)

# Function to prepare data for feedforward neural network
def prepare_ffnn_data(df, weather_columns, sequence_length, weather_var_types):
    X = []
    y = []
    # Define the days range based on the sequence length
    days_range = range(-sequence_length, 0)
    
    for i in range(len(df)):
        weather_sequence = []
        for day in days_range:
            daily_data = []
            for var in weather_var_types:
                col_name = f'{day}.{var}'
                daily_data.append(df.iloc[i][col_name])
            weather_sequence.append(daily_data)
        
        X.append(np.array(weather_sequence).flatten())  # Flatten the sequence into a single vector
        y.append(df.iloc[i]['real_observation'])
    
    return np.array(X), np.array(y)

# Store AUC values for different absence ratios
auc_values = []
best_overall_auc = 0
best_overall_model = None
best_overall_input_size = 0
best_absence_ratio = None

# Define feedforward neural network hyperparameters
input_size = sequence_length * len(weather_var_types)

for absence_ratio in absence_ratios:
    fold_aucs = []
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(remaining_presence)):
        # Create balanced validation set
        val_presence = remaining_presence.iloc[val_idx]
        val_set = create_balanced_dataset(val_presence, remaining_absence, len(val_presence))
        
        # Create training set with specified absence ratio
        train_presence = remaining_presence.iloc[train_idx]
        n_train_absence = int(len(train_presence) * absence_ratio)
        train_absence = remaining_absence.sample(n=n_train_absence, replace=False, random_state=42)
        train_set = pd.concat([train_presence, train_absence]).reset_index(drop=True)
        
        # Prepare features and target for feedforward neural network
        X_train, y_train = prepare_ffnn_data(train_set, weather_columns, sequence_length, weather_var_types)
        X_val, y_val = prepare_ffnn_data(val_set, weather_columns, sequence_length, weather_var_types)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
        
        # Create DataLoader for training and validation sets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize the feedforward neural network model, loss function, and optimizer
        model = FeedforwardNN(input_size)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training the model with early stopping
        best_val_auc = 0.0
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(1000):  # Set a high number, we'll use early stopping
            model.train()
            train_loss = 0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_probs = torch.sigmoid(val_outputs).numpy().flatten()
                
                fpr, tpr, thresholds = roc_curve(y_val, val_probs)
                val_auc = auc(fpr, tpr)
                
            print(f'Absence Ratio: {absence_ratio}, Fold {fold+1}, Epoch [{epoch+1}], '
                  f'Train Loss: {train_loss/len(train_loader):.4f}, '
                  f'Val AUC: {val_auc:.4f}')
        
            # Save the model if validation AUC improves
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Check if we should stop early
            if patience_counter >= kfold_patience:
                print("Early stopping")
                break
        
        # Load the best model state
        model.load_state_dict(best_model_state)
        fold_aucs.append(best_val_auc)
        
        # Update the best overall model if this fold has the highest AUC
        if best_val_auc > best_overall_auc:
            best_overall_auc = best_val_auc
            best_overall_model = model
            best_overall_input_size = input_size
            best_absence_ratio = absence_ratio
    
    # Store the mean AUC for this absence ratio
    mean_auc = np.mean(fold_aucs)
    auc_values.append(mean_auc)

# Save AUC values to a file
auc_df = pd.DataFrame({'absence_ratio': absence_ratios, 'auc': auc_values})
auc_df.to_csv(auc_csv_file, index=False)

# Plot AUC vs Absence Ratio
plt.figure()
plt.plot(absence_ratios, auc_values, marker='o')
plt.xlabel('Absence Ratio')
plt.ylabel('AUC')
plt.title('AUC vs Absence Ratio')
plt.grid(True)
plt.savefig(auc_plot_file)
plt.close()

# Train the final model using the best hyperparameters
final_train_set = pd.concat([remaining_presence, remaining_absence]).reset_index(drop=True)
X_final, y_final = prepare_ffnn_data(final_train_set, weather_columns, sequence_length, weather_var_types)

# Perform an 80/20 split for training and validation sets
X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(X_final, y_final, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_final_tensor = torch.tensor(X_train_final, dtype=torch.float32)
y_train_final_tensor = torch.tensor(y_train_final, dtype=torch.float32).view(-1, 1)
X_val_final_tensor = torch.tensor(X_val_final, dtype=torch.float32)
y_val_final_tensor = torch.tensor(y_val_final, dtype=torch.float32).view(-1, 1)

# Create DataLoaders for final training and validation sets
final_train_dataset = TensorDataset(X_train_final_tensor, y_train_final_tensor)
final_val_dataset = TensorDataset(X_val_final_tensor, y_val_final_tensor)
final_train_loader = DataLoader(final_train_dataset, batch_size=batch_size, shuffle=True)
final_val_loader = DataLoader(final_val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the final model
final_model = FeedforwardNN(best_overall_input_size)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(final_model.parameters(), lr=learning_rate)

# Train the final model with early stopping
best_final_val_auc = 0.0
best_final_model_state = None
final_patience_counter = 0

for epoch in range(1000):  # Set a high number, we'll use early stopping
    final_model.train()
    train_loss = 0
    
    for X_batch, y_batch in final_train_loader:
        optimizer.zero_grad()
        outputs = final_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    final_model.eval()
    with torch.no_grad():
        val_outputs = final_model(X_val_final_tensor)
        val_probs = torch.sigmoid(val_outputs).numpy().flatten()
        
        fpr, tpr, thresholds = roc_curve(y_val_final, val_probs)
        val_auc = auc(fpr, tpr)
        
    print(f'Final Model Training - Epoch [{epoch+1}], Train Loss: {train_loss/len(final_train_loader):.4f}, '
          f'Val AUC: {val_auc:.4f}')
    
    # Save the model if validation AUC improves
    if val_auc > best_final_val_auc:
        best_final_val_auc = val_auc
        best_final_model_state = final_model.state_dict()
        final_patience_counter = 0
    else:
        final_patience_counter += 1
    
    # Check if we should stop early
    if final_patience_counter >= final_model_patience:
        print("Early stopping for final model")
        break

# Load the best final model state
final_model.load_state_dict(best_final_model_state)

# Save the final model state dict
torch.save(final_model.state_dict(), final_model_file)

# Evaluate the final model on the test set
X_test, y_test = prepare_ffnn_data(test_set, weather_columns, sequence_length, weather_var_types)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

final_model.eval()
with torch.no_grad():
    test_outputs = final_model(X_test_tensor)
    test_probs = torch.sigmoid(test_outputs).numpy().flatten()
    
    fpr, tpr, thresholds = roc_curve(y_test, test_probs)
    test_auc = auc(fpr, tpr)
    
    test_preds = (test_probs > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()
    test_tpr = tp / (tp + fn)
    test_tnr = tn / (tn + fp)
    test_accuracy = accuracy_score(y_test, test_preds)
    test_precision = precision_score(y_test, test_preds)
    test_recall = recall_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds)

# Save test predictions to a file
test_predictions_df = pd.DataFrame({'y_true': y_test, 'y_pred': test_preds, 'y_prob': test_probs})
test_predictions_df.to_csv(test_predictions_file, index=False)

# Save test statistics to a file
test_statistics_df = pd.DataFrame({
    'test_loss': [criterion(torch.tensor(test_preds, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)).item()],
    'test_accuracy': [test_accuracy],
    'test_precision': [test_precision],
    'test_recall': [test_recall],
    'test_tpr': [test_tpr],
    'test_tnr': [test_tnr],
    'test_auc': [test_auc],
    'test_f1': [test_f1]
})
test_statistics_df.to_csv(test_statistics_file, index=False)

print(f"Final Model Test Results:")
print(f"Test AUC: {test_auc:.4f}")
print(f"Test TPR: {test_tpr:.4f}")
print(f"Test TNR: {test_tnr:.4f}")

# SHAP Analysis
background = X_train_final[np.random.choice(X_train_final.shape[0], 100, replace=False)]
background_tensor = torch.tensor(background, dtype=torch.float32)
test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Create the explainer
e = shap.GradientExplainer(final_model, background_tensor)

# Calculate SHAP values
shap_values = e.shap_values(test_tensor)

# If shap_values is a list, take the first element (for binary classification)
if isinstance(shap_values, list):
    shap_values = shap_values[0]

# Reshape SHAP values for plotting (sequence_length days, selected weather variables)
shap_values_reshaped = shap_values.reshape(-1, sequence_length, len(weather_var_types))

# Calculate mean absolute SHAP values across all samples
mean_shap = np.abs(shap_values_reshaped).mean(axis=0)

# Create a DataFrame for plotting
weather_var_names = weather_var_types  # Use the selected weather variable names as the columns
shap_df = pd.DataFrame(mean_shap, columns=weather_var_names, index=range(-sequence_length, 0))

# Save SHAP values to a file
shap_values_df = pd.DataFrame(shap_values_reshaped.reshape(-1, sequence_length * len(weather_var_types)), columns=[f"{var}_{day}" for day in range(-sequence_length, 0) for var in weather_var_names])
shap_values_df.to_csv(shap_values_file, index=False)

# Plot SHAP feature importance
plt.figure(figsize=(20, 10))
shap_plot = sns.heatmap(shap_df, cmap='RdBu_r', center=0, yticklabels=50)
plt.title('SHAP Feature Importance')
plt.ylabel('Days before prediction')
plt.xlabel('Weather Variables')
plt.tight_layout()
plt.savefig(shap_image_file)
plt.close()

# Print the top 10 most important features
shap_df_melted = shap_df.reset_index().melt(id_vars='index', var_name='Weather Variable', value_name='SHAP Value')
shap_df_melted = shap_df_melted.rename(columns={'index': 'Day'})
top_features = shap_df_melted.sort_values('SHAP Value', ascending=False).head(10)
print("Top 10 most important features:")
print(top_features)

# Save ROC data to a file
roc_data_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})
roc_data_df.to_csv(roc_data_file, index=False)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.grid(True)
plt.savefig(roc_plot_file)
plt.close()
