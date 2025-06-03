import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError

# Configuration
product_name = 'IQIYI VIP'  # Change to target a different product
SEQ_LEN = 1
EPOCHS = 100
BATCH_SIZE = 16
FUTURE_DAYS = 30
VAL_SPLIT = 0.1  # Percentage of training data to use for validation
train_test_split_ratio = 0.8 # 80% for training, 20% for testing

# Feature columns - UPDATED
# Ensure 'total_revenue' is the first feature if it's still implicitly part of the input,
# otherwise, the scaler and prediction logic might need adjustment.
# For now, assuming 'total_revenue' is still the target but features for input are these:
# If 'total_revenue' itself should also be an input feature, add it to the list.
# Based on your list, it seems 'total_revenue' is implied as the base for these features.
# Let's assume 'total_revenue' is still the primary value from which others are derived and is the first column for scaling.
features = [
    'total_revenue', # Assuming this is still needed as the base and for scaling the target
    'total_revenue_ema_10',
    'total_revenue_lag_1'
]
# If 'total_revenue' is NOT meant to be a direct input feature itself, remove it from the list above
# and ensure the data preprocessing and model input shape are adjusted accordingly.
# For this revision, I'm keeping 'total_revenue' as the first feature, assuming it's still the target
# and the base for other calculations and scaling.

# Load data from a single CSV
try:
    full_df = pd.read_csv('final_dataset.csv', parse_dates=['Tanggal'])
except FileNotFoundError:
    print("Error: 'final_dataset.csv' not found. Please ensure the file is in the correct directory.")
    exit()
    

# --- Feature Engineering (Applied before splitting and scaling) ---
# It's crucial that these features are generated on the DataFrame before splitting and scaling.
# The original script assumed features were pre-calculated in the CSV.
# If they are not, they MUST be calculated here.

# Filter by product first
product_df_base = full_df[full_df['Produk'] == product_name].copy()
print(len(product_df_base))

if product_df_base.empty:
    print(f"No data found for product: {product_name} in 'final_dataset.csv'.")
    exit()

product_df_base = product_df_base.sort_values('Tanggal').reset_index(drop=True)

# Calculate the new features if not already present
# Make sure 'total_revenue' exists in product_df_base
if 'total_revenue' not in product_df_base.columns:
    print("Error: 'total_revenue' column is missing, which is required for feature generation.")
    exit()

# EMA
if 'total_revenue_ema_10' in features:
    product_df_base['total_revenue_ema_10'] = product_df_base['total_revenue'].ewm(span=10, adjust=False).mean()

# Rolling window features
if 'total_revenue_roll_std_10' in features:
    product_df_base['total_revenue_roll_std_10'] = product_df_base['total_revenue'].rolling(window=10, min_periods=1).std()
if 'total_revenue_roll_min_10' in features:
    product_df_base['total_revenue_roll_min_10'] = product_df_base['total_revenue'].rolling(window=10, min_periods=1).min()
if 'total_revenue_roll_max_10' in features:
    product_df_base['total_revenue_roll_max_10'] = product_df_base['total_revenue'].rolling(window=10, min_periods=1).max()

# Lag features
if 'total_revenue_lag_1' in features:
    product_df_base['total_revenue_lag_1'] = product_df_base['total_revenue'].shift(1)
# --- End of Initial Feature Engineering ---


# Handle potential NaNs from feature engineering
product_df_cleaned = product_df_base.dropna(subset=features).reset_index(drop=True)

if product_df_cleaned.empty:
    print(f"Not enough data for product: {product_name} after generating features and dropping NaNs. Consider checking feature generation or data source.")
    exit()

# Split data into training and testing sets
split_index = int(len(product_df_cleaned) * train_test_split_ratio)
train_df = product_df_cleaned.copy().iloc[:split_index] # Use .copy() to avoid SettingWithCopyWarning
test_df = product_df_cleaned.copy().iloc[split_index:]

if train_df.empty or test_df.empty:
    print(f"Not enough data to create train/test split for product: {product_name}.")
    print(f"Train DF shape: {train_df.shape}, Test DF shape: {test_df.shape}")
    exit()
    
# Normalize features
scaler = MinMaxScaler()
# Ensure only feature columns are passed to the scaler
train_scaled = scaler.fit_transform(train_df[features])
test_scaled = scaler.transform(test_df[features])

# Sequence generation
def create_sequences(data, seq_len=SEQ_LEN):
    x, y = [], []
    if len(data) <= seq_len: 
        print(f"Warning: Data length ({len(data)}) is less than or equal to sequence length ({seq_len}). Cannot create sequences for this segment.")
        return np.array(x), np.array(y)
    for i in range(seq_len, len(data)):
        x.append(data[i-seq_len:i])
        y.append(data[i][0])  # Predict 'total_revenue' (which is the first feature in `features` list)
    return np.array(x), np.array(y)

x_train, y_train = create_sequences(train_scaled)
x_test, y_test = create_sequences(test_scaled)

if x_train.shape[0] == 0:
    print(f"Not enough data to create training sequences for product: {product_name} with SEQ_LEN={SEQ_LEN}.")
    if x_test.shape[0] == 0:
         print(f"Test data also too short for sequences. Test set might be too small for SEQ_LEN={SEQ_LEN}.")
    exit()



# Build model
# The input_shape must match the number of features used.
model = Sequential([
    LSTM(50, activation='tanh', return_sequences=True, input_shape=(SEQ_LEN, len(features)),
        ), 
    Dropout(0.2), 
        LSTM(50, activation='tanh'), 
            Dropout(0.2), 
    Dense(units=30, activation='tanh'), 
    Dense(1)
])

# --- 5. Compile the Model ---
optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', RootMeanSquaredError(name='rmse')])
model.summary()

# --- 6. Train the Model ---
early_stopping = EarlyStopping(monitor='val_loss', patience=35, restore_best_weights=True, verbose=1) 
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=15, verbose=1, min_lr=0.000001)
callbacks_list = [early_stopping, reduce_lr] 
optimizer = Adam(learning_rate=0.001) 
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[RootMeanSquaredError(name='rmse')])
model.summary()


# Train model with validation
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True) 
history = model.fit(
    x_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VAL_SPLIT, 
    callbacks=callbacks_list,
    verbose=1
)

# Plot training and validation loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title(f"Training and Validation Loss - {product_name}")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
filename_base = product_name.replace(" ", "_").replace("/", "_")
plt.savefig(f'{filename_base}_loss_plot.png')
plt.show()

# --- Corrected Forecasting Logic ---
if len(test_scaled) >= SEQ_LEN:
    current_sequence_scaled = test_scaled[-SEQ_LEN:].copy()
    # Determine the amount of history needed from the unscaled test_df for feature re-calculation
    # Max window/lag from new features is 10.
    min_history_for_features = max(SEQ_LEN, 10) + 7 # Max of SEQ_LEN or largest window/lag (10) + buffer
    history_df_unscaled = test_df[['Tanggal', 'Produk', 'total_revenue']].iloc[-min_history_for_features:].copy() # Ensure this has 'total_revenue'
    
    if len(history_df_unscaled) < min_history_for_features and len(train_df) >= min_history_for_features :
        print("Warning: Test data too short for full feature history, using training data for initial forecast history.")
        history_df_unscaled = train_df[['Tanggal', 'Produk', 'total_revenue']].iloc[-min_history_for_features:].copy()

elif len(train_scaled) >= SEQ_LEN:
    print("Warning: Test data is too short to form a sequence. Using last sequence from training data for forecasting.")
    current_sequence_scaled = train_scaled[-SEQ_LEN:].copy()
    min_history_for_features = max(SEQ_LEN, 10) + 7
    history_df_unscaled = train_df[['Tanggal', 'Produk', 'total_revenue']].iloc[-min_history_for_features:].copy()
else:
    print("Error: Neither training nor testing data is long enough to form an initial sequence for forecasting.")
    exit()

history_df_unscaled = history_df_unscaled.sort_values('Tanggal').reset_index(drop=True)
predicted_revenues_unscaled_list = [] 

for day_step in range(FUTURE_DAYS):
    input_for_model = current_sequence_scaled.reshape(1, SEQ_LEN, len(features))
    pred_scaled_revenue_component = model.predict(input_for_model, verbose=0)[0, 0]

    dummy_scaled_row = np.zeros((1, len(features)))
    dummy_scaled_row[0, 0] = pred_scaled_revenue_component # Assumes target ('total_revenue') is the first feature
    pred_unscaled_revenue = scaler.inverse_transform(dummy_scaled_row)[0, 0]
    predicted_revenues_unscaled_list.append(pred_unscaled_revenue)

    last_tanggal = history_df_unscaled['Tanggal'].iloc[-1]
    next_tanggal = last_tanggal + pd.Timedelta(days=1)
    # Append only the predicted 'total_revenue' and essential identifiers
    new_row_dict = {'Tanggal': next_tanggal, 'Produk': product_name, 'total_revenue': pred_unscaled_revenue}
    history_df_unscaled = pd.concat([history_df_unscaled, pd.DataFrame([new_row_dict])], ignore_index=True)
    
    # Re-calculate all features for the new row based on the updated history_df_unscaled
    temp_df_for_feature_eng = history_df_unscaled.copy() # Work on a copy for feature engineering
    
    # EMA
    if 'total_revenue_ema_10' in features:
        temp_df_for_feature_eng['total_revenue_ema_10'] = temp_df_for_feature_eng['total_revenue'].ewm(span=10, adjust=False).mean()
    # Rolling window features
    if 'total_revenue_roll_std_10' in features:
        temp_df_for_feature_eng['total_revenue_roll_std_10'] = temp_df_for_feature_eng['total_revenue'].rolling(window=10, min_periods=1).std()
    if 'total_revenue_roll_min_10' in features:
        temp_df_for_feature_eng['total_revenue_roll_min_10'] = temp_df_for_feature_eng['total_revenue'].rolling(window=10, min_periods=1).min()
    if 'total_revenue_roll_max_10' in features:
        temp_df_for_feature_eng['total_revenue_roll_max_10'] = temp_df_for_feature_eng['total_revenue'].rolling(window=10, min_periods=1).max()
    # Lag features
    if 'total_revenue_lag_1' in features:
        temp_df_for_feature_eng['total_revenue_lag_1'] = temp_df_for_feature_eng['total_revenue'].shift(1)
        
    # Get the last row, which contains all newly calculated unscaled features
    next_full_unscaled_feature_row_series = temp_df_for_feature_eng[features].iloc[-1]
    next_full_unscaled_feature_row_filled = next_full_unscaled_feature_row_series.fillna(method='ffill').fillna(0)
    
    # Scale this full feature row
    next_full_scaled_feature_row = scaler.transform(next_full_unscaled_feature_row_filled.values.reshape(1, -1))
    
    current_sequence_scaled = np.vstack((current_sequence_scaled[1:], next_full_scaled_feature_row))

pred_rescaled = np.array(predicted_revenues_unscaled_list)

# Save 30-day forecast
forecast_df = pd.DataFrame({
    'day_ahead': list(range(1, FUTURE_DAYS + 1)),
    'predicted_revenue': pred_rescaled
})
forecast_df.to_csv(f'{filename_base}_forecast_next_30_days.csv', index=False)
print(f"Forecast saved to {filename_base}_forecast_next_30_days.csv")

# Plot forecast only
plt.figure(figsize=(10, 5))
plt.plot(pred_rescaled, label="Forecast (Next 30 days)")
plt.title(f"LSTM Forecast: {product_name} (Next 30 Days)")
plt.xlabel("Days Ahead")
plt.ylabel("Revenue")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{filename_base}_forecast_plot.png')
plt.show()

# ===================
# Actual vs Predicted + Forecast Combined Plot
# ===================
if x_test.shape[0] > 0 and len(y_test) > 0: 
    true_y_test_scaled = y_test.reshape(-1, 1)
    # Create padding based on the number of features
    padding_test = np.zeros((len(true_y_test_scaled), len(features) - 1)) 
    true_y_inverse = scaler.inverse_transform(np.hstack((true_y_test_scaled, padding_test)))[:, 0]

    pred_test_scaled = model.predict(x_test, verbose=0) 
    pred_test_rescaled = scaler.inverse_transform(
        np.hstack((pred_test_scaled, np.zeros((len(pred_test_scaled), len(features) - 1))))
    )[:, 0]
    
    if len(test_df) >= SEQ_LEN + len(true_y_inverse):
        actual_test_dates = test_df['Tanggal'].iloc[SEQ_LEN : SEQ_LEN + len(true_y_inverse)]
    elif len(test_df) >= SEQ_LEN and len(test_df) < SEQ_LEN + len(true_y_inverse):
        actual_test_dates = test_df['Tanggal'].iloc[SEQ_LEN : len(test_df)] 
        num_available_dates = len(actual_test_dates)
        true_y_inverse = true_y_inverse[:num_available_dates]
        pred_test_rescaled = pred_test_rescaled[:num_available_dates]
    else: 
        actual_test_dates = pd.Series([], dtype='datetime64[ns]') 

    if not actual_test_dates.empty:
       last_actual_date = actual_test_dates.iloc[-1]
    elif not test_df['Tanggal'].empty: 
       last_actual_date = test_df['Tanggal'].iloc[-1]
    elif not train_df['Tanggal'].empty: 
        last_actual_date = train_df['Tanggal'].iloc[-1]
    else: 
        last_actual_date = pd.Timestamp('today') - pd.Timedelta(days=FUTURE_DAYS) 

    forecast_dates = pd.date_range(start=last_actual_date + pd.Timedelta(days=1), periods=FUTURE_DAYS)

    plt.figure(figsize=(14, 7)) 
    if not actual_test_dates.empty and len(true_y_inverse) > 0:
        plt.plot(actual_test_dates, true_y_inverse, label='Actual Revenue (Test)', color='black', marker='.', linestyle='-')
        plt.plot(actual_test_dates, pred_test_rescaled, label='Predicted Revenue (Test)', color='blue', linestyle='--')
    else:
        print("Warning: Not enough actual test data to plot for the combined chart.")
        
    plt.plot(forecast_dates, pred_rescaled, label=f'Forecast (Next {FUTURE_DAYS} Days)', color='orange', linestyle='--')
    
    plt.title(f'{product_name} - Actual vs Predicted with {FUTURE_DAYS}-Day Forecast', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Revenue', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{filename_base}_test_and_forecast_plot.png')
    plt.show()
else:
    print("x_test or y_test is empty. Cannot plot actual vs predicted results. Forecasting will still proceed if possible.")
    plt.figure(figsize=(10, 5))
    if not test_df.empty:
        last_historical_date = test_df['Tanggal'].iloc[-1]
    elif not train_df.empty:
        last_historical_date = train_df['Tanggal'].iloc[-1]
    else:
        last_historical_date = pd.Timestamp('today') - pd.Timedelta(days=FUTURE_DAYS)
    
    forecast_dates_only = pd.date_range(start=last_historical_date + pd.Timedelta(days=1), periods=FUTURE_DAYS)
    plt.plot(forecast_dates_only, pred_rescaled, label=f'Forecast (Next {FUTURE_DAYS} Days)', color='orange', linestyle='--')
    plt.title(f'{product_name} - {FUTURE_DAYS}-Day Forecast (No Test Data to Plot)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Revenue', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{filename_base}_forecast_only_plot_no_test.png')
    plt.show()

print("Script finished.")
