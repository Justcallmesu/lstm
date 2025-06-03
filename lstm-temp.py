import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError

# Configuration
product_name = 'VIU Premium'  # Change to target a different product
SEQ_LEN = 14
EPOCHS = 100
BATCH_SIZE = 8
FUTURE_DAYS = 30
VAL_SPLIT = 0.1  # 10% validation split

# Feature columns
features = [
    'total_revenue', 'total_revenue_ema_3', 'total_revenue_roll_std_3',
    'total_revenue_roll_min_3', 'total_revenue_roll_max_3',
    'total_revenue_ema_7', 'total_revenue_roll_std_7',
    'total_revenue_roll_min_7', 'total_revenue_roll_max_7',
    'total_revenue_lag_1', 'total_revenue_lag_2',
    'total_revenue_lag_3', 'total_revenue_lag_7'
]

# Load data
train_df = pd.read_csv('data_training.csv', parse_dates=['Tanggal'])
test_df = pd.read_csv('data_test.csv', parse_dates=['Tanggal'])

# Filter by product
train_df = train_df[train_df['Produk'] == product_name].copy()
test_df = test_df[test_df['Produk'] == product_name].copy()

# Normalize features
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df[features])
test_scaled = scaler.transform(test_df[features])

# Sequence generation
def create_sequences(data, seq_len=SEQ_LEN):
    x, y = [], []
    for i in range(seq_len, len(data)):
        x.append(data[i-seq_len:i])
        y.append(data[i][0])  # Predict 'total_revenue'
    return np.array(x), np.array(y)

x_train, y_train = create_sequences(train_scaled)
x_test, y_test = create_sequences(test_scaled)


# Build model
model = Sequential([
    LSTM(32, activation='relu', return_sequences=True, input_shape=(SEQ_LEN, len(features)),kernel_regularizer=l2(0.005)),
    Dropout(0.5),
    LSTM(64, activation='tanh'),
    Dropout(0.5),
    Dense(1)
])
optimizer = Adam(learning_rate=0.0001, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', RootMeanSquaredError(name='rmse')])
model.summary()


# Train model with validation
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    x_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VAL_SPLIT,
    callbacks=[early_stop],
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
plt.savefig('loss_plot.png')
plt.show()

# Forecast next 30 days
last_sequence = test_scaled[-SEQ_LEN:].copy()
future_predictions = []

for _ in range(FUTURE_DAYS):
    input_seq = last_sequence.reshape(1, SEQ_LEN, len(features))
    pred = model.predict(input_seq, verbose=0)[0][0]

    new_row = np.zeros((len(features),))
    new_row[0] = pred
    future_predictions.append(pred)

    last_sequence = np.vstack((last_sequence[1:], new_row))

# Inverse scale 30-day predictions
pred_array = np.array(future_predictions).reshape(-1, 1)
padding = np.zeros((FUTURE_DAYS, len(features) - 1))
inverse_input = np.hstack((pred_array, padding))
pred_rescaled = scaler.inverse_transform(inverse_input)[:, 0]

# Save 30-day forecast
forecast_df = pd.DataFrame({
    'day_ahead': list(range(1, FUTURE_DAYS + 1)),
    'predicted_revenue': pred_rescaled
})
filename_base = product_name.replace(" ", "_")
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

# Inverse transform y_test (true values)
true_y_test_scaled = y_test.reshape(-1, 1)
padding_test = np.zeros((len(true_y_test_scaled), len(features) - 1))
true_y_inverse = scaler.inverse_transform(np.hstack((true_y_test_scaled, padding_test)))[:, 0]

# Predicted test values
pred_test = model.predict(x_test, verbose=0)
pred_test_rescaled = scaler.inverse_transform(
    np.hstack((pred_test, np.zeros((len(pred_test), len(features) - 1))))
)[:, 0]

# Combine test dates and forecast dates
full_timeline = list(test_df['Tanggal'][SEQ_LEN:])
forecast_dates = pd.date_range(start=full_timeline[-1] + pd.Timedelta(days=1), periods=FUTURE_DAYS)

# Plot combined
plt.figure(figsize=(12, 6))
plt.plot(full_timeline, true_y_inverse, label='Actual Revenue (Test)', color='black')
plt.plot(full_timeline, pred_test_rescaled, label='Predicted Revenue (Test)', linestyle='--', color='blue')
plt.plot(forecast_dates, pred_rescaled, label='Forecast (Next 30 Days)', linestyle='--', color='orange')
plt.title(f'{product_name} - Actual vs Predicted with 30-Day Forecast')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{filename_base}_test_and_forecast_plot.png')
plt.show()
