import joblib

# Compress the trained model
model = joblib.load('stacked_model.pkl')
joblib.dump(model, 'stacked_model_compressed.pkl', compress=3)

# Compress the scaler
scaler = joblib.load('scaler.pkl')
joblib.dump(scaler, 'scaler_compressed.pkl', compress=3)

# Compress the model_columns
columns = joblib.load('model_columns.pkl')
joblib.dump(columns, 'model_columns_compressed.pkl', compress=3)
