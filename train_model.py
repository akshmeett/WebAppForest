import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
df = pd.read_csv("dataset/forest_cover_dataset.csv", header=None)

# Assign column names
columns = [
    "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points"
]
columns += [f"Wilderness_Area_{i}" for i in range(4)]
columns += [f"Soil_Type_{i}" for i in range(40)]
columns += ["Cover_Type"]
df.columns = columns

# Split features and labels
X = df.drop("Cover_Type", axis=1)
y = df["Cover_Type"]

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Use a single RandomForest model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and preprocessing files
joblib.dump(model, "stacked_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("✅ FAST model trained and saved successfully!")
