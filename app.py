from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# ✅ Load your model, scaler, and column info
model = joblib.load('stacked_model_compressed.pkl')
scaler = joblib.load('scaler_compressed.pkl')
model_columns = joblib.load('model_columns_compressed.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ✅ 1. Collect numeric inputs
        numeric_features = [
            "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
            "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
            "Horizontal_Distance_To_Fire_Points"
        ]

        # ✅ Get input values
        input_data = [float(request.form[feature]) for feature in numeric_features]

        # ✅ 2. Validate numeric ranges
        limits = [
            (0, 5000),   # Elevation (m)
            (0, 360),    # Aspect (degrees)
            (0, 90),     # Slope (degrees)
            (0, 5000),   # Horizontal_Distance_To_Hydrology (m)
            (0, 1000),   # Vertical_Distance_To_Hydrology (m)
            (0, 5000),   # Horizontal_Distance_To_Roadways (m)
            (0, 255),    # Hillshade_9am
            (0, 255),    # Hillshade_Noon
            (0, 255),    # Hillshade_3pm
            (0, 5000)    # Horizontal_Distance_To_Fire_Points (m)
        ]

        for value, (min_val, max_val) in zip(input_data, limits):
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"Input value {value} out of range ({min_val}-{max_val}). "
                    "Please check your inputs."
                )

        # ✅ 3. Handle Wilderness Area (4 binary inputs — exactly 1 must be selected)
        wilderness_area = [0, 0, 0, 0]
        for i in range(4):
            if request.form.get(f"Wilderness_Area_{i}") == '1':
                wilderness_area[i] = 1

        if sum(wilderness_area) != 1:
            raise ValueError("Please select exactly one Wilderness Area option.")

        # ✅ 4. Handle Soil Type (dropdown mapped to one-hot)
        soil_type_index = int(request.form.get("soil_type_dropdown"))
        if not (0 <= soil_type_index < 40):
            raise ValueError("Invalid Soil Type selected. Please choose a valid option.")

        soil_type = [0] * 40
        soil_type[soil_type_index] = 1

        # ✅ 5. Combine final input
        final_input = np.array(input_data + wilderness_area + soil_type).reshape(1, -1)

        # ✅ 6. Scale & predict
        final_input_scaled = scaler.transform(final_input)
        prediction = model.predict(final_input_scaled)[0]

        cover_types = {
            1: "Spruce/Fir",
            2: "Lodgepole Pine",
            3: "Ponderosa Pine",
            4: "Cottonwood/Willow",
            5: "Aspen",
            6: "Douglas-fir",
            7: "Krummholz"
        }

        result = cover_types.get(prediction, f"Cover Type {prediction}")

        return render_template('index.html', prediction=result)

    except Exception as e:
        # ✅ If any error happens, show it on the page
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
