from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('stacked_model_compressed.pkl')
scaler = joblib.load('scaler_compressed.pkl')
model_columns = joblib.load('model_columns_compressed.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Collect numeric inputs (first 10)
        numeric_features = [
            "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
            "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
            "Horizontal_Distance_To_Fire_Points"
        ]
        input_data = [float(request.form[feature]) for feature in numeric_features]

        # 2. Handle Wilderness Area (4 binary columns, only 1 is 1)
        wilderness_area = [0, 0, 0, 0]
        for i in range(4):
            if request.form.get(f"Wilderness_Area_{i}") == '1':
                wilderness_area[i] = 1

        # 3. Handle Soil Type (dropdown converted to one-hot)
        soil_type_index = int(request.form.get("soil_type_dropdown"))
        soil_type = [0] * 40
        soil_type[soil_type_index] = 1

        # 4. Combine full input
        final_input = np.array(input_data + wilderness_area + soil_type).reshape(1, -1)

        # 5. Scale input and predict
        final_input_scaled = scaler.transform(final_input)
        prediction = model.predict(final_input_scaled)[0]

        # 6. Return prediction
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
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

