import joblib
import pandas as pd

# Load model
model = joblib.load("model/iris_model.pkl")

# Example input
input_data = pd.DataFrame({
    'SepalLengthCm': [5.1],
    'SepalWidthCm': [3.5],
    'PetalLengthCm': [1.4],
    'PetalWidthCm': [0.2]
})

# Predict
prediction = model.predict(input_data)

print("Predicted Flower:", prediction[0])