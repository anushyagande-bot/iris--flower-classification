import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model/iris_model.pkl")

st.title("Iris Flower Classification")

sepal_length = st.number_input("Sepal Length Cm")
sepal_width = st.number_input("Sepal Width Cm")
petal_length = st.number_input("Petal Length Cm")
petal_width = st.number_input("Petal Width Cm")

if st.button("Predict Flower"):

    input_data = pd.DataFrame({
        'SepalLengthCm': [sepal_length],
        'SepalWidthCm': [sepal_width],
        'PetalLengthCm': [petal_length],
        'PetalWidthCm': [petal_width]
    })

    prediction = model.predict(input_data)

    st.success(f"Predicted Flower: {prediction[0]}")