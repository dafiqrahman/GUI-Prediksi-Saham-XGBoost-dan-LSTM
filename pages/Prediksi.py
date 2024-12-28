import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from predict import load_model, predict

# Load models


MODELS = ["Model XGBoost GridSearch", "Model XGBoost PSO", "Model XGBoost Default"
          "Model LSTM Adam", "Model LSTM RMSprop"]


# Load scaler (gunakan scaler yang sama seperti saat training untuk semua model)
scaler = MinMaxScaler()

# Streamlit App
st.title("Prediksi Saham Kalbe Farma (KLBF)")
st.write("memprediksi **harga penutupan** dari saham berdasarkan harga **open**, **high**, dan **low** ")

# Dropdown for selecting model
selected_model_name = st.selectbox("Pilih Model Prediksi", MODELS)

if selected_model_name == "Model LSTM Adam":
    st.text('''
    Mean Squared Error (MSE): 10812.72
    Root Mean Squared Error (RMSE): 103.98
    Mean Absolute Error (MAE): 85.10
    Mean Absolute Percentage Error (MAPE): 5.42
    R-squared: 0.09
''')

elif selected_model_name == "Model LSTM RMSprop":
    st.text('''
    Mean Squared Error (MSE): 15181.06
    Root Mean Squared Error (RMSE): 123.21
    Mean Absolute Error (MAE): 100.09
    Mean Absolute Percentage Error (MAPE): 6.29
    R-squared: -0.27
''')

elif selected_model_name == "Model XGBoost GridSearch":
    st.text('''
    Mean Squared Error (MSE): 919.37
    Root Mean Squared Error (RMSE): 30.32
    Mean Absolute Error (MAE): 22.57
    Mean Absolute Percentage Error (MAPE): 1.50
    R-squared: 0.98
''')

elif selected_model_name == "Model XGBoost PSO":
    st.text('''
    Mean Squared Error (MSE): 925.0080253001507
    Root Mean Squared Error (RMSE): 30.413944586326693
    Mean Absolute Error (MAE): 22.456805170798788
    Mean Absolute Percentage Error (MAPE): 1.4949884698803528
    R-squared: 0.9843424448553131
''')

elif selected_model_name == "Model XGBoost Default":
    st.text('''
    Mean Squared Error (MSE): 1406.367046204681
    Root Mean Squared Error (RMSE): 37.501560583590134
    Mean Absolute Error (MAE): 27.51816771630527
    Mean Absolute Percentage Error (MAPE): 1.8396499395483343
    R-squared: 0.976194509693605 
''')

# pass to the function
model = load_model(selected_model_name)

# Input fields
open_price = st.number_input("Open Price", min_value=1530, step=50)
high_price = st.number_input("High Price", min_value=1550, step=50)
low_price = st.number_input("Low Price", min_value=1500, step=50)
close_price = st.number_input("Close Price", min_value=1500, step=50)

# Predict button
if st.button("Predict"):
    # Input validation
    if high_price >= low_price:
        # Predict
        predicted_close = predict(
            model, open_price, high_price, low_price, close_price, selected_model_name)
        print(predicted_close)
        st.success(
            f"Using {selected_model_name}, Predicted Close Price: ${predicted_close:.2f}")
    else:
        st.error("High price must be greater than or equal to low price.")
