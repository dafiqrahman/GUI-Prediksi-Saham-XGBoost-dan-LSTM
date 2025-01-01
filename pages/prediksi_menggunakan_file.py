import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from predict import load_model, predict
import pandas as pd

# Load models


MODELS = ["Model XGBoost Default", "Model XGBoost GridSearch", "Model XGBoost PSO",
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

# upload file for data
uploaded_file = st.file_uploader(
    "Upload File dengan header Date,Close,High,Low,Open,Volume", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["Date"])
    st.write(df)
    st.write("Jumlah data: ", df.shape[0])
    st.write("Jumlah kolom: ", df.shape[1])
    # get the latest data
    last_row = df.tail(1)
    open_price = last_row["Open"].values[0]
    high_price = last_row["High"].values[0]
    low_price = last_row["Low"].values[0]
    close_price = last_row["Close"].values[0]


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
