import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pickle


MODELS_PATH = {
    "Model XGBoost GridSearch": "models/xgboost_model_gridsearch.pkl",
    "Model XGBoost PSO": "models/xgboost_model_gridsearch.pkl",
    "Model LSTM Adam": "models/model_lstm_adam.keras",
    "Model LSTM RMSprop": "models/model_lstm_rmsprop.keras"
}


def load_model(model_name):

    model_path = MODELS_PATH[model_name]

    if model_path.endswith(".pkl"):
        return pickle.load(open(model_path, "rb"))
    elif model_path.endswith(".keras"):
        return tf.keras.models.load_model(model_path)
    else:
        raise ValueError("Unsupported model format.")


def predict(model, open_price, high_price, low_price, close_price, model_name):
    # Load scaler
    if model_name.startswith("Model LSTM"):
        scaler = pickle.load(open("models/scaler_lstm.pkl", "rb"))
    else:
        scaler = pickle.load(open("models/scaler_xgb.pkl", "rb"))
    # Normalize the input
    input_data = np.array([[open_price, high_price, low_price, close_price]])
    # Replace fit_transform with transform if scaler is pre-fitted

    # predict for xgboost and lstm
    if model_name.startswith("Model XGBoost"):
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)
        prediction_close = prediction[0]
    else:
        scaled_data = scaler.transform(input_data)
        lstm_input = scaled_data.reshape(1, 1, 4)
        prediction = model.predict(lstm_input)
        scaler_y = pickle.load(open("models/scaler_lstm_y.pkl", "rb"))
        prediction_close = scaler_y.inverse_transform(prediction[0])[0][0]
        print(prediction_close)

    return prediction_close
