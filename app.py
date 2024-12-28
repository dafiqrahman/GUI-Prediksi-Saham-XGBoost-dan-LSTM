import streamlit as st
import pandas as pd
# make it center

st.markdown("""
<style>
.center {
    text-align: center;

}
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image('logo.png', width=200,)

with col3:
    st.write(' ')


st.markdown('<h1 class="center">Prediksi Harga Saham PT Kalbe Farma Tbk (KLBF) Menggunakan Metode XGBoost dan LSTM</h1>',
            unsafe_allow_html=True)
st.markdown('<h4 class="center">Oleh :</h4>',
            unsafe_allow_html=True)
st.markdown('<p class="center">Chandra Putra C</p>',
            unsafe_allow_html=True)
