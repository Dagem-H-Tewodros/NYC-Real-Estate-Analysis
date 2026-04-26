import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.set_page_config(page_title="NY Real Estate Predictor", page_icon="🏙️", layout="centered")

@st.cache_resource
def load_model():
    with open('ny_realestate_model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

st.title("🏙️ NY Real Estate Price Estimator")
st.caption("Gradient Boosting model trained on New York listings")

if not model_loaded:
    st.error("Model file not found. Make sure ny_realestate_model.pkl is in the same folder.")
    st.stop()

st.subheader("Property Details")
col1, col2 = st.columns(2)
with col1:
    beds = st.number_input("Bedrooms",  min_value=0,   max_value=20,    value=3,    step=1)
    sqft = st.number_input("Square Feet", min_value=100, max_value=20000, value=1200, step=50)
with col2:
    baths = st.number_input("Bathrooms", min_value=0.0, max_value=20.0, value=2.0, step=0.5)

st.subheader("Location")
col3, col4 = st.columns(2)
with col3:
    lat = st.number_input("Latitude",  value=40.7128, format="%.4f")
with col4:
    lon = st.number_input("Longitude", value=-74.0060, format="%.4f")

if st.button("Estimate Price"):
    input_df = pd.DataFrame({
        'beds':  [beds],
        'baths': [baths],
        'sqft':  [sqft],
        'lat':   [lat],
        'lon':   [lon],
    })
    price = np.exp(model.predict(input_df)[0])

    st.success(f"### Estimated Price: ${price:,.0f}")
    st.caption(f"Indicative range: ${price*0.9:,.0f} – ${price*1.1:,.0f}")