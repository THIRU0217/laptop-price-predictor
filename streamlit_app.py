import streamlit as st
import pandas as pd
import joblib

# Load saved components
model = joblib.load("laptop_price_model.pkl")
scaler = joblib.load("scaler.pkl")
le_dict = joblib.load("label_encoders.pkl")

st.title("💻 Laptop Price Predictor")
st.write("Fill in the specifications to estimate laptop price.")

# User input form
def get_user_input():
    data = {}
    for col, encoder in le_dict.items():
        data[col] = st.selectbox(f"{col}", encoder.classes_)
    for col in ["RAM", "SSD", "HDD", "Display_size", "Weight"]:
        data[col] = st.number_input(col, min_value=0.0)
    return pd.DataFrame([data])

input_df = get_user_input()

if st.button("Predict Price"):
    try:
        # Encode
        for col in le_dict:
            input_df[col] = le_dict[col].transform(input_df[col])
        # Scale
        scaled_input = scaler.transform(input_df)
        # Predict
        price = model.predict(scaled_input)[0]
        st.success(f"💰 Predicted Price: ₹{price:,.2f}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
