import streamlit as st
import pandas as pd
import joblib
import os

# Load model and scaler
model_path = os.path.abspath("../notebooks/violence_rf_model.pkl")
scaler_path = os.path.abspath("../notebooks/scaler.pkl")

# Load saved model and scaler
model = joblib.load(r'C:\Users\ADMIN\Desktop\Software\AI_Software Engineering\Final Project\gbvtrackai\notebooks\violence_rf_model.pkl')
scaler = joblib.load(r'C:\Users\ADMIN\Desktop\Software\AI_Software Engineering\Final Project\gbvtrackai\notebooks\scaler.pkl')

# Expected features
expected_features = list(scaler.feature_names_in_)

# Streamlit UI
st.title("GBV Prediction App")
st.subheader("Enter the required information below:")

# Collect inputs
age = st.number_input("Age", min_value=10, max_value=100, value=25)
education = st.selectbox("Education Level", ["None", "Primary", "Secondary", "Tertiary"])
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
employment = st.selectbox("Employment Status", ["Employed", "Unemployed", "Dependant"])

# Prepare input dictionary
input_dict = {
    "Age": age,
    "Education": education,
    "Children": children,
    "Employment": employment,

}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# One-hot encode categorical columns (if your model expects it)
input_df = pd.get_dummies(input_df)

# Align with scaler's expected features
for col in expected_features:
    if col not in input_df.columns:
        input_df[col] = 0  # fill missing features

# Reorder columns
input_df = input_df[expected_features]

# Predict
if st.button("Predict"):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)
    st.success(f"Prediction: {'Violence Likely' if prediction[0] == 1 else 'No Violence'}")
