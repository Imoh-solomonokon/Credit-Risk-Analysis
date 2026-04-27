import streamlit as st # for creating the web app interface
import pandas as pd # for writing the input data in a dataframe format for the model to predict on
import joblib # for loading the pikl file of the trained model

# Load the trained model and encoder into dictionaries for use in the app
model = joblib.load("models/extra_trees_credit_model.pkl")
encoders = {col : joblib.load(f"models/{col}_label_encoder.pkl") for col in ["Sex","Housing", "Saving accounts", "Checking account"]}

# give the app a title and description
st.title("Credit Risk Prediction App")
st.write("Enter applicant information to predict if their credit risk is good or bad.")

# create input fields for user to enter applicant information
age = st.number_input("Age", min_value=18, max_value=80, value=30)
sex = st.selectbox("Sex", ["male", "female"])
job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)
housing = st.selectbox("Housing", ["own", "rent", "free"])
saving_account = st.selectbox("Saving accounts", ["little", "moderate", "rich", "quite rich"])
checking_account = st.selectbox("Checking account", ["little", "moderate", "rich"])
credit_amount = st.number_input("Credit amount", min_value=0, value=1000)
duration = st.number_input("Duration (months)", min_value=1, value=12)

# prepare the input data for prediction
input_data = pd.DataFrame({
    "Age": [age],
    "Sex": [encoders["Sex"].transform([sex])[0]],
    "Job": [job],
    "Housing": [encoders["Housing"].transform([housing])[0]],
    "Saving accounts": [encoders["Saving accounts"].transform([saving_account])[0]],
    "Checking account": [encoders["Checking account"].transform([checking_account])[0]],
    "Credit amount": [credit_amount],
    "Duration": [duration]
})

# make a prediction when the user clicks the button
if st.button("Predict Risk"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("The predicted credit risk is: **GOOD**")
    else:   
        st.error("The predicted credit risk is: **BAD**")