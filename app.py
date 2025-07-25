import streamlit as st
import pickle
import pandas as pd

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🎓 Student Grade Predictor")
st.write("Enter your study habits below to predict your grade.")

hours = st.slider("📘 Hours Studied", 0, 12, 1)
attendance = st.slider("🏫 Attendance (%)", 0, 100, 75)
sleep = st.slider("🛌 Sleep Hours", 0, 12, 6)

if st.button("📊 Predict Grade"):
    input_df = pd.DataFrame([[hours, attendance, sleep]],
                            columns=["Hours_Studied", "Attendance", "Sleep_Hours"])
    prediction = model.predict(input_df)
    st.success(f"🎯 Predicted Grade: {round(prediction[0], 2)}")
