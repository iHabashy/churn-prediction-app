#!/usr/bin/env python
# coding: utf-8

# In[1]:



import streamlit as st
import joblib
import numpy as np

model = joblib.load("D:\iTi\Data_Mining\churn_classification.pkl")

st.title("Customer Churn Prediction")
st.sidebar.header("Customer Parameters")

age = st.sidebar.slider("Age", 18, 80, 30)
tenure = st.sidebar.slider("Tenure (years)", 0, 20, 5)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

sex = 1 if gender == "Male" else 0
if st.button("Predict"):

    data = np.array([[age, tenure, sex]])

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    if prediction == 1:
        st.error("Customer will churn")
    else:
        st.success("Customer will NOT churn")

    st.write("Churn Probability:", round(probability,2))


# In[ ]:





# In[ ]:




