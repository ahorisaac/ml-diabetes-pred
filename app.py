# -- import the dependencies 
import streamlit as st
import time as t

import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm

# -- data collection and analysis
# -- PIMA Diabetes Dataset

# -- load the diabetes dataset into pandas DataFrame
diabetes_dataset = pd.read_csv('diabetes.csv')

# -- separating the data and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
y = diabetes_dataset['Outcome']

# -- data standardization 
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data

# -- training 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, stratify=y, random_state=2)

# -- training the model
classifier = svm.SVC(kernel='linear')

# -- training the support vector machine classifier
classifier.fit(X_train, y_train)

# -- making a predictive system 
def predict_diab(preg, gluc, bp, sktn, insu, bmi, dpfn, age):
    input_data = (int(preg), int(gluc), int(bp), int(sktn), int(insu), float(bmi), (dpfn), int(age),)

    # -- change the input data to numpy array 
    input_data_as_nparr = np.asarray(input_data)

    # -- reshape the array as we are predicting for one instance 
    input_data_reshaped = input_data_as_nparr.reshape(1, -1)

    # -- standardize the input data 
    standardized_input_data = scaler.transform(input_data_reshaped)

    prediction = classifier.predict(standardized_input_data)

    if (prediction[0] == 0):
        st.success("The patient is NON-DIABETIC :white_check_mark:")
        return True
    else:
        st.error("The patient is DIABETIC :heavy_exclamation_mark:")    
        return False

# -- web application (code)

# -- diabetes features input
with st.columns(3)[1]:
    st.image("./images/icon.png", width=3**4)

st.title("Diabetes Prediction Application")

st.subheader("Diabetes Prediction Form")

with st.form("diabetes_pred_form", clear_on_submit=True):
    pregnancies_input = st.text_input("Pregnancies")

    glucose_input = st.text_input("Glucose")

    blood_pressure_input = st.text_input("Blood Pressure")

    skin_thickness_input = st.text_input("Skin Thickness")

    insulin_input = st.text_input("Insulin")

    bmi_input = st.number_input("BMI")

    dpf_input = st.number_input("Diabetes Pedigree Function")

    age_input = st.text_input("Age")

    # -- prediction, form submit button 
    submitted = st.form_submit_button("Predict", type="primary", help="click to predict diabetes status")
    
    if submitted:
        predict_diab(pregnancies_input, glucose_input, blood_pressure_input, skin_thickness_input, insulin_input, bmi_input, dpf_input, age_input)

