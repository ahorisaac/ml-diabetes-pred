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


# -- web application (code)

# -- diabetes features input
st.title("Diabetes Prediction Application")

with st.form("diabetes_pred_form"):
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
