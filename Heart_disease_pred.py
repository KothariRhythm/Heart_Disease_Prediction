# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 21:01:13 2023

@author: rhythm
"""

import pickle
import streamlit as st

# loading the models
rfc_model = pickle.load(open('/mount/src/heart_disease_prediction/heart_disease_trained_model_rfc.sav','rb'))
dt_model = pickle.load(open('/mount/src/heart_disease_prediction/heart_disease_trained_model_dt.sav','rb'))
lr_model = pickle.load(open('/mount/src/heart_disease_prediction/heart_disease_trained_model_lr.sav','rb'))
standardize = pickle.load(open('/mount/src/heart_disease_prediction/StandardFunction.sav','rb'))

#title of web page
st.title('Heart Disease Prediction using Machine Learning')

# getting input from the user
col1, col2, col3 = st.columns(3)

with col1:
    Age = st.number_input('Age of Patient')
with col2:
    Sex = st.number_input('Gender of Patient')
with col3:
    ChestPainType = st.number_input('Chestpain Type')
    
with col1:
    RestingBP = st.number_input('Resting Blood Pressure')
with col2:
    Cholesterol = st.number_input('Cholestrol level of Patient')
with col3:
    FastingBS = st.number_input('Fasting Blood Sugar')
    
with col1:
    RestingECG = st.number_input('Resting ECG')
with col2:
    MaxHR = st.number_input('Max Heart Rate')
with col3:
    ExerciseAngina = st.number_input('Exercise Angina (0->No,1->Yes)')
    
with col1:
    Oldpeak = st.number_input('Old peak')
with col2:
    ST_Slope = st.number_input('ST Slope')
    
    
#standardizing data for prediction
mean = [53.510893246187365, 0.789760348583878, 0.7810457516339869, 132.39651416122004, 198.7995642701525, 0.23311546840958605, 0.9891067538126361, 136.80936819172112, 0.40413943355119825, 0.8873638344226579, 1.3616557734204793]
std = [9.43261650673201, 0.4077008804691498, 0.956519383238432, 18.5141541199078, 109.38414455220348, 0.423045624739303, 0.6316714317993976, 25.4603341382503, 0.49099221882320604, 1.0665701510493257, 0.6070561850472695]

input_data = [Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope]
input_data_std = [0,0,0,0,0,0,0,0,0,0,0]

for i in range(11):
    input_data_std[i] = (float(input_data[i]) - mean[i]) / std[i]
    
#code for prediction
heart = ''

#button for prediction
if st.button('Heart disease test Result'):
    heart_pred_rfc = rfc_model.predict([[Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope]])
    heart_pred_dt = dt_model.predict([input_data_std])
    heart_pred_lr = rfc_model.predict([[Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope]])

    if ((heart_pred_dt[0] + heart_pred_lr[0] + heart_pred_rfc[0])/3 > 0.5):
        heart = 'The patient has Heart Disease'
    else:
        heart = 'The patient does not have Heart Disease'
        
st.success(heart)



