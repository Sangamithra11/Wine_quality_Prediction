import streamlit as st
import numpy as np
import joblib

# Load the model
model=joblib.load('wine_quality_model.pkl')

st.title("Wine Quality Prediction")
fixed_acidity=st.number_input("Fixed Acidity",0.0,20.0,7.4)
volatile_acidity=st.number_input("Volatile Acidity",0.0,5.0,0.7)
citric_acid=st.number_input("Citric Acid",0.0,5.0,0.0)
residual_sugar=st.number_input("Residual Sugar",0.0,20.0,1.9)
chlorides=st.number_input("Chlorides",0.0,0.5,0.076)
free_sulfur_dioxide=st.number_input("Free Sulfur Dioxide",0.0,100.0,11.0)
total_sulfur_dioxide=st.number_input("Total Sulfur Dioxide",0.0,300.0,34.0)
density=st.number_input("Density",0.9900,1.0100,0.9978)
pH=st.number_input("pH",2.0,5.0,3.51)
sulphates=st.number_input("Sulphates",0.0,2.0,0.56)
alcohol=st.number_input("Alcohol",5.0,20.0,9.4)

inputs=np.array([[fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol]])
if st.button("Predict Quality"):
    prediction=model.predict(inputs)
    st.success(f"The predicted quality of the wine is: {prediction[0]}")
    