import streamlit as st
import pickle

import pandas as pd

# Load your machine learning model
model=pickle.load(open("crop.pkl",'rb'))




st.title("Crop Recommendation System")


def user_report():
    N=st.slider("Nitrogen",0,200,1)
    P=st.slider("phosphorous",0,200,1)
    K= st.slider("potassium", 0, 200, 1)
    temperature = st.slider("temperature", 0, 100, 1)
    humidity= st.slider("humidity", 0, 200, 1)
    ph = st.slider("ph", 1, 14, 1)
    rainfall = st.slider("rainfall", 0, 200, 1)


    user_report_data={
        'N':N,
        'P':P,
        'K':K,
        'temperature':temperature,
        'humidity':humidity,
        'ph':ph,
        'rainfall':rainfall
    }

    report_data=pd.DataFrame(user_report_data,index=[0])
    return report_data


user_data=user_report()
st.header("Soil features: ")
st.write(user_data)

crop=model.predict(user_data)
st.subheader('Crop recommended: ')
st.subheader(str(crop))



