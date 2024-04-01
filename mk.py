import streamlit as st
import joblib

import pandas as pd

# Load your machine learning model
model=joblib.load("crop.pkl")




st.title("Crop Recommendation System")


def user_report():
    N=st.slider("Nitrogen(in mg/kg)",0.0,200.00,1.00)
    P=st.slider("phosphorous(in mg/kg)",0.0,200.00,1.00)
    K= st.slider("potassium(in mg/kg)", 0.0, 200.00, 1.00)
    temperature = st.slider("temperature(in degree celcius)", 0.0, 50.00, 1.00)
    humidity= st.slider("humidity(in %)", 0.0, 200.00, 1.00)
    ph = st.slider("ph", 0.0, 14.00, 1.00)
    rainfall = st.slider("rainfall(in mm)", 0.0, 200.00, 1.00)


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
crop = ' '.join(map(str, crop))

if st.checkbox("Show Result"):
    st.subheader('This system suggest to grow {} in this soil.'.format(crop))


st.text("")
st.text("")
st.text("")
st.text("")

st.markdown("© All rights reserved.")




