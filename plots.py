import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def app(car_df): 
    st.markdown("<p style='color:blue;font-size:25px'>This app uses <b>Linear regression</b> to predict the price of a car based on your inputs.", unsafe_allow_html = True) 
    st.subheader("Select Values:")
    car_wid = st.slider("Car Width", float(car_df["carwidth"].min()), float(car_df["carwidth"].max()))     
    eng_siz = st.slider("Engine Size", int(car_df["enginesize"].min()), int(car_df["enginesize"].max()))
    hor_pow = st.slider("Horse Power", int(car_df["horsepower"].min()), int(car_df["horsepower"].max()))    
    drw_fwd = st.radio("Is it a forward drive wheel car?", ("Yes", "No"))
    if drw_fwd == 'No':
        drw_fwd = 0
    else:
        drw_fwd = 1
    com_bui = st.radio("Is the car manufactured by Buick?", ("Yes", "No"))
    if com_bui == 'No':
        com_bui = 0
    else:
        com_bui = 1
    
    if st.button("Predict"):
        st.subheader("Prediction results:")
        price, score, car_r2, car_mae, car_msle, car_rmse = prediction(car_df, car_wid, eng_siz, hor_pow, drw_fwd, com_bui)
        st.success("The predicted price of the car: ${:,}".format(int(price)))
        st.info("Accuracy score of this model is: {:2.2%}".format(score))
        st.info(f"R-squared score of this model is: {car_r2:.3f}")  
        st.info(f"Mean absolute error of this model is: {car_mae:.3f}")  
        st.info(f"Mean squared log error of this model is: {car_msle:.3f}")  
        st.info(f"Root mean squared error of this model is: {car_rmse:.3f}")
