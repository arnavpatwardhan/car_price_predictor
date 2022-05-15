import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import home
import data
import plots
import predict

st.set_page_config(page_title = None, 
                          page_icon = None, 
                          layout = 'centered', 
                          initial_sidebar_state = 'auto')

words_dict = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "eight": 8, "twelve": 12}
def num_map(series):
    return series.map(words_dict)
@st.cache()
def load_data():
  cars_df = pd.read_csv("car-prices.csv")
  car_companies = pd.Series([car.split(" ")[0] for car in cars_df['CarName']], index = cars_df.index)
  cars_df['car_company'] = car_companies
  cars_df.loc[(cars_df['car_company'] == "vw") | (cars_df['car_company'] == "vokswagen"), 'car_company'] = 'volkswagen'
  cars_df.loc[cars_df['car_company'] == "porcshce", 'car_company'] = 'porsche'
  cars_df.loc[cars_df['car_company'] == "toyouta", 'car_company'] = 'toyota'
  cars_df.loc[cars_df['car_company'] == "Nissan", 'car_company'] = 'nissan'
  cars_df.loc[cars_df['car_company'] == "maxda", 'car_company'] = 'mazda'
  cars_df.drop(columns= ['CarName'], axis = 1, inplace = True)
  cars_numeric_df = cars_df.select_dtypes(include = ['int64', 'float64']) 
  cars_numeric_df.drop(columns = ['car_ID'], axis = 1, inplace = True)
  cars_df[['cylindernumber', 'doornumber']] = cars_df[['cylindernumber', 'doornumber']].apply(num_map, axis = 1)
  car_body_dummies = pd.get_dummies(cars_df['carbody'], dtype = int)
  car_body_new_dummies = pd.get_dummies(cars_df['carbody'], drop_first = True, dtype = int)
  cars_categorical_df = cars_df.select_dtypes(include = ['object'])
  cars_dummies_df = pd.get_dummies(cars_categorical_df, drop_first = True, dtype = int)
  cars_df.drop(list(cars_categorical_df.columns), axis = 1, inplace = True)
  cars_df = pd.concat([cars_df, cars_dummies_df], axis = 1)
  cars_df.drop('car_ID', axis = 1, inplace = True)
  final_columns = ['carwidth', 'enginesize', 'horsepower', 'drivewheel_fwd', 'car_company_buick', 'price']
  return cars_df[final_columns]

final_cars_df=load_data()


pages_dict = {
             "Home": home,
             "View Data": data, 
             "Visualise Data": plots, 
             "Predict": predict
         }
st.sidebar.title("navigation")
user_choice = st.sidebar.radio("go to",tuple(pages_dict.keys()))
if user_choice=='Home':
  home.app()
else:
  selected_page=pages_dict[user_choice]
  selected_page.app(final_cars_df)
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

