import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def app(car_df):
	st.header("view data")
	with st.beta_expander("view dataset"):
		st.table(car_df)
	st.subheader("columns description")
	if st.checkbox("show summary"):
		st.table(car_df.describe())
	beta_col1,beta_col2,beta_col3=st.beta_columns(3)
	with beta_col1:
		if st.checkbox("show all column names"):
			st.table(list(car_df.columns))
	with beta_col2:
		if st.checkbox("view column data type"):
			st.table(car_df.dtypes)
	with beta_col3:
		if st.checkbox("view column data"):
			column_data = st.selectbox("select column",tuple(car_df.columns))
			st.write(car_df[column_data])
			
