import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def app():
	st.header("car price prediction app")
	st.text("""This web app allows a user to predict the prices of a car based on their engine size,
	 horse power, dimensions and the drive wheel type parameters""")