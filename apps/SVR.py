import streamlit as st
import numpy as np
from sklearn.svm import SVR 
import matplotlib.pyplot as plt 
import pandas as pd 
!pip install yfinance
import yfinance as yf
# %matplotlib inline

def app():
    st.title('Modelo 1')

    """##Cargas datos

    data = yf.download('NTDOY')

    #Verificar data
    st.write(data)

