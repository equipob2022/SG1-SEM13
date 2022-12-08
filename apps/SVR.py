import streamlit as st
import numpy as np
from sklearn.svm import SVR 
import matplotlib.pyplot as plt 
import pandas as pd 
import yfinance as yf


def app():
    st.title('Modelo 1')



    data = yf.download('NTDOY')


    st.write(data)

