import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as datas
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import plotly.express as px
from stockai import Stock
import datetime

def app():
    st.title('Model 1 - SVR')
    
    #start = '2004-08-18'
    #end = '2022-01-20'
    start = st.date_input('Start' , value=pd.to_datetime('2004-08-18'))
    end = st.date_input('End' , value=pd.to_datetime('today'))
    
    st.title('Predicción de tendencia de acciones')

    user_input = st.text_input('Introducir cotización bursátil' , 'NTDOY')

    df = datas.DataReader(user_input, 'yahoo', start, end)
    
    # Describiendo los datos
    st.subheader('Tabla de datos') 
    st.write(df)
    st.subheader('Datos del 2004 al 2022') 
    st.write(df.describe())


    st.subheader('Support Vector Regression') 
    
    def get_data(df):  
        df['Open']=df['Open'].astype(str)
        df['Open'] = df['Open'].str.split('-').str[2]
        df['Open'] = pd.to_numeric(df['Open'])
    return [ df['Open'].tolist(), df['Close'].tolist() ] 
    
    dates, prices = get_data(df)
    
    st.write(dates, prices)
