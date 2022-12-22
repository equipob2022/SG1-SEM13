import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pandas_datareader import data as pdr
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR 
from sklearn import metrics
import plotly.express as px
from stockai import Stock
import datetime

def app():
    st.title('Model 1 - SVR')
    
    #start = '2019-08-18'
    #end = '2022-01-20'
    start = st.date_input('Start' , value=pd.to_datetime('2019-08-18'))
    end = st.date_input('End' , value=pd.to_datetime('today'))
    
    st.title('Predicción de tendencia de acciones')

    user_input = st.text_input('Introducir cotización bursátil' , 'NTDOY')

    lista = [user_input]

    y_symbols = ['SCHAND.NS', 'TATAPOWER.NS', 'ITC.NS']

    df = pdr.get_data_yahoo([user_input], start,end)
    
    # Describiendo los datos
    st.subheader('Tabla de datos') 
    st.write(df)
    st.subheader('Resumen de datos') 
    st.write(df.describe())
    st.subheader('Support Vector Regression') 
    st.write('Valores evaluados')
    df = df.reset_index()
    def get_data(df):  
        df['Date']=df['Date'].astype(str)
        df['Date'] = df['Date'].str.split('-').str[2]
        df['Date'] = pd.to_numeric(df['Date'])
        return [ df['Date'].tolist(), df['Close'].tolist() ] 
    dates, prices = get_data(df)
    st.write(prices)  
    def predict_prices(dates, prices, x):
        dates = np.reshape(dates,(len(dates), 1)) # convert to 1xn dimension
        x = np.reshape(x,(len(x), 1))
    
        svr_lin  = SVR(kernel='linear', C=1e3)
        svr_poly = SVR(kernel='poly', C=1e3, degree=2)
        svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    
        # Fit regression model
        svr_lin .fit(dates, prices)
        svr_poly.fit(dates, prices)
        svr_rbf.fit(dates, prices)
    
        plt.scatter(dates, prices, c='k', label='Data')
        plt.plot(dates, svr_lin.predict(dates), c='g', label='Linear model')
        plt.plot(dates, svr_rbf.predict(dates), c='r', label='RBF model')    
        plt.plot(dates, svr_poly.predict(dates), c='b', label='Polynomial model')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Support Vector Regression')
        plt.legend()
        plt.show()
        return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0] 
    st.subheader('Prediccion') 
    predicted_price = predict_prices(dates, prices, [21])
    st.write(predicted_price)
    RVF, LIN, POLY=predicted_price
    valores = {
        'modelo' : ['RVF', 'LIN', 'POLY'],
        'valor': [RVF, LIN, POLY]
    }
    
    valores = pd.DataFrame(valores)  
    ### Gráfica de las métricas
    st.subheader('Valores predecidos segun el modelo') 
    fig = px.bar(        
        valores,
        x = "modelo",
        y = "valor",
        title = "Valores predecidos segun el modelo",
        color="modelo"
    )
    st.plotly_chart(fig)
