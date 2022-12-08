import streamlit as st
import numpy as np
from sklearn.svm import SVR 
import matplotlib.pyplot as plt 
import pandas as pd 

# %matplotlib inline
st.title('Modelo 1')
"""##Cargas datos

Los datos de Nintendo Co. serán cargados desde Yahoo Finance
"""

!pip install yfinance

import yfinance as yf

data = yf.download('NTDOY')

#Verificar data
data

"""##Preparar datos

Se elimina Date como indice y se convierte en columna
"""

data = data.reset_index()
data

"""Eliminamos 6532 filas y solo nos quedamos con 21 """

n = 6532
  
data.drop(data.tail(n).index, inplace = True) 
data

data.info()

"""##Crear modelo y predecir

Guardar datos
"""

import datetime
def get_data(data):  
    df = data.copy()
    df['Date']=df['Date'].astype(str)
    df['Date'] = df['Date'].str.split('-').str[2]
    df['Date'] = pd.to_numeric(df['Date'])
    return [ df['Date'].tolist(), df['Close'].tolist() ] # Convert Series to list
dates, prices = get_data(data)

"""Crear modelo"""

# predict and plot function
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

"""Obtener predicción"""

predicted_price = predict_prices(dates, prices, [21])

predicted_price

"""##Segundo caso"""

data = yf.download('NTDOY')

n = 6532
  
data.drop(data.head(n).index, inplace = True) 
data

data = data.reset_index()
import datetime
def get_data(data):  
    df = data.copy()
    df['Date']=df['Date'].astype(str)
    df['Date'] = df['Date'].str.split('-').str[2]
    df['Date'] = pd.to_numeric(df['Date'])
    return [ df['Date'].tolist(), df['Close'].tolist() ] # Convert Series to list
dates, prices = get_data(data)

# predict and plot function
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

predicted_price = predict_prices(dates, prices, [31])

predicted_price
