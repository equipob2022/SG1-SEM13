import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout,GRU
from keras import optimizers 
import pandas_datareader as datas

def app():
    st.title('Model 3 - GRU')
    #start = '2019-08-18'
    #end = '2022-01-20'
    start = st.date_input('Start' , value=pd.to_datetime('2019-08-18'))
    end = st.date_input('End' , value=pd.to_datetime('today'))
    
    st.title('Predicción de tendencia de acciones')

    user_input = st.text_input('Introducir cotización bursátil' , 'NTDOY')

    df = datas.DataReader(user_input, 'yahoo', start, end)
    
    # Describiendo los datos
    st.subheader('Tabla de datos') 
    st.write(df)
    st.subheader('Resumen de datos') 
    st.write(df.describe())
    st.subheader('Gated Recurrent Unit') 
    st.write('Valores evaluados')

    
    #Elegir atributos
    dataset = pd.DataFrame(df['Close'])
    fig = plt.figure(figsize = (12,6))
    plt.plot(dataset, linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('Precio de cierre')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.title('Precio de cierre')
    st.pyplot(fig)
    
    #Normalización de datos
    st.subheader('Normalización de MIN-MAX')
    dataset_norm = dataset.copy()
    dataset[['Close']]
    scaler = MinMaxScaler()
    dataset_norm['Close'] = scaler.fit_transform(dataset[['Close']])
    st.write(dataset_norm)
    
    # Gráfica de data normalizada
    fig2 = plt.figure(figsize=(10, 4))
    plt.plot(dataset_norm, linewidth=2)
    plt.xlabel('Date')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.title('Data Normalizada')
    st.pyplot(fig2)

    st.subheader('Partición de data')
    # Partición de datos en datos de entrenamiento, validación y prueba
    totaldata = dataset.values
    totaldatatrain = int(len(totaldata)*0.7)
    totaldataval = int(len(totaldata)*0.1)
    totaldatatest = int(len(totaldata)*0.2)

    # Datos en cada partición
    training_set = dataset_norm[0:totaldatatrain]
    val_set=dataset_norm[totaldatatrain:totaldatatrain+totaldataval]
    test_set = dataset_norm[totaldatatrain+totaldataval:]
    
    #Grafico de Partición
    st.subheader("Data entrenada")
    fig3 = plt.figure(figsize=(10, 4))
    plt.plot(training_set, linewidth=2)
    plt.xlabel('Date')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.title('Data entrenada')
    st.pyplot(fig3)
    st.write(training_set)
    
    st.subheader("Data de validación")
    fig4 = plt.figure(figsize=(10, 4))
    plt.plot(val_set, linewidth=2)
    plt.xlabel('Date')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.title('Datos de Validación')
    st.pyplot(fig4)
    st.write(val_set)
    
    st.subheader("Data de prueba")
    fig5 = plt.figure(figsize=(10, 4))
    plt.plot(test_set, linewidth=2)
    plt.xlabel('Date')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.title('Datos de Prueba')
    st.pyplot(fig5)
    st.write(test_set)
    
    #Sliding Windows
    # Iniciamos la variable lag
    lag = 2
    # Determinamos la función de sliding windows 
    def create_sliding_windows(data,len_data,lag):
        x=[]
        y=[]
        for i in range(lag,len_data):
            x.append(data[i-lag:i,0])
            y.append(data[i,0]) 
        return np.array(x),np.array(y)
    
    # Formateando los datos en un array para crear los sliding windows
    array_training_set = np.array(training_set)
    array_val_set = np.array(val_set)
    array_test_set = np.array(test_set)
    
    # Crear Sliding Window para la data de entrenamiento
    x_train, y_train = create_sliding_windows(array_training_set,len(array_training_set), lag)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

    # Crear Sliding Window para la data de validación
    x_val,y_val = create_sliding_windows(array_val_set,len(array_val_set),lag)
    x_val = np.reshape(x_val, (x_val.shape[0],x_val.shape[1],1))

    # Crear Sliding Window para la data de prueba
    x_test,y_test = create_sliding_windows(array_test_set,len(array_test_set),lag)
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
    
    # Modelo GRU (Gated Recurrent Unit
    # Hiperparametros a usar
    learning_rate = 0.0001
    hidden_unit = 64
    batch_size= 256
    epoch = 100
    
    # Arqutectura del modelo Gated Recurrent Unit
    regressorGRU = Sequential()
    
    # Primer layer(capa) GRU con dropout
    regressorGRU.add(GRU(units=hidden_unit, return_sequences=True, input_shape=(x_train.shape[1],1), activation = 'tanh'))
    regressorGRU.add(Dropout(0.2))

    # Segundo layer(capa) GRU con dropout
    regressorGRU.add(GRU(units=hidden_unit, return_sequences=True, activation = 'tanh'))
    regressorGRU.add(Dropout(0.2))

    # Tercer layer(capa) GRU con dropout
    regressorGRU.add(GRU(units=hidden_unit, return_sequences=False, activation = 'tanh'))
    regressorGRU.add(Dropout(0.2))
    
    # Output layer
    regressorGRU.add(Dense(units=1))
    
    # Compilando el modelo Gated Recurrent Unit
    regressorGRU.compile(optimizer=optimizers.Adam(lr=learning_rate),loss='mean_squared_error')
    
    # Ajustar la data de entrenamiento y la data de validación 
    pred = regressorGRU.fit(x_train, y_train, validation_data=(x_val,y_val), batch_size=batch_size, epochs=epoch)
    
    st.subheader("Grafica del comportamiento de la perdida (loss) tanto de lo entrenado como lo validado")
    fig6 = plt.figure(figsize=(10, 4))
    plt.plot(pred.history['loss'], label='train loss')
    plt.plot(pred.history['val_loss'], label='val loss')
    plt.title('modelo de pérdida')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    st.pyplot(fig6)
    
    st.subheader("Tabla de las variables train loss y val loss")
    learningrate_parameter = learning_rate
    train_loss=pred.history['loss'][-1]
    validation_loss=pred.history['val_loss'][-1]
    learningrate_parameter=pd.DataFrame(data=[[learningrate_parameter, train_loss, validation_loss]],
                                    columns=['Learning Rate', 'Training Loss', 'Validation Loss'])
    st.write(learningrate_parameter)
    
    #Implementación del modelo en la data de prueba
    y_pred_test = regressorGRU.predict(x_test)

    # Invertimos la normalización min-max
    y_pred_invert_norm = scaler.inverse_transform(y_pred_test)
    
    st.subheader('Comparación de los datos de prueba con los resultados predichos')
    datacompare = pd.DataFrame()
    datatest=np.array(dataset['Close'][totaldatatrain+totaldataval+lag:])
    datapred= y_pred_invert_norm

    datacompare['Data Test'] = datatest
    datacompare['Resultados predichos'] = datapred
    st.write(datacompare)
        
    st.subheader('Evaluación de los resultados predichos')    
    # Calculamos los valores de RMSE (Root Mean Square Error)
    def rmse(datatest, datapred):
        return np.round(np.sqrt(np.mean((datapred - datatest) ** 2)), 4)
    
    # Calculamos los valores de MAPE (Mean Absolute Percentage Error)
    def mape(datatest, datapred): 
        return np.round(np.mean(np.abs((datatest - datapred) / datatest) * 100), 4)
    
    st.write('.: Result Root Mean Square Error (RMSE) Prediction Model :',rmse(datatest, datapred))
    st.write('.: Result Mean Absolute Percentage Error (MAPE) Prediction Model : ', mape(datatest, datapred), '%')
    
    st.header("Grafico con los datos de prueba y los resultado de predicción")
    fig7=plt.figure(num=None, figsize=(10, 4), dpi=80,facecolor='w', edgecolor='k')
    plt.title('Gráfico de comparación de la Data Actual y la Data Predicha')
    plt.plot(datacompare['Data Test'], color='red',label='Data Test',linewidth=2)
    plt.plot(datacompare['Resultados predichos'], color='blue',label='Resultados predichos',linewidth=2)
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig7)



