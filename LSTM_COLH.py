# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 18:05:10 2019

@author: ABDO
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, LSTM

#from keras.callbacks import EarlyStopping
time_series=120
#fetures=0
name_compony="GOOG15"
df = pd.read_csv('GOOG15.csv')
#creating dataframe
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close' ,'Open','High','Low'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]
    new_data['Open'][i] = data['Open'][i]
    new_data['High'][i] = data['High'][i]
    new_data['Low'][i] = data['Low'][i]

#setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)
# get count cloumn
fetures=len(new_data.columns)
#split
def pridictOfDay(day):
    remaind_DAY= len(data)-day
    return remaind_DAY
#creating train and test sets
dataset = new_data.values
day_of_predict = pridictOfDay(100)
train = dataset[0:day_of_predict,:]
valid = dataset[day_of_predict:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(time_series,len(train)):
    x_train.append(scaled_data[i-time_series:i,:])
    y_train.append(scaled_data[i,:])
x_train, y_train = np.array(x_train), np.array(y_train)
y_train=y_train[:,0]
# reshape input to be [samples, time steps, features]
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],fetures))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=time_series, return_sequences=True, input_shape=(x_train.shape[1],fetures)))
model.add(Dropout(0.2))
model.add(LSTM(units=time_series, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=time_series))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
#earlyStop=EarlyStopping(monitor="val_loss",verbose=2,mode='min',patience=3)
#,callbacks=[earlyStop]
strName='modelsave/testYyyyy.h5'
if( os.path.exists(strName)):
   model.fit(x_train, y_train, epochs=1, batch_size=32)
   model.save(strName)
   my_model=load_model(strName)

#predicting 246 values, using past 120 from the train data
inputs = new_data[len(new_data) - len(valid) - time_series:].values
inputs = inputs.reshape(-1,fetures)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(time_series,inputs.shape[0]):
    X_test.append(inputs[i-time_series:i,:])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],fetures))

closing_price = my_model.predict(X_test)
################3
trainPredict_dataset_like = np.zeros(shape=(len(closing_price), 4) )
trainPredict_dataset_like[:,0] = closing_price[:,0]

###################
#closing_price = scaler.inverse_transform(closing_price)[:, [0]]
closing_price = scaler.inverse_transform(trainPredict_dataset_like)[:,0]
rms_test=np.sqrt(np.mean(np.power((valid[:,0]-closing_price),2)))
train = new_data[day_of_predict:]
valid = new_data[day_of_predict:]
valid['Predictions_Close'] = closing_price

#result
def plot(name_compony,index):
    plt.plot(train[index], color = 'red', label = "Real "+name_compony+" Stock "+index+" Price")
    plt.plot(valid["Predictions_"+index], color = 'blue', label = "Predicted "+name_compony+" "+index+" Stock Price")
    plt.title(name_compony+" "+index+" Stock Price Prediction")
    plt.xlabel('Time')
    plt.ylabel(name_compony+" "+index+" Stock Price")
    plt.legend()
    plt.show()
plot(name_compony,"Close")
#accuracy
def acc(index):
    count = 0
    for i in range(1,len(valid)):
         #print("predicted" + str(valid['Predictions'][i]) + "   " + "Expected" + str(valid['Close'][i]))
         if valid['Predictions_Close'][i] >= valid['Predictions_Close'][i-1]  and valid['Close'][i] >= valid['Close'] [i-1]:
            print("up")
            count += 1
         if valid['Predictions_Close'][i] < valid['Predictions_Close'][i-1]  and valid['Close'][i] < valid['Close'] [i-1]:
            print("down")
            count +=1
    print('======')
    print(str(count)+" is right predict from "+str(len(valid)))
    print(str((count/len(valid))*100) +'%')

acc("Close")
"""
from google.colab import files
files.upload()
"""