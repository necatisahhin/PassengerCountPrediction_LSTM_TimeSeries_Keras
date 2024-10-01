
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense,LSTM  # I use keras over Tensorflow 2, so I don't need extra Keras libraries.
from tensorflow.keras.models import Sequential  # Tensorflow includes Keras libraries after 2nd version of Tensorflow
from sklearn.preprocessing import MinMaxScaler  # If you have Tensorflow 1, please upgrade your library using:

data = pd.read_csv('AirPassengers.csv')
data.head()

data.rename(columns={'#Passengers':'passengers'},inplace=True)

data = data['passengers']

type(data)

data

data=np.array(data).reshape(-1,1)

type(data)

plt.plot(data)
plt.show()

scaler = MinMaxScaler()
data = scaler.fit_transform(data)

len(data)

train = data[0:100,:]
test = data[100:,:]

def get_data(data, steps):      
    dataX = []
    dataY = []
    for i in range(len(data)-steps-1):
        a = data[i:(i+steps), 0]
        dataX.append(a)
        dataY.append(data[i+steps, 0])
    return np.array(dataX), np.array(dataY)

steps = 2

X_train, y_train = get_data(train, steps)
X_test, y_test = get_data(test, steps)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
model = Sequential()
model.add(LSTM(128, input_shape = (1, steps)))
model.add(Dense(64))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')

model.summary()

model.fit(X_train, y_train, epochs=25, batch_size=1)

y_pred = model.predict(X_test)

y_pred = scaler.inverse_transform(y_pred)
y_test = y_test.reshape(-1, 1)
y_test = scaler.inverse_transform(y_test)

plt.plot(y_test, label = 'real number of passengers')
plt.plot(y_pred, label = 'predicted number of passengers')
plt.ylabel('Months')
plt.ylabel('Number of passengers')
plt.legend()
plt.show()