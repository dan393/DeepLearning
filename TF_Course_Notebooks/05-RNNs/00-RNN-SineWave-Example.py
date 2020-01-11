
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0,50,501)
y = np.sin(x)
plt.plot(x,y)
plt.show()

df = pd.DataFrame(data = y, index = x, columns = ['Sine'])

test_percent = 0.1


test_point = np.round(len(df)*test_percent)
test_ind= int(len(df) -test_point)

train = df.iloc[:test_ind]
test = df.iloc[test_ind:]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(train)

scaled_train = scaler.transform(train)
scaler_test = scaler.transform(test)

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

length = 25
batch_size = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length = length, batch_size = batch_size)
X, y = generator[0]

