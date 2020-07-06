import pandas as pd
import json
import urllib.request as url
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential

url = url.urlopen('https://api.covid19india.org/states_daily.json')
data = json.load(url)
data = data['states_daily']
df = pd.DataFrame(data)
df = df[df['status'] == 'Confirmed']
test = df.tail(15)
test = test.drop(['date','status'], axis=1)
test = test.astype(np.int64)
test['total'] = test.sum(axis=1)
test_set = test.iloc[:,39:40].values
minmax = MinMaxScaler()
scaled_test = minmax.fit_transform(test_set)
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(15,1)))
model.add(Dropout(0.5))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(units=50))
model.add(Dropout(0.5))
model.add(Dense(units=1))
model.compile(optimizer='rmsprop', loss='mean_squared_error')
model.load_weights('covid.h5')
test_x = test_set.reshape(1,15,1)
pred = model.predict(test_x)
pred = minmax.inverse_transform(pred)
print(int(pred[0][0]))

