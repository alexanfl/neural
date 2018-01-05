import numpy as np
np.random.seed(7)

import Gnuplot
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

X = np.linspace(0.0 , 2.0 * np.pi, 100).reshape(-1, 1)
Y = np.sin(X)

x_scaler = MinMaxScaler()
#y_scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
y_scaler = MinMaxScaler()

X = x_scaler.fit_transform(X)
Y = y_scaler.fit_transform(Y)

model = Sequential()
model.add(Dense(4, input_dim=X.shape[1], kernel_initializer='uniform', activation='relu'))
# model.add(Dense(4, input_dim=X.shape[1], kernel_initializer='uniform', activation='sigmoid'))
# model.add(Dense(4, input_dim=X.shape[1], kernel_initializer='uniform', activation='tanh'))
model.add(Dense(1, kernel_initializer='uniform', activation='linear'))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=100, batch_size=32, verbose=2)

res = model.predict(X, batch_size=32)

res_rscl = y_scaler.inverse_transform(res)

Y_rscl = y_scaler.inverse_transform(Y)

plt = Gnuplot.Gnuplot()
plt.plot(res_rscl)
