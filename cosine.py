from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D 
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import Gnuplot
import numpy as np

x_min = 0
x_max = 2*np.pi
N = 100
x = np.linspace(x_min, x_max, N).reshape(-1,1)
y = (np.cos(x) + 1)/2.

# Select model
model = Sequential()

# add first layer of model and verify
model.add(Dense(11, input_dim=x.shape[1], kernel_initializer='he_normal', activation='sigmoid'))
model.add(Dense(19, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(11, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(1))

# Compile model
model.compile(loss='mean_squared_error',
              optimizer='SGD',
              metrics=['mean_squared_error'])

# Fit model
model.fit(x, y, batch_size=32, epochs=10000, verbose=1)
result = model.predict(x, batch_size=32)

g = Gnuplot.Gnuplot()
# g.plot(result, y)
plt.plot(x, result, x, y)
plt.show()
