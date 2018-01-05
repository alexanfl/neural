from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D 
import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import Gnuplot
import numpy as np

x_min = 0
x_max = 2*np.pi
N = 100
x = np.linspace(x_min, x_max, N).reshape(-1,1)
y = np.cos(x)

# Select model
model = Sequential()

# add first layer of model and verify
model.add(Dense(100, input_dim=x.shape[1], kernel_initializer='he_normal', activation='relu'))
model.add(Dense(1000, kernel_initializer='he_normal', activation='relu'))
model.add(Dense(1, kernel_initializer='he_normal', activation='linear'))

# Compile model
model.compile(loss='mse',
              optimizer='sgd',
              metrics=['accuracy'])

# Fit model
model.fit(x, y, batch_size=32, epochs=100, verbose=1)
result = model.predict(x, batch_size=32)

plt = Gnuplot.Gnuplot()
plt.plot(result, y)
plt.show()
