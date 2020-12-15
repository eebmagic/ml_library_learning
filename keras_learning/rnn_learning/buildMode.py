print("Importing Libraries...")
import numpy as np
from random import randint
import tensorflow as tf
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
print("Finished importing.")
'''
Helpful link:
https://stackoverflow.com/questions/44873387/varying-sequence-length-in-keras-without-padding
'''

# Build data
X = list()
Y = list()
X = [x for x in range(5, 301, 5)]
Y = [y for y in range(20, 316, 5)]
X = np.array(X).reshape(20, 3, 1)
Y = np.array(Y).reshape(20, 3, 1)

print(X, X.shape)
print(Y, Y.shape)
quit()


model = Sequential()
# encoder layer
model.add(LSTM(100, activation='relu', input_shape=(None, 1)))
# model.add(LSTM(100, activation='relu'))

# repeat vector
model.add(RepeatVector(3))

# decoder layer
model.add(LSTM(100, activation='relu', return_sequences=True))

model.add(TimeDistributed(Dense(1)))

# Compile and print summary
model.compile(optimizer='adam', loss='mse')
model.summary()

# Train the mode
for x, y in zip(X, Y):
    # history = model.fit(x, y, epochs=600, validation_split=0.2, verbose=1, batch_size=1)
    history = model.train(x, y, epochs=600, validation_split=0.2, verbose=1, batch_size=1)


# Test with a sample
test_input = np.array([300, 305, 310]).reshape((1, 3, 1))
predictions = model.predict(test_input, verbose=0)
print(test_input)
print(predictions)