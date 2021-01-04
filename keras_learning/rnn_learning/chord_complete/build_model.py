print("Importing libraries")
from build_data import build
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed

# Load the data
chords = build()
X, Y = chords
print(X)
print(Y)

# Build the model
model = Sequential()
model.add(LSTM(20, activation='relu', input_shape=(3, 1)))
model.add(Dense(1))

# Compile and print summary
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(X, Y, epochs=1000, verbose=2, batch_size=1)

# Save to file
model.save('models/chords.h5')
print("Saved model to file.")
