print("Importing libraries...")
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.utils import shuffle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import loaders


# Load the data from the file
print("Loading data...")
names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'output']
scaled_data = loaders.load_with_strings('datasets/census/adult.data', names=names)
scaled_data = shuffle(scaled_data)
print(scaled_data.shape)

# Sort into inputs and outputs
training_input = scaled_data[:, :-2]
training_output = scaled_data[:, -1].reshape(-1, 1)
print(training_input.shape)
print(training_output.shape)

# Build structure
layers = [
    Dense(units=110, input_shape=(108, ), activation='relu'),
    Dense(units=110, activation='relu'),
    Dense(units=110, activation='relu'),
    Dense(units=110, activation='relu'),
    Dense(units=110, activation='relu'),
    Dense(units=2, activation='softmax')
]
model = Sequential(layers)
model.summary()

# Setup the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# Start training
model.fit(x=training_input, y=training_output, batch_size=100,
    epochs=1000, shuffle=True, verbose=2)

# Save the model to a file
model.save('models/census_model.h5')
print("Saved model to file")