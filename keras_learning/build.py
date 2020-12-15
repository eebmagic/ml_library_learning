'''
NOTE: Following tutorial: Keras with TensorFlow Course
                          - Python Deep Learning and Neural Networks for Beginners Tutorial
URL:
https://www.youtube.com/watch?v=qFJeN9V1ZsI&t=1185s
'''

print("Importing libraries...")
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import loaders


def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()


print("Loading data")
scaled_data = loaders.load('datasets/diabetes.csv')

# Separate off a testing set
split_index = len(scaled_data) // 12
test_data = scaled_data[:split_index]
scaled_data = scaled_data[split_index:]

# Separate into intput/output arrays
train_input = scaled_data[:, :-1]
train_output = scaled_data[:, -1].reshape(-1, 1)
test_input = test_data[:, :-1]
test_output = test_data[:, -1].reshape(-1, 1)
print(train_input[:5])
print(train_output[:5])
print(train_input.shape)

'''
NOTE: For running on GPU hardware check video at 19:45
'''

# Build the structure
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

layers = [
    # First hidden layer
    Dense(units=10, input_shape=(8, ), activation='relu'),
    # Second hidden layer
    Dense(units=32, activation='relu'),
    # Output layer - disease present or disease NOT present
    Dense(units=2, activation='softmax')
    # Dense layers are fully connected layers
]
model = Sequential(layers)
model.summary()

# Setup and train the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

print("Finished compiling model")
print("Starting training")

# model.fit(x=train_input, y=train_output, validation_split=0.1, batch_size=12,
#     epochs=1000, shuffle=True, verbose=2)
model.fit(x=train_input, y=train_output, batch_size=12,
    epochs=1000, shuffle=True, verbose=2)

# Test model with test set
predictions = model.predict(x=test_input, batch_size=12, verbose=0)
rounded_predictions = np.argmax(predictions, axis=-1)
correct_prediction_count = 0
for i in range(len(predictions)):
    print(predictions[i], rounded_predictions[i],
        test_output[i], test_output[i] == rounded_predictions[i])
    if test_output[i] == rounded_predictions[i]:
        correct_prediction_count += 1
print(f"{correct_prediction_count} / {len(predictions)} ", end='')
print(f"= {100 * correct_prediction_count/len(predictions):.2f}%")

# Save model to file
import os.path
if os.path.isfile('models/diabetes_model.h5') is False:
    model.save('models/diabetes_model.h5')
    print("Automatically saved model to file")
elif not input("Save model to file? (Y/n): ").lower().startswith('n'):
    model.save('models/diabetes_model.h5')
    print("Automatically saved model to file")
