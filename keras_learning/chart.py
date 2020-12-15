from tensorflow.keras.models import load_model
import loaders
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
import time

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

start = time.time()
# Load the built model
print("Loading model...")
model = load_model('models/diabetes_model.h5')
model.summary()

# Load data and split into input/output
print("Loading data...")
data = loaders.load('datasets/diabetes.csv')
data_input = data[:, :-1]
data_output = data[:, -1].reshape(-1, 1)

# Make predictions from data
print("Getting predictions from model...")
predictions = model.predict(x=data_input, batch_size=12, verbose=0)
rounded_predictions = np.argmax(predictions, axis=-1)

correct_count = 0
for i in range(len(rounded_predictions)):
    # print(data_output[i], rounded_predictions[i])
    if data_output[i] == rounded_predictions[i]:
        correct_count += 1
print(f"Accuracy: {correct_count} / {len(rounded_predictions)} = {100 * correct_count / len(rounded_predictions):.2f}%")

# plot confusion matrix
print("Graphing matrix...")
confuse = confusion_matrix(y_true=data_output, y_pred=rounded_predictions)
cm_plot_labels = ['no_disease_present', 'disease_present']
plot_confusion_matrix(cm=confuse, classes=cm_plot_labels, title='Confusion Matrix')
