print("Importing libraries...")
from tensorflow.keras.models import load_model
import loaders
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

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


# Load model
print("Loading model...")
model = load_model('models/census_model.h5')
model.summary()

# Load data and split into input/output
print("Loading data...")
data = loaders.load_with_strings('datasets/census/adult.test', padsize=110, skippad=2)
print(data.shape)
test_input = data[:, :-2]
test_output = data[:, -1].reshape(-1, 1)
print(test_input.shape)
print(test_output.shape)

# Make predictions for testing data
print("Getting predictions...")
predictions = model.predict(x=test_input, batch_size=100, verbose=0)
rounded_predictions = np.argmax(predictions, axis=-1)

correct = 0
under_correct = 0
under_total = len(test_output) - test_output.sum()
over_correct = 0
over_total = test_output.sum()
for i in range(len(rounded_predictions)):
    if test_output[i] == rounded_predictions[i]:
        correct += 1
        if test_output[i] == 1:
            over_correct += 1
        else:
            under_correct += 1
print(f"Accuracy: {correct} / {len(test_output)} = {100 * correct / len(test_output):.2f}%")
print(f"Under: {under_correct} / {under_total} = {100 * under_correct / under_total:.2f}%")
print(f"Over: {over_correct} / {over_total} = {100 * over_correct / over_total:.2f}%")


# plot confusion matrix
print("\nGraphing matrix...")
confuse = confusion_matrix(y_true=test_output, y_pred=rounded_predictions)
cm_plot_labels = ['less_than_50K', 'greater_than_50K']
plot_confusion_matrix(cm=confuse, classes=cm_plot_labels, title='Confusion Matrix')