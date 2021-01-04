print("Importing libraries...")
from tensorflow.keras.models import load_model
import numpy as np

# Load the model from file
print("Loading model...")
model = load_model('models/chords.h5')
model.summary()

# Test with a small problem
print("\nSTARTING TESTS:")
test_sets = [([50, 53, 57], 60),
             ([60, 63, 67], 70),
             ([55, 59, 62], 66),
             ([40, 43, 47], 50),
             ([40, 44, 47], 51)]

correct_count = 0
for ins, out in test_sets:
    test_x = np.array(ins)
    test_x = test_x.reshape(1, 3, 1)
    print(f"\ninput: {test_x.flatten()}")
    test_prediction = model.predict(test_x, verbose=0)
    print(f"output: {test_prediction.flatten()} = {round(test_prediction.flatten()[0])} (correct: {out})")
    if int(round(test_prediction.flatten()[0])) == out:
        correct_count += 1
    else:
        print("INCORRECT PREDICTION")
print(f"\nTotal correct: {correct_count} / {len(test_sets)}\n")