# File Guide:

### 7ths.mid
Source midi file for training data. Has several 7th chords.

### build_data.py
Splits the chords from 7ths.mid into arrays of 4 or input/output arrays

### build_model.py
Builds the LSTM model, trains on the data, & exports to file in models/

### test_mode.py
Loads the LSTM model from file in models/ and tests it on a few chords. (Test set is relateively close to training set and not within full scope of possible midi signals)
