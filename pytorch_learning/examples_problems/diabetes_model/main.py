import pandas as pd
import torch
from torch.autograd import Variable
import time

# Load data
names = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
    'insulin', 'bmi', 'diabetes_pedigree_function', 'age', 'outcome']
data = pd.read_csv('diabetes.csv', names=names)
data = data[1:]
for col in data:
    data[col] = data[col].astype(float)
print(data)
print(data['outcome'].value_counts())
print(data['pregnancies'].value_counts())
import matplotlib.axes
import matplotlib.pyplot as plt
ax = data['pregnancies'].value_counts().plot.hist()
plt.show()
quit()

# Normalize data and split into train/test
normalized = (data - data.min()) / (data.max() - data.min())

train_perc = 0.7
split_index = int(len(data) * train_perc)

train_data = normalized[:split_index]
test_data = normalized[split_index:]

train_in = train_data[train_data.columns.difference(['outcome'])]
train_out = train_data.outcome

test_in = test_data[test_data.columns.difference(['outcome'])]
test_out = test_data.outcome

# Make tensors
N = len(train_data)
D_in = len(train_in.columns)
H = 10
D_out = 1

dtype = torch.FloatTensor
x = Variable(torch.FloatTensor(train_in.to_numpy()), requires_grad=False)
y = Variable(torch.FloatTensor(train_out.to_numpy()), requires_grad=False)

test_x = Variable(torch.FloatTensor(test_in.to_numpy()), requires_grad=False)
test_y = Variable(torch.FloatTensor(test_out.to_numpy()), requires_grad=False)

w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)


# Start iterations
learning_rate = 1e-6
training_iterations = 10000
losses = []
start = time.time()
for t in range(training_iterations):
    # Feed-forward and calculate loss
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.data.item())
    losses.append(loss.data.item())

    # backprop along gradient
    loss.backward()
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data
    w1.grad.data.zero_()
    w2.grad.data.zero_()

print(f"training duration: {time.time() - start}")

# Get testing accuracy
y_test_pred = test_x.mm(w1).clamp(min=0).mm(w2)
print(y_test_pred)
print(test_y)

import matplotlib.pyplot as plt
plt.plot(list(range(training_iterations)), losses)
plt.show()