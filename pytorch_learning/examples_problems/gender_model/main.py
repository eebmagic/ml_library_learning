import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# Load data
names = ['color', 'music', 'beverage', 'soft_drink', 'gender']
data = pd.read_csv('data.csv', names=names)
data = data[1:]

# Convert data to ints
ints = pd.DataFrame()
ints['color_ind'] = [list(set(data['color'].values)).index(x) for x in data['color']]
ints['music_ind'] = [list(set(data['music'].values)).index(x) for x in data['music']]
ints['beverage_ind'] = [list(set(data['beverage'].values)).index(x) for x in data['beverage']]
ints['soft_drink_ind'] = [list(set(data['soft_drink'].values)).index(x) for x in data['soft_drink']]
ints['gender_ind'] = [list(set(data['gender'].values)).index(x) for x in data['gender']]

# Make normalized Data
normalized = (ints - ints.min()) / (ints.max() - ints.min())
print(normalized)

# Split into train/test data
train_perc = 0.7
split_index = int(len(normalized) * train_perc)
normalized = normalized.sample(frac=1)

train_data = normalized[:split_index]
test_data = normalized[split_index:]

train_in = train_data[train_data.columns.difference(['gender_ind'])]
train_out = train_data.gender_ind

test_in = test_data[test_data.columns.difference(['gender_ind'])]
test_out = test_data.gender_ind

# Make tensors
N = len(train_data)
D_in = len(train_in.columns)
H = 3
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
# training_iterations = 1_000_000
training_iterations = 60_000
# training_iterations = 10_000

losses = []
from tqdm import tqdm
start = time.time()
for t in tqdm(range(training_iterations)):
    # Feed-forward and calc loss
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred - y).pow(2).sum()
    if t % 3000 == 0:
        print(t, loss.data.item())
    losses.append(loss.data.item())

    # backprop along gradient
    loss.backward()
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data
    w1.grad.data.zero_()
    w2.grad.data.zero_()

print(f"training duration: {time.time() - start}")

y_test_pred = test_x.mm(w1).clamp(min=0).mm(w2)
out = y_test_pred.detach().numpy()[:]
out = out.reshape(len(out))
rounded = np.around(out)
original = rounded
# rounded = 1 - rounded
correct = test_y.detach().numpy()[:]
print(original)
print(out)
print(rounded)
print(correct)
matching = 0
for i in range(len(rounded)):
    if rounded[i] == correct[i]:
        matching += 1
acc = matching / len(out)
print(f"MATHCING: {matching}")
print(f"ACCURACY: {acc}")

# graph out losses
# plt.plot(list(range(training_iterations)), losses)
# plt.show()