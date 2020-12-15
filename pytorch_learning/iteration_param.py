import torch
from torch.autograd import Variable
from NeuralNetUtil import buildExamplesFromCensus
import time
from tqdm import tqdm

census = buildExamplesFromCensus()
train_data, test_data = census
print(len(train_data), len(test_data))
train_data_inputs = []
train_data_outputs = []
for x, y in train_data:
    train_data_inputs.append(x)
    train_data_outputs.append(y)

test_data_inputs = []
test_data_outputs = []
for x, y in test_data:
    test_data_inputs.append(x)
    test_data_outputs.append(y)

print(len(train_data[0][0]))
print(len(test_data[0][0]))

print(len(train_data_inputs), len(train_data_inputs[0]))
print(len(train_data_outputs), len(train_data_outputs[0]))

iteration_counts = []
accuracies = []
min_iters = 5
max_iters = 1000
skip_rate = 5
for training_iterations in range(min_iters, max_iters+1, skip_rate):
    N = len(train_data)
    H = 75
    D_in = 112
    D_out = 1

    dtype = torch.FloatTensor
    x = Variable(torch.FloatTensor(train_data_inputs), requires_grad=False)
    y = Variable(torch.FloatTensor(train_data_outputs), requires_grad=False)

    test_x = Variable(torch.FloatTensor(test_data_inputs), requires_grad=False)
    test_y = Variable(torch.FloatTensor(test_data_outputs), requires_grad=False)

    w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
    w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

    learning_rate = 1e-6
    start = time.time()
    print(f"Running with {training_iterations = }")
    for t in tqdm(range(training_iterations)):
        # Make a feed forward prediction
        y_pred = x.mm(w1).clamp(min=0).mm(w2)

        # Calculate loss
        # loss = sum of ((diff at each point in array) ^ 2)
        loss = (y_pred - y).pow(2).sum()

        # updates gradiants for all Variables with requires_grad set to True
        loss.backward()

        # Update weights regarding the gradients calculated with loss
        # adjust with consideration for learning rate
        w1.data -= learning_rate * w1.grad.data
        w2.data -= learning_rate * w2.grad.data

        # reset gradiant data from last backprop for next iteration
        w1.grad.data.zero_()
        w2.grad.data.zero_()

    print(f"training duration: {time.time() - start}")

    # Run on testing data
    y_test_pred = test_x.mm(w1).clamp(min=0).mm(w2)
    correct_total = test_y.eq(y_test_pred.long()).sum()
    total = len(y_test_pred)

    testing_accuracy = 100 * correct_total / total
    iteration_counts.append(training_iterations)
    accuracies.append(testing_accuracy)

    print(f"\nTESTING ACCURACY: {testing_accuracy:.2f}%")
    print(f"\tHidden: {H}\n\tIterations: {training_iterations}")


print("EXPORTING DATA TO FILE")
data = {}
for ind in range(len(iteration_counts)):
    it = iteration_counts[ind]
    print(f"{it = }")
    acc = accuracies[ind].item()
    print(acc)
    data[it] = acc

print(f"data: {data}")
import json
with open('data.json', 'w') as file:
    json.dump(data, file)

print("SHOWING GRAPH NOW!")
import matplotlib.pyplot as plt
plt.plot(iteration_counts, accuracies)
plt.show()
