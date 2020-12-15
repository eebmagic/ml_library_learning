'''
Code from here:
https://github.com/llSourcell/pytorch_in_5_minutes/blob/master/demo.py
'''

import torch
from torch.autograd import Variable

dtype = torch.FloatTensor

N = 64
D_in = 1000
H = 100
D_out = 10

x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

learning_rate = 1e-6
training_iterations = 5000
losses = []
runs = []
for t in range(training_iterations):
    # Make a feed forward prediction
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # Calculate loss
    # loss = sum of ((diff at each point in array) ^ 2)
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.data.item())
    losses.append(loss.data.item())
    runs.append(t)

    # updates gradiants for all Variables with requires_grad set to True
    loss.backward()

    # if t > 0:
    #     quit()

    # Update weights regarding the gradients calculated with loss
    # adjust with consideration for learning rate
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    # reset gradiant data from last backprop for next iteration
    w1.grad.data.zero_()
    w2.grad.data.zero_()

print(x)
print(x.size())
print(y_pred)
print(y_pred.size())
# print(w1)
# print(w1.size())
# print(w2)
# print(w2.size())

import matplotlib.pyplot as plt
plt.plot(runs, losses)
# plt.yscale("log")
plt.show()