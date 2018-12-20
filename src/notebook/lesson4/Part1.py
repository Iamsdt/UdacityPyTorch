import torch
import numpy as np


def activation(x):
    return 1 / (1 + torch.exp(-x))


torch.manual_seed(7)
features = torch.randn((1, 5))
weights = torch.randn_like(features)
bias = torch.randn((1, 1))

y = activation(torch.sum(features * weights) + bias)
print(y)

y = activation((weights * features).sum() + bias)
print(y)

# torch.mm(features, weights)
"""Metrics mismatch"""

y = activation(torch.mm(features, weights.view(5, 1)) + bias)
print(y)

print("\n")
torch.manual_seed(7)
features = torch.randn(1, 3)
n_input = features.shape[1]  # return new tensor
n_hidden = 2
n_output = 1

W1 = torch.randn(n_input, n_hidden)  # input to hidden layer
W2 = torch.randn(n_hidden, n_output)  # hidden to output layer

B1 = torch.randn(1, n_hidden)
B2 = torch.randn(1, n_output)

h = activation(torch.mm(features, W1) + bias)
print(h)
output = activation(torch.mm(h, W2) + bias)
print(output)

# Numpy to torch
a = np.random.rand(4, 3)
print(a)

b = torch.from_numpy(a)
print(b)
b.numpy()
b.mul_(2)
print(a)