get_ipython().run_line_magic("matplotlib", " inline")
import math
import time
import numpy as np
import torch
from d2l import torch as d2l
import random


def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)
x = np.arange(-7, 7, 0.01)

# Mean and standard deviation pairs
params = [(0, 1), (0, 2), (3, 1)]

d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])


def synthetic_data(w, b, num_examples):  #@save
    """Generate y = Xw + b + noise."""
    X = torch.normal(mean=0, std=1, size=(num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y = y + torch.normal(0, 0.01, size=y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)


d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)


#FUNCTION TO SHUFFLE THE DATA AND ACCESS IT IN MINIBATCHES
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))    
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
        


num_examples = len(features)
indices = list(range(num_examples)); indices[:3]


random.shuffle(indices); indices[:3]


mylist = [x*x for x in range(3)]
for i in mylist:
    print(i)


mygenerator = (x*x for x in range(3))
for i in mygenerator:
    print(i)


for i in mygenerator:
    print(i)
#see? we can't iterate over it again


def create_generator():
...    mylist = range(3)
...    for i in mylist:
...        yield i*i# basically returns this only once


mygenerator = create_generator() # create a generator
print(mygenerator) # mygenerator is an object!


for i in mygenerator:
...     print(i)



for i in mygenerator:
...     print(i)



def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break


w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
w,b


def linreg(X, w, b):  #@save
    """The linear regression model."""
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):  #@save
    """Squared loss."""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent. params for linreg is w, b"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


lr = 0.03
num_epochs = 3
net = linreg
######## with pytorch
# from torch import nn
# num_input_features = 2
# num_output_features=1
# net = nn.Sequential(nn.Linear(num_input_features, num_output_features))
##############
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # Compute gradient on `l` with respect to [`w`, `b`]
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # Update parameters using their gradient
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')


print(f'error in estimating w: {true_w - w.reshape(true_w.shape)}')
print(f'error in estimating b: {true_b - b}')



