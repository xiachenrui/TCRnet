# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from layer import Net
from loss import cirterion_function
from datasets import test_loader_function, train_loader_function
import os


# final_model = torch.load(os.path.join(r'E:\OneDrive\program\deeptcr','model.pt'))
final_model = torch.load(os.path.join(os.path.abspath(os.curdir), 'model.pt'))

train_loader = train_loader_function()
test_loader = test_loader_function()


def final_test():
    final_model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        data = data.view(64, 20 * 23)

        output = final_model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False)
        # get the index of t he max log-probability
        pred = output.data.max(1, keepdim=True)[1]

        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print("This is the final test of yout dataset")
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_acc = 100. * correct / len(test_loader.dataset)
    return test_acc
