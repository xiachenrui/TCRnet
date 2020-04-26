import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


def cirterion_function(output, target):
    # target = torch.tensor(target, dtype=torch.long)
    cirterion = nn.CrossEntropyLoss()
    loss = cirterion(output, target.long())
    return loss




