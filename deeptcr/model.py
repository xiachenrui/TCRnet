# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torch.autograd import Variable
from layer import Net
from loss import cirterion_function
from datasets import test_loader_function, train_loader_function
import os


model = Net()
train_loader = train_loader_function()
test_loader = test_loader_function()

optimizer = optim.SGD(model.parameters(), lr=0.1)


# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


def train(epoch):
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    for batch_idx, (data, target) in enumerate(train_loader):  # batch_idx是enumerate（）函数自带的索引，从0开始
        data = data.view(64, 20 * 23)
        # print(data.size())
        # print(data)
        # print(target)
        # break
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        # output:64*10
        # print(output.size(), target.size())
        loss = cirterion_function(output, target)

        # 调用optimizer进行梯度下降更新参数
        ## 参数统计

        train_correct += (predicted == target.data).sum()
        running_loss += loss.item()
        train_total += target.size(0)
        # print("train_total= "+str(train_total))
        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ]'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset))
            )
        optimizer.zero_grad()  # 所有参数的梯度清零
        loss.backward()  # 即反向传播求梯度
        optimizer.step()
        # break
    # print("loss=" + str(loss.item()))
    # print("running_loss= " + str(running_loss))
    print("train_correct=" + str(int(train_correct)))
    print('train  epoch %d loss= %.3f  acc: %.3f ' % ( \
        epoch, running_loss, 100 * train_correct / train_total))


def test():
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        # target = torch.tensor(target, dtype=torch.long)
        # 这里的23应该改成从其他文件中读取tcr的最大长度，否则会报错
        data = data.view(64, 20 * 23)

        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target.long(), size_average=False)
        # get the index of t he max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        # print(pred)
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_acc = 100. * correct / len(test_loader.dataset)
    return test_acc


def run(epoch):
    for epoch in range(1, epoch):
        best_acc = 0
        train(epoch)
        test_acc = test()
        if test_acc > best_acc:
            torch.save(model, os.path.join(os.path.abspath(os.curdir), 'nn_model.pt'))

run(50)