# -*- coding: utf-8 -*-
from second_order_gradients import get_second_order_grad
from model import Model
import numpy as np
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

batch_size = 256
test_dataset = mnist.MNIST(root='./MNIST', download=True, train=False, transform=ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size)
model = Model()
path = "./models/mnist_0.86.pkl"
model.load_state_dict(torch.load(path))

loss_fuc = CrossEntropyLoss()
optimizer = SGD(model.parameters(),lr = 0.001,momentum = 0.9)

def grad():

    correct = 0
    total = 0
    # 弄个列表，来存放每次计算的1、2阶导数，最后区每次计算的对应位置的平均值
    final_1st_grad = []
    final_2nd_grad = []

    for data in test_loader:
        test_inputs, labels = data
        outputs_test = model(test_inputs)
        loss = loss_fuc(outputs_test, labels)
        # loss.backward()
        _, predicted = torch.max(outputs_test.data, 1)
        optimizer.zero_grad()
        xs = optimizer.param_groups[0]['params']
        ys = loss  # put your own loss into ys
        grads = torch.autograd.grad(ys, xs, create_graph=True)  # first order gradient
        print(grads)
        final_1st_grad.append(grads.numpy())
        grads2 = get_second_order_grad(grads, xs)  # second order gradient
        print(grads2)
        final_2nd_grad.append(grads2.numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('识别准确率为：{}%'.format(100 * correct.item() / total))

    # 计算一阶和二阶平均数
    print("1st_grad", np.mean(final_1st_grad, axis=0))
    print("2nd_grad", np.mean(final_2nd_grad, axis=0))
    # 输出并保存权值和一阶导数
    # print("conv1.weight:", model.conv1.weight)
    # print("conv1.weight.grad:", model.conv1.weight.grad)
    # np.save('./logs/grad.csv', model.conv1.weight.grad.numpy())
    # np.savetxt('./logs/grad1.txt', model.conv1.weight.grad.numpy().reshape(6, -1))


if __name__ == '__main__':
    grad()