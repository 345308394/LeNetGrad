from model import Model
import numpy as np
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

def test():
    batch_size = 256
    test_dataset = mnist.MNIST(root='./MNIST', download = True, train=False, transform=ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model = Model()
    path = "./models/mnist_0.86.pkl"
    model.load_state_dict(torch.load(path))
    sgd = SGD(model.parameters(), lr=1e-1)
    cost = CrossEntropyLoss()

    correct = 0
    _sum = 0
    model.eval()
    for idx, (test_x, test_label) in enumerate(test_loader):
        predict_y = model(test_x.float()).detach()
        predict_ys = np.argmax(predict_y, axis=-1)
        label_np = test_label.numpy()
        _ = predict_ys == test_label
        correct += np.sum(_.numpy(), axis=-1)
        _sum += _.shape[0]

    print('accuracy: {:.2f}'.format(correct / _sum))


if __name__ == '__main__':
    test()