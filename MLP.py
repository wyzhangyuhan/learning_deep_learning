"""
多层线性感知机
"""

import torch
from torch import nn
from softmax import load_data_fashion_mnist, torch_train, train_epoch_torch, predict_ch3

def torch_train(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_metrics = train_epoch_torch(net, train_iter, loss, updater)
        # test_acc = evaluate_accuracy(net, test_iter)
        print(f'训练误差为：{train_metrics[0]},准确率为：{train_metrics[1]}')
    train_loss, train_acc = train_metrics
    

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)


def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)  # 这里“@”代表矩阵乘法
    return (H @ W2 + b2)

def MLP_raw():
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    num_inputs, num_outputs, num_hiddens = 784, 10, 256 #
    W1 = nn.Parameter(
        torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = nn.Parameter(
        torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
    params = [W1, b1, W2, b2]

    num_epochs, lr = 10, 0.1
    updater = torch.optim.SGD(params, lr=lr)

def MLP():
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                    nn.Linear(256, 10))
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights)
    batch_size, lr, num_epochs = 256, 0.1, 20
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    torch_train(net, train_iter, test_iter, loss, num_epochs, trainer)

    predict_ch3(net, test_iter)

if __name__ == '__main__':
    MLP()
