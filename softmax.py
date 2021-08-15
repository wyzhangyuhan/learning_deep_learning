import matplotlib

import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from torch import nn

"""
实现个简单的softmax分类器
"""

class Accumulator:  #@save
    """在`n`个变量上累加。"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签。"""
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
        'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def get_dataloader_workers():  #@save
    """使用4个进程来读取数据。"""
    return 4

def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中。"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="../data",
                                                    train=True,
                                                    transform=trans,
                                                    download=False)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data",
                                                   train=False,
                                                   transform=trans,
                                                   download=False)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))

def accuracy(y_hat, y):  #@save
    """计算预测正确的数量。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = (y_hat.type(y.dtype) == y)
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度。"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

def train_epoch_torch(net, train_iter, loss, updater):
    net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        # 使用PyTorch内置的优化器和损失函数
        updater.zero_grad()
        l.backward()
        updater.step()
        metric.add(
            float(l) * len(y), accuracy(y_hat, y),
            y.size().numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def torch_train(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_metrics = train_epoch_torch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, print(train_loss)
    assert train_acc <= 1 and train_acc > 0.7, print(train_acc)
    assert test_acc <= 1 and test_acc > 0.7, print(test_acc)

def softmax():
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    net.apply(init_weights)
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)
    num_epochs = 10
    torch_train(net, train_iter, test_iter, loss, num_epochs, trainer)

    predict_ch3(net, test_iter) #检验

def predict_ch3(net, test_iter, n=6):  #@save
    """预测"""
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    cnt = 0
    for i in range(len(trues)):
        if trues[i] == preds[i]:
            cnt+=1
        print(f'[groundtruth:{trues[i]}, predict:{preds[i]}]\n')
    print(f'accuracy:{cnt/len(trues)}')


if __name__ == '__main__':
    softmax()

    
