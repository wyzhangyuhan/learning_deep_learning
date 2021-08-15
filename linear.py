
"""
线性回归实现(pytorch)
"""
import random
from torch.utils import data
from torch import nn
import torch

# 生成样本
def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声。"""
    X = torch.normal(0, 1, (num_examples, len(w))) #随机生成正态样本
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape) #加点燥
    return X, y.reshape((-1, 1))

# 随机小批量
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices] #返回一个迭代

def linreg(X, w, b):  #@save
    """线性回归模型。"""
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):  #@save
    """均方损失。"""
    return (y_hat - y.reshape(y_hat.shape))**2 / 2

def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降。"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def raw_linear():
    #真实值
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    #设置超参数
    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squared_loss

    #初始化模型参数
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)  
    batch_size = 10
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)  # `X`和`y`的小批量损失
            # 因为`l`形状是(`batch_size`, 1)，而不是一个标量。`l`中的所有元素被加到一起，
            # 并以此计算关于[`w`, `b`]的梯度
            l.sum().backward()
            sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

    print('w的值：', w.reshape(true_w.shape))
    print('w的估计误差：', true_w - w.reshape(true_w.shape))
    print('b的值：', b)
    print('b的估计误差：', true_b - b)

def torch_linear():
    #真实值
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    batch_size = 10
    data_iter = load_array((features, labels), batch_size)

    #网络设置
    net = nn.Sequential(nn.Linear(2, 1)) #第一个指定输入特征形状，第二个指定输出特征形状
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)
    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)

    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')

    w = net[0].weight.data
    print('w的值：', w.reshape(true_w.shape))
    print('w的估计误差：', true_w - w.reshape(true_w.shape))
    b = net[0].bias.data
    print('b的值：', b)
    print('b的估计误差：', true_b - b)

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器。"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

if __name__ == '__main__': 
    
    #不使用torch的网络
    raw_linear()

    #使用torch框架
    torch_linear()

    
