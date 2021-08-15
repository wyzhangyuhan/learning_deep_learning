import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import data

class Datasturct:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.train_features = None
        self.test_features = None
        self.batch_size = 64
        self.learning_rate = 5
        self.num_epochs = 1000
        self.weight_decay = 0
        self.k = 2

    def read_dataset(self):
        self.train_data = pd.read_csv('train.csv')
        self.test_data = pd.read_csv('test.csv')
        self.all_features = pd.concat((self.train_data.iloc[:, 1:-1], self.test_data.iloc[:, 1:]))
       
    def data_preprocess(self):

        """
        数据归一化 (data normalization)
        """
        numeric_features = self.all_features.dtypes[self.all_features.dtypes != 'object'].index
        self.all_features[numeric_features] = self.all_features[numeric_features].apply(
            lambda x: (x - x.mean()) / (x.std()))
        # 在标准化数据之后，所有数据都意味着消失，因此我们可以将缺失值设置为0
        self.all_features[numeric_features] = self.all_features[numeric_features].fillna(0)

        """
        one-hot 编码
        """
        self.all_features = pd.get_dummies(self.all_features, dummy_na=True)

    def format_trans(self):
        n_train = self.train_data.shape[0]
        # print(list(all_features[:n_train]))
        self.train_features = torch.tensor(self.all_features[:n_train].values,
                                dtype=torch.float32)
        self.test_features = torch.tensor(self.all_features[n_train:].values,
                                dtype=torch.float32)
        self.train_labels = torch.tensor(self.train_data.SalePrice.values.reshape(-1, 1),
                            dtype=torch.float32)

    def net_init(self):
        self.loss = nn.MSELoss()
        self.in_features = self.train_features.shape[1]
        # self.net = nn.Sequential(nn.Linear(self.in_features, 1)) #先试试线性神经网络

    def getnet(self):
        net = nn.Sequential(nn.Linear(self.in_features, 256), nn.Dropout(0.05), nn.ReLU(), nn.Linear(256, 1))
        return net

    def load_array(self, data_arrays, is_train=True): 
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, self.batch_size, shuffle=is_train)

    def log_rmse(self, net, features, labels):
        # 为了在取对数时进一步稳定该值，将小于1的值设置为1
        clipped_preds = torch.clamp(net(features), 1, float('inf'))
        rmse = torch.sqrt(self.loss(torch.log(clipped_preds), torch.log(labels)))
        return rmse.item()

    def train(self, net, train_feature, train_label, test_feature, test_label):
        train_ls, test_ls = [], []
        train_iter = self.load_array((train_feature, train_label), self.batch_size)
        # 这里使用的是Adam优化算法
        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        for epoch in range(self.num_epochs):
            for X, y in train_iter:
                optimizer.zero_grad()
                l = self.loss(net(X), y) 
                l.backward()
                optimizer.step()
            train_ls.append(self.log_rmse(net, train_feature, train_label))
            if test_label is not None:
                test_ls.append(self.log_rmse(net, test_feature, test_label))
        return train_ls, test_ls

    def k_fold(self):
        X_train = self.train_features
        y_train = self.train_labels
        train_l_sum, valid_l_sum = 0, 0
        for i in range(self.k):
            data = self.get_k_fold_data(i, X_train, y_train)
            net = self.getnet()
            train_ls, valid_ls = self.train(net, *data)
            train_l_sum += train_ls[-1]
            valid_l_sum += valid_ls[-1]     
            print(f'fold {i + 1}, train log rmse {float(train_ls[-1]):f}, '
                f'valid log rmse {float(valid_ls[-1]):f}')
        return train_l_sum / self.k, valid_l_sum / self.k
    
    def get_k_fold_data(self, i, X, y): #构造不同的k折罢了
        assert self.k > 1
        fold_size = X.shape[0] // self.k
        X_train, y_train = None, None
        for j in range(self.k):
            idx = slice(j * fold_size, (j + 1) * fold_size)
            X_part, y_part = X[idx, :], y[idx]
            if j == i:
                X_valid, y_valid = X_part, y_part
            elif X_train is None:
                X_train, y_train = X_part, y_part
            else:
                X_train = torch.cat([X_train, X_part], 0)
                y_train = torch.cat([y_train, y_part], 0)
        return X_train, y_train, X_valid, y_valid

    def train_predict(self, train_features, test_feature, train_labels, test_data,):
        net = self.getnet()
        train_ls, _ = self.train(net, train_features, train_labels, None, None)
        # 将网络应用于测试集。
        preds = net(test_feature).detach().numpy()
        # 将其重新格式化以导出到Kaggle
        test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
        submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
        submission.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    d = Datasturct()
    d.read_dataset()
    d.data_preprocess()
    d.format_trans()
    d.net_init()
    d.k_fold()
    d.train_predict(d.train_features, d.test_features, d.train_labels, d.test_data)

