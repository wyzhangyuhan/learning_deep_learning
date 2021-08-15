import math

import torch
from torch import nn
from torch._C import DeviceObjType
from torch.nn import functional as F
from text_process import load_data_time_machine
from softmax import Accumulator, accuracy
from Lenet import try_gpu
from raw_RNN import RNNModelScratch, train_ch8
from RNN import RNNModel


#####################################
#####################################
"""GRU从零实现"""
#####################################
#####################################

def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size
    
    def normal(shape): #初始化模型参数为0.01的高斯分布
        return torch.randn(size=shape, device=device)*0.01
    
    def three():
        return (normal((num_inputs,num_hiddens)),
                normal((num_hiddens,num_hiddens)),
                torch.zeros(num_hiddens,device = device))
    
    W_xz, W_hz, b_z = three()  # 更新门参数
    W_xr, W_hr, b_r = three()  # 重置门参数
    W_xh, W_hh, b_h = three()  # 候选隐藏状态参数

    W_hq = normal((num_hiddens,num_outputs))
    b_q =  torch.zeros(num_outputs,device = device)

    #为参数附加梯度，以便后续更新
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    
    #返回网络中的各个参数
    return params

def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []

    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


if __name__ == '__main__':
    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)

    vocab_size, num_hiddens, device = len(vocab), 256, try_gpu()
    num_epochs, lr = 500, 1

    #从零实现调用
    # model = RNNModelScratch(len(vocab), num_hiddens, device, 
    #                         get_params, init_gru_state, gru)
    # train_ch8(model, train_iter, vocab, lr, num_epochs, device)

    #简洁实现调用
    num_inputs = vocab_size
    gru_layer = nn.GRU(num_inputs, num_hiddens)
    model = RNNModel(gru_layer, len(vocab))
    model = model.to(device)
    train_ch8(model, train_iter, vocab, lr, num_epochs, device)



