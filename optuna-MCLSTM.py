
"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.
In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.
"""

import os
from torch.utils.data import TensorDataset
import numpy as np
from functions import ReverseLayerF
import optuna
import pandas as pd
import sys
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import math


DEVICE = torch.device("cpu")
DIR = os.getcwd()
input_size_number = 17  
num_layers = 1  
win_len = 30  
sli_step = 1 
model_root = 'models'  
n_epochs = 10
loss_func_rul = torch.nn.MSELoss().to(DEVICE)
loss_func_domain = torch.nn.NLLLoss().to(DEVICE)



class MCLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_sz = input_size
        self.hidden_sz = hidden_size
        self.W = nn.Parameter(torch.Tensor(input_size, hidden_size * 4+5))
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4+5))
        self.kernel1 = nn.Parameter(torch.Tensor(torch.ones(1, hidden_size * 5)))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4+5))
        self.bias1 = nn.Parameter(torch.Tensor(hidden_size * 2))# s1,s2
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_sz)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x,
                init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        c_t_1_seq = []
        c_t_2_seq = []
        c_t_3_seq = []
        c_t_4_seq = []
        c_t_5_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_sz).to(x.device),
                        torch.zeros(bs, self.hidden_sz).to(x.device))
        else:
            h_t, c_t = init_states
        HS = self.hidden_sz
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]),  # input
                torch.sigmoid(gates[:, HS:HS * 2]),  # forget
                torch.tanh(gates[:, HS * 2:HS * 3]),
                torch.sigmoid(gates[:, HS * 3:HS * 4]),
                )# output
            d_t = torch.tanh(gates[:, HS * 4:])
            d_t = torch.softmax(d_t,dim=1)
            s1=self.bias1[:HS]
            s2=self.bias1[HS: HS * 2]
            c_t_1=torch.tanh(g_t)
            c_t_5=c_t
            c_t_3 = f_t * c_t + i_t * g_t
            c_t_2 = s2 * c_t_3 + (1 - s2) * c_t_1
            c_t_4 = s1 * c_t_3 + (1 - s1) * c_t_5
            d1 = torch.index_select(d_t, 1, torch.LongTensor([0]))
            d2 = torch.index_select(d_t, 1, torch.LongTensor([1]))
            d3 = torch.index_select(d_t, 1, torch.LongTensor([2]))
            d4 = torch.index_select(d_t, 1, torch.LongTensor([3]))
            d5 = torch.index_select(d_t, 1, torch.LongTensor([4]))
            kernel_1 = self.kernel1[:, :HS]
            kernel_2 = self.kernel1[:, HS: HS * 2]
            kernel_3 = self.kernel1[:, HS * 2: HS * 3]
            kernel_4 = self.kernel1[:, HS * 3:HS * 4]
            kernel_5 = self.kernel1[:, HS * 4:]
            c_t = torch.mm(d1, kernel_1) * c_t_1 + torch.mm(d2, kernel_2) * c_t_2 + torch.mm(d3,kernel_3) * c_t_3 + torch.mm(d4, kernel_4) * c_t_4 + torch.mm(d5, kernel_5) * c_t_5
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
            c_t_1_seq.append(c_t_1.unsqueeze(0))
            c_t_2_seq.append(c_t_2.unsqueeze(0))
            c_t_3_seq.append(c_t_3.unsqueeze(0))
            c_t_4_seq.append(c_t_4.unsqueeze(0))
            c_t_5_seq.append(c_t_5.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        c_t_1_seq = torch.cat(c_t_1_seq, dim=0)
        c_t_1_seq = c_t_1_seq.transpose(0, 1).contiguous()
        c_t_2_seq = torch.cat(c_t_2_seq, dim=0)
        c_t_2_seq = c_t_2_seq.transpose(0, 1).contiguous()
        c_t_3_seq = torch.cat(c_t_3_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        c_t_3_seq = c_t_3_seq.transpose(0, 1).contiguous()
        c_t_4_seq = torch.cat(c_t_4_seq, dim=0)
        c_t_4_seq = c_t_4_seq.transpose(0, 1).contiguous()
        c_t_5_seq = torch.cat(c_t_5_seq, dim=0)
        c_t_5_seq = c_t_5_seq.transpose(0, 1).contiguous()
        return hidden_seq, c_t_1_seq,c_t_2_seq,c_t_3_seq,c_t_4_seq,c_t_5_seq, (h_t, c_t),d_t

class DANNModel(nn.Module):

    def __init__(self, input_num, hidden_num, layer_num=1, dropout=0):
        super(DANNModel, self).__init__()  
        self.hidden_size = hidden_num
        self.layer_num = layer_num
        self.dropout = dropout
        self.hidden = None  
        self.generator = MCLSTM(input_size=input_num, hidden_size=hidden_num)
        self.RUL_regression = nn.Sequential(
            nn.Linear(hidden_num, hidden_num // 2),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_num // 2, hidden_num // 4),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_num // 4),
            nn.Linear(hidden_num // 4, 1)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_num, hidden_num // 2),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_num // 2),
            nn.Linear(hidden_num // 2, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input_data, alpha):
        feature, c1_feature, c2_feature, c3_feature, c4_feature, c5_feature, self.hidden, d_reslut = self.generator(
            input_data.float())
        feature = feature[:, -1, :]
        c1_feature = c1_feature[:, -1, :]
        c2_feature = c2_feature[:, -1, :]
        c3_feature = c3_feature[:, -1, :]
        c4_feature = c4_feature[:, -1, :]
        c5_feature = c5_feature[:, -1, :]
        reverse_c1_feature = ReverseLayerF.apply(c1_feature, alpha)
        reverse_c2_feature = ReverseLayerF.apply(c2_feature, alpha)
        reverse_c3_feature = ReverseLayerF.apply(c3_feature, alpha)
        reverse_c4_feature = ReverseLayerF.apply(c4_feature, alpha)
        reverse_c5_feature = ReverseLayerF.apply(c5_feature, alpha)
        rul = self.RUL_regression(feature)
        domain1 = self.domain_classifier(reverse_c1_feature)
        domain2 = self.domain_classifier(reverse_c2_feature)
        domain3 = self.domain_classifier(reverse_c3_feature)
        domain4 = self.domain_classifier(reverse_c4_feature)
        domain5 = self.domain_classifier(reverse_c5_feature)
        return rul, domain1, domain2, domain3, domain4, domain5, d_reslut



def define_model(trial):
    out_features = trial.suggest_int("n_units_l{}".format(1), 4, 128)
    p = trial.suggest_float("dropout_l{}".format(1), 0.2, 0.5)
    model = DANNModel(input_num=input_size_number, hidden_num=out_features, layer_num=num_layers,
                      dropout=p)
    return model

def get_data(batch_size):
    """
            ===================dada===================
            ===================transfer_task_12 ### FD001----FD002===================
            """

    # #
    # trainDataFD001 = np.load("trainDataFD001.npy")
    # trainTargetFD001 = np.load("trainTargetFD001.npy")
    # source_data_x = trainDataFD001    # print(source_data_x.shape())
    # source_data_y = trainTargetFD001
    # 
    # trainDataFD002 = np.load("trainDataFD002.npy")
    # trainTargetFD002 = np.load("trainTargetFD002.npy")
    # target_data_x = trainDataFD002
    # target_data_y = trainTargetFD002
    # #Amplified sample
    # source_data_x = np.concatenate((source_data_x, source_data_x, source_data_x), axis=0)
    # source_data_x = source_data_x[:-6974, :, :]
    # source_data_y = np.concatenate((source_data_y, source_data_y, source_data_y), axis=0)
    # source_data_y = source_data_y[:-6974]
    # # 
    # testData = np.load("testDataFD002.npy")
    # testData = torch.tensor(testData)
    # testTarget = np.load("testTargetFD002.npy")
    # testTarget = torch.tensor(testTarget)
    """
          ===================dada===================
          ===================transfer_task_13 ### FD001----FD003===================
          """
  
    # trainDataFD001 = np.load("trainDataFD001.npy")
    # trainTargetFD001 = np.load("trainTargetFD001.npy")
    # source_data_x = trainDataFD001    # print(source_data_x.shape())
    # source_data_y = trainTargetFD001
    # # 
    # trainDataFD003 = np.load("trainDataFD003.npy")
    # trainTargetFD003 = np.load("trainTargetFD003.npy")
    # target_data_x = trainDataFD003
    # target_data_y = trainTargetFD003
    # # 
    # source_data_x = np.concatenate((source_data_x, source_data_x), axis=0)
    # source_data_x = source_data_x[:-13642, :, :]
    # source_data_y = np.concatenate((source_data_y, source_data_y), axis=0)
    # source_data_y = source_data_y[:-13642]
    # # 
    # testData = np.load("testDataFD003.npy")
    # testData = torch.tensor(testData)
    # testTarget = np.load("testTargetFD003.npy")
    # testTarget = torch.tensor(testTarget)

    """
    ===================dada===================
    ===================transfer_task_14 ### FD001----FD004===================
    """
    #
    # # 
    # trainDataFD001 = np.load("trainDataFD001.npy")
    # trainTargetFD001 = np.load("trainTargetFD001.npy")
    # source_data_x = trainDataFD001    # print(source_data_x.shape())
    # source_data_y = trainTargetFD001
    # #
    # trainDataFD004 = np.load("trainDataFD004.npy")
    # trainTargetFD004 = np.load("trainTargetFD004.npy")
    # target_data_x = trainDataFD004
    # target_data_y = trainTargetFD004
    # # 
    # source_data_x = np.concatenate((source_data_x, source_data_x, source_data_x, source_data_x), axis=0)
    # source_data_x = source_data_x[:-16896, :, :]
    # source_data_y = np.concatenate((source_data_y, source_data_y, source_data_y, source_data_y), axis=0)
    # source_data_y = source_data_y[:-16896]
    # # 
    # testData = np.load("testDataFD004.npy")
    # testData = torch.tensor(testData)
    # testTarget = np.load("testTargetFD004.npy")
    # testTarget = torch.tensor(testTarget)
    """
          ==================dada==================
          ===================transfer_task_21 ### FD002----FD001===================
          """

    # #
    # trainDataFD002 = np.load("trainDataFD002.npy")
    # trainTargetFD002 = np.load("trainTargetFD002.npy")
    # source_data_x = trainDataFD002    # print(source_data_x.shape())
    # source_data_y = trainTargetFD002
    # #
    # trainDataFD001 = np.load("trainDataFD001.npy")
    # trainTargetFD001 = np.load("trainTargetFD001.npy")
    # target_data_x = trainDataFD001
    # target_data_y = trainTargetFD001
    # # 
    # target_data_x = np.concatenate((target_data_x, target_data_x, target_data_x), axis=0)
    # target_data_x = target_data_x[:-6974, :, :]
    # target_data_y = np.concatenate((target_data_y, target_data_y, target_data_y), axis=0)
    # target_data_y = target_data_y[:-6974]
    # # 
    # testData = np.load("testDataFD001.npy")
    # testData = torch.tensor(testData)
    # testTarget = np.load("testTargetFD001.npy")
    # testTarget = torch.tensor(testTarget)
    """
             ===================dada===================
             ===================transfer_task_23 ### FD002----FD003===================
             """

    #
    # trainDataFD002 = np.load("trainDataFD002.npy")
    # trainTargetFD002 = np.load("trainTargetFD002.npy")
    # source_data_x = trainDataFD002  # print(source_data_x.shape())
    # source_data_y = trainTargetFD002
    # # 
    # trainDataFD003 = np.load("trainDataFD003.npy")
    # trainTargetFD003 = np.load("trainTargetFD003.npy")
    # target_data_x = trainDataFD003
    # target_data_y = trainTargetFD003
    # # 
    # target_data_x = np.concatenate((target_data_x, target_data_x, target_data_x), axis=0)
    # target_data_x = target_data_x[:-19241, :, :]
    # target_data_y = np.concatenate((target_data_y, target_data_y, target_data_y), axis=0)
    # target_data_y = target_data_y[:-19241]
    # # 
    # testData = np.load("testDataFD003.npy")
    # testData = torch.tensor(testData)
    # testTarget = np.load("testTargetFD003.npy")
    # testTarget = torch.tensor(testTarget)
    """
               ===================dada===================
               ===================transfer_task_24 ### FD002----FD004===================
               """

    # # 
    # trainDataFD002 = np.load("trainDataFD002.npy")
    # trainTargetFD002 = np.load("trainTargetFD002.npy")
    # source_data_x = trainDataFD002  # print(source_data_x.shape())
    # source_data_y = trainTargetFD002
    # # 
    # trainDataFD004 = np.load("trainDataFD004.npy")
    # trainTargetFD004 = np.load("trainTargetFD004.npy")
    # target_data_x = trainDataFD004
    # target_data_y = trainTargetFD004
    # # 
    # source_data_x = np.concatenate((source_data_x, source_data_x), axis=0)
    # source_data_x = source_data_x[:-38410, :, :]
    # source_data_y = np.concatenate((source_data_y, source_data_y), axis=0)
    # source_data_y = source_data_y[:-38410]
    # #
    # testData = np.load("testDataFD004.npy")
    # testData = torch.tensor(testData)
    # testTarget = np.load("testTargetFD004.npy")
    # testTarget = torch.tensor(testTarget)
    """
                  ===================dada===================
                  ===================transfer_task_31 ### FD003----FD001===================
                  """

    # # 
    # trainDataFD003 = np.load("trainDataFD003.npy")
    # trainTargetFD003 = np.load("trainTargetFD003.npy")
    # source_data_x = trainDataFD003  # print(source_data_x.shape())
    # source_data_y = trainTargetFD003
    # #
    # trainDataFD001 = np.load("trainDataFD001.npy")
    # trainTargetFD001 = np.load("trainTargetFD001.npy")
    # target_data_x = trainDataFD001
    # target_data_y = trainTargetFD001
    # #
    # target_data_x = np.concatenate((target_data_x, target_data_x), axis=0)
    # target_data_x = target_data_x[:-13642, :, :]
    # target_data_y = np.concatenate((target_data_y, target_data_y), axis=0)
    # target_data_y = target_data_y[:-13642]
    # # 
    # testData = np.load("testDataFD001.npy")
    # testData = torch.tensor(testData)
    # testTarget = np.load("testTargetFD001.npy")
    # testTarget = torch.tensor(testTarget)

    """
                    ===================dada===================
                    ===================transfer_task_32 ### FD003----FD002===================
                    """

    # 
    # trainDataFD003 = np.load("trainDataFD003.npy")
    # trainTargetFD003 = np.load("trainTargetFD003.npy")
    # source_data_x = trainDataFD003  # print(source_data_x.shape())
    # source_data_y = trainTargetFD003
    # # 
    # trainDataFD002 = np.load("trainDataFD002.npy")
    # trainTargetFD002 = np.load("trainTargetFD002.npy")
    # target_data_x = trainDataFD002
    # target_data_y = trainTargetFD002
    # # 
    # source_data_x = np.concatenate((source_data_x, source_data_x, source_data_x), axis=0)
    # source_data_x = source_data_x[:-19241, :, :]
    # source_data_y = np.concatenate((source_data_y, source_data_y, source_data_y), axis=0)
    # source_data_y = source_data_y[:-19241]
    # # 
    # testData = np.load("testDataFD002.npy")
    # testData = torch.tensor(testData)
    # testTarget = np.load("testTargetFD002.npy")
    # testTarget = torch.tensor(testTarget)
    """
                      ===================dada===================
                      ===================transfer_task_34 ### FD003----FD004===================
                      """
    #
    # # 
    # trainDataFD003 = np.load("trainDataFD003.npy")
    # trainTargetFD003 = np.load("trainTargetFD003.npy")
    # source_data_x = trainDataFD003  # print(source_data_x.shape())
    # source_data_y = trainTargetFD003
    # # 
    # trainDataFD004 = np.load("trainDataFD004.npy")
    # trainTargetFD004 = np.load("trainTargetFD004.npy")
    # target_data_x = trainDataFD004
    # target_data_y = trainTargetFD004
    # # 
    # source_data_x = np.concatenate((source_data_x, source_data_x, source_data_x), axis=0)
    # source_data_x = source_data_x[:-11432, :, :]
    # source_data_y = np.concatenate((source_data_y, source_data_y, source_data_y), axis=0)
    # source_data_y = source_data_y[:-11432]
    # # 
    # testData = np.load("testDataFD004.npy")
    # testData = torch.tensor(testData)
    # testTarget = np.load("testTargetFD004.npy")
    # testTarget = torch.tensor(testTarget)

    """
    ===================dada===================
    ===================transfer_task_41 ### FD004----FD001===================
    """
    #
    # # 
    # trainDataFD004 = np.load("trainDataFD004.npy")
    # trainTargetFD004 = np.load("trainTargetFD004.npy")
    # source_data_x = trainDataFD004    # print(source_data_x.shape())
    # source_data_y = trainTargetFD004
    # # 
    # trainDataFD001 = np.load("trainDataFD001.npy")
    # trainTargetFD001 = np.load("trainTargetFD001.npy")
    # target_data_x = trainDataFD001
    # target_data_y = trainTargetFD001
    # # 
    # target_data_x = np.concatenate((target_data_x, target_data_x, target_data_x, target_data_x), axis=0)
    # target_data_x = target_data_x[:-16896, :, :]
    # target_data_y = np.concatenate((target_data_y, target_data_y, target_data_y, target_data_y), axis=0)
    # target_data_y = target_data_y[:-16896]
    # # 
    # testData = np.load("testDataFD001.npy")
    # testData = torch.tensor(testData)
    # testTarget = np.load("testTargetFD001.npy")
    # testTarget = torch.tensor(testTarget)
    """
       ===================dada===================
       ===================transfer_task_42 ### FD004----FD002===================
       """

    # # 
    # trainDataFD004 = np.load("trainDataFD004.npy")
    # trainTargetFD004 = np.load("trainTargetFD004.npy")
    # source_data_x = trainDataFD004    # print(source_data_x.shape())
    # source_data_y = trainTargetFD004
    # # 
    # trainDataFD002 = np.load("trainDataFD002.npy")
    # trainTargetFD002 = np.load("trainTargetFD002.npy")
    # target_data_x = trainDataFD002
    # target_data_y = trainTargetFD002
    # # 
    # target_data_x = np.concatenate((target_data_x, target_data_x), axis=0)
    # target_data_x = target_data_x[:-38410, :, :]
    # target_data_y = np.concatenate((target_data_y, target_data_y), axis=0)
    # target_data_y = target_data_y[:-38410]
    # # 
    # testData = np.load("testDataFD002.npy")
    # testData = torch.tensor(testData)
    # testTarget = np.load("testTargetFD002.npy")
    # testTarget = torch.tensor(testTarget)

    """
         ===================dada===================
         ===================transfer_task_43 ### FD004----FD003===================
         """

    # 
    trainDataFD004 = np.load("trainDataFD004.npy")
    trainTargetFD004 = np.load("trainTargetFD004.npy")
    source_data_x = trainDataFD004  # print(source_data_x.shape())
    source_data_y = trainTargetFD004
    # 
    trainDataFD003 = np.load("trainDataFD003.npy")
    trainTargetFD003 = np.load("trainTargetFD003.npy")
    target_data_x = trainDataFD003
    target_data_y = trainTargetFD003
    # 
    target_data_x = np.concatenate((target_data_x, target_data_x, target_data_x), axis=0)
    target_data_x = target_data_x[:-11432, :, :]
    target_data_y = np.concatenate((target_data_y, target_data_y, target_data_y), axis=0)
    target_data_y = target_data_y[:-11432]
    # 
    testData = np.load("testDataFD003.npy")
    testData = torch.tensor(testData)
    testTarget = np.load("testTargetFD003.npy")
    testTarget = torch.tensor(testTarget)

    # 
    src_dataset = TensorDataset(torch.tensor(source_data_x), torch.tensor(source_data_y))
    tar_dataset = TensorDataset(torch.tensor(target_data_x), torch.tensor(target_data_y))
    # 
    source_data = torch.utils.data.DataLoader(
        dataset=src_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0)
    target_data = torch.utils.data.DataLoader(
        dataset=tar_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0)

    return source_data, target_data, testData, testTarget

def model_training(n_epochs, model, loss_func_rul=None, loss_func_domain=None, opt=None, source_ds=None,
                    target_ds=None,testData=None,testTarget=None, alpha=1, transfer=True):
    """

    :param n_epochs: 
    :param model: 
    :param loss_func_rul: 
    :param loss_func_domain: domain classification
    :param opt: 
    :param source_ds: ，TensorDataset
    :param target_ds: TensorDataset
    :param alpha: 
    :param transfer: 
    :return: 
    """
    length = min(len(source_ds), len(target_ds))  
    loss_s = list(np.zeros(n_epochs))
    loss_t = list(np.zeros(n_epochs))

    for epoch in range(n_epochs):
        model.train()
        src_iter = iter(source_ds)
        tar_iter = iter(target_ds)
        R = 2 / (1 + math.exp(-1 * epoch / n_epoch)) - 1
        for idx in range(min(len(src_iter), len(tar_iter))):
            src_data, src_label = src_iter.next()
            tar_data, tar_label = tar_iter.next()
            err_s_rul, err_s_domain, _ = source_loss_batch(model, loss_func_rul, loss_func_domain,
                                                           src_data, src_label, alpha)
            err_t_domain, _ = target_loss_batch(model, loss_func_domain, tar_data, tar_label, alpha)
            opt.zero_grad()  
            loss = err_s_rul + R * (err_t_domain + err_s_domain)
            loss.backward()
            opt.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            rul_output, _ = model.forward(testData, alpha=1)
            yPreds = rul_output * 125
            test_mse = loss_func_rul(yPreds.float(), testTarget.float())
            test_rmse = test_mse.sqrt()
            test_score = myScore(testTarget.float(), yPreds.float())
            print('test_rmse', test_rmse, 'test_score', test_score)
        accuracy = test_rmse + test_score / 1000
    return loss_s, loss_t

    return test_rmse, test_score
def source_loss_batch(model, loss_func_rul=None, loss_func_domain=None, xb=None, yb=None, alpha_num=1, transfer=True):
    """
    :param model: 
    :param loss_func_rul: 
    :param loss_func_domain: 
    :param xb: 
    :param yb: 
    :param alpha_num: 
    :return: 
    """
    if transfer:
        domain_label = torch.zeros(len(yb)).long() 
       
        rul_output, domain1_output, domain2_output, domain3_output, domain4_output, domain5_output, d_reslut = model.forward(
            xb, alpha=alpha_num)
        err_s_rul = loss_func_rul(rul_output.float(), yb.float())
        d1 = torch.index_select(d_reslut, 1, torch.LongTensor([0]))
        d2 = torch.index_select(d_reslut, 1, torch.LongTensor([1]))
        d3 = torch.index_select(d_reslut, 1, torch.LongTensor([2]))
        d4 = torch.index_select(d_reslut, 1, torch.LongTensor([3]))
        d5 = torch.index_select(d_reslut, 1, torch.LongTensor([4]))
        domain1_output = domain1_output * d1
        domain2_output = domain2_output * d2
        domain3_output = domain3_output * d3
        domain4_output = domain1_output * d4
        domain5_output = domain1_output * d5
        err_s_domain1 = loss_func_domain(domain1_output, domain_label)
        err_s_domain2 = loss_func_domain(domain2_output, domain_label)
        err_s_domain3 = loss_func_domain(domain3_output, domain_label)
        err_s_domain4 = loss_func_domain(domain4_output, domain_label)
        err_s_domain5 = loss_func_domain(domain5_output, domain_label)
        err_s_domain = err_s_domain1 + err_s_domain2 + err_s_domain3 + err_s_domain4 + err_s_domain5
        return err_s_rul, err_s_domain, len(xb)
    else:
        rul_output = model.forward(xb, alpha=alpha_num)
        err_s_rul = loss_func_rul(rul_output, yb)
        return err_s_rul


def target_loss_batch(model, loss_func_domain, xb, yb, alpha_num):
    """
    :param model:
    :param loss_func_domain:
    :param xb:
    :param yb:
    :param alpha_num:
    :return: 
    """
    domain_label = torch.ones(len(yb)).long()  
    rul_output, domain1_output, domain2_output, domain3_output, domain4_output, domain5_output, d_reslut = model.forward(
        xb,
        alpha=alpha_num)
    d1 = torch.index_select(d_reslut, 1, torch.LongTensor([0]))
    d2 = torch.index_select(d_reslut, 1, torch.LongTensor([1]))
    d3 = torch.index_select(d_reslut, 1, torch.LongTensor([2]))
    d4 = torch.index_select(d_reslut, 1, torch.LongTensor([3]))
    d5 = torch.index_select(d_reslut, 1, torch.LongTensor([4]))
    domain1_output = domain1_output * d1
    domain2_output = domain2_output * d2
    domain3_output = domain3_output * d3
    domain4_output = domain1_output * d4
    domain5_output = domain1_output * d5
    err_t_domain1 = loss_func_domain(domain1_output, domain_label)
    err_t_domain2 = loss_func_domain(domain2_output, domain_label)
    err_t_domain3 = loss_func_domain(domain3_output, domain_label)
    err_t_domain4 = loss_func_domain(domain4_output, domain_label)
    err_t_domain5 = loss_func_domain(domain5_output, domain_label)
    err_t_domain = err_t_domain1 + err_t_domain2 + err_t_domain3 + err_t_domain4 + err_t_domain5  # 理论上应该选一个loss，而不是累加
    return err_t_domain, len(xb)

def myScore(y_true, y_pred):
    y_true=y_true.detach().numpy()
    y_pred=y_pred.detach().numpy()
    output_errors = np.ones_like(y_true)
    for i in range(0,len(y_true)):
        try:
            if y_true[i] > y_pred[i]:
                output_errors[i] = np.exp(-1 * ((y_pred[i]-y_true[i]) / 13)) - 1
            else:
                output_errors[i] = np.exp(1 * ((y_pred[i]-y_true[i]) / 10)) - 1
        except:
            print("wrong.")
    Score = np.sum(output_errors)
    return Score
def objective(trial):

    model = define_model(trial)
    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    source_ds, target_ds, testData, testTarget= get_data(batch_size)
    alpha=1

    # Training of the model.
    for epoch in range(n_epochs):
        model.train()
        src_iter = iter(source_ds)
        tar_iter = iter(target_ds)
        R = 2 / (1 + math.exp(-1 * epoch / n_epochs)) - 1
        for idx in range(min(len(src_iter), len(tar_iter))):
            src_data, src_label = src_iter.next()
            tar_data, tar_label = tar_iter.next()
            err_s_rul, err_s_domain, _ = source_loss_batch(model, loss_func_rul, loss_func_domain,
                                                           src_data, src_label, alpha)
            err_t_domain, _ = target_loss_batch(model, loss_func_domain, tar_data, tar_label, alpha)
            optimizer.zero_grad() 
            loss = err_s_rul + 0.5*R * (err_t_domain + err_s_domain)
            loss.backward()
            optimizer.step()
        # Validation of the model.
        model.eval()
        with torch.no_grad():
            rul_output, _, _, _, _, _, _= model.forward(testData, alpha=1)
            yPreds = rul_output * 125
            test_mse = loss_func_rul(yPreds.float(), testTarget.float())
            test_rmse = test_mse.sqrt()
            test_score = myScore(testTarget.float(), yPreds.float())
        # accuracy = test_rmse + test_score / 1000
        accuracy = test_rmse
        trial.report(accuracy, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, timeout=10000000)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

