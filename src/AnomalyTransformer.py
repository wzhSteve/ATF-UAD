import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from math import sqrt
import os
from torch.nn.parameter import Parameter
from .embed import DataEmbedding, TokenEmbedding

class GraphConvolution(nn.Module):  # GCN AHW
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        support = torch.einsum("bsl,le->bse", inputs, self.weight)
        output = torch.einsum("bsl,ble->bse", adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class Splitting(nn.Module):
    def __init__(self, step_size):
        super(Splitting, self).__init__()
        self.step_size = step_size

    def forward(self, x):
        out_list = []
        for i in range(self.step_size):
            out_list.append(x[:, i::self.step_size, :])
        return out_list

class FeatureExtraction(nn.Module):

    def __init__(self, input_length, dim, dropout=0.1):
        super(FeatureExtraction, self).__init__()
        self.softmax2d = nn.Softmax2d()
        self.dropout = nn.Dropout(dropout)
        self.projection_q = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=5, stride=1, padding=2,
                                    padding_mode='circular', bias=False)
        self.projection_k = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=5, stride=1, padding=2,
                                    padding_mode='circular', bias=False)
        self.projection_v = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=5, stride=1, padding=2,
                                    padding_mode='circular', bias=False)
        self.GCN = GraphConvolution(dim, dim)
        self.dropout_t = nn.Dropout(dropout)
        self.projection_out = nn.Conv1d(in_channels=input_length, out_channels=2*input_length, kernel_size=5,
                                      stride=1, padding=2,
                                      padding_mode='circular', bias=False)

    def forward(self, input):
        # input = torch.diff(input, dim=1)
        B, L, D = input.shape
        input_q = self.projection_q(input.permute(0, 2, 1)).permute(0, 2, 1) # B, D, L
        input_k = self.projection_k(input.permute(0, 2, 1)).permute(0, 2, 1)  # B, D, L
        input_v = self.projection_v(input.permute(0, 2, 1)).permute(0, 2, 1)  # B, D, L

        mean_input_q = input_q.mean(1, keepdim=True)
        std_input_q = torch.sqrt(torch.var(input_q, dim=1, keepdim=True, unbiased=False) + 1e-5)
        input_q = (input_q - mean_input_q.repeat(1, L, 1)) / std_input_q

        mean_input_k = input_k.mean(1, keepdim=True)
        std_input_k = torch.sqrt(torch.var(input_k, dim=1, keepdim=True, unbiased=False) + 1e-5)
        input_k = (input_k - mean_input_k.repeat(1, L, 1)) / std_input_k

        scale = 1. / sqrt(D)
        # 内积 scores bhll
        scores = torch.einsum("ble,bse->bls", input_q, input_k)
        cross_value = self.dropout(F.softmax((scale * scores), -3, _stacklevel=5)) # B, D, D
        # GCN
        dim_temp = self.GCN(input_v, cross_value)
        dim_temp = self.dropout_t(F.relu(dim_temp)) + input
        dim_temp = self.projection_out(dim_temp)
        return dim_temp # B, D, D


class TimeReconstructor(nn.Module):
    def __init__(self, win_size, dim, d_model, step_size=2, dropout=0.1, activation="relu"):
        super(TimeReconstructor, self).__init__()
        self.win_size = win_size
        self.dim = dim
        self.d_model = d_model
        self.step_size = step_size

        # Encoding
        self.embedding = DataEmbedding(dim, d_model, dropout)
        self.projection_out = nn.Linear(d_model, dim, bias=True)
        self.split = Splitting(step_size)
        self.feature_list = nn.ModuleList()
        for i in range(step_size):
            self.feature_list.append(FeatureExtraction(win_size//step_size, d_model, dropout))
        # self.feature_extraction = FeatureExtraction(win_size//step_size, d_model, dropout)
        self.re_linear = nn.Linear(self.win_size, 1)

    def forward(self, input):
        B, L, D = input.shape
        input = self.embedding(input)
        input_list = self.split(input)
        output_list = []
        for i in range(self.step_size):
            temp_input = input_list[i]
            # temp_output = self.feature_extraction(temp_input)
            temp_output = self.feature_list[i](temp_input)
            output_list.append(temp_output)

        output = sum(output_list) / self.step_size
        rec = self.projection_out(output)
        # rec = self.re_linear(rec.permute(0, 2, 1)).permute(0, 2, 1)
        return rec, output_list

class FrequenceReconstructor(nn.Module):
    def __init__(self, win_size, dim, d_model, step_size=2, dropout=0.5, activation="relu"):
        super(FrequenceReconstructor, self).__init__()
        self.win_size = win_size
        self.dim = dim
        self.d_model = d_model
        self.step_size = step_size

        # Encoding
        self.embedding = DataEmbedding(dim, d_model, dropout)
        self.projection_out = nn.Linear(d_model, dim, bias=True)
        self.split = Splitting(step_size)
        self.feature_list = nn.ModuleList()
        self.mse = torch.nn.MSELoss(reduction='none')
        for i in range(step_size):
            self.feature_list.append(FeatureExtraction(win_size // step_size, d_model, dropout))
        self.amp_linear = nn.Linear(win_size//2+1, (win_size//2+1), bias=True)
        self.phase_linear = nn.Linear(win_size//2+1, (win_size//2+1), bias=True)
        self.amp_FeatureEx = FeatureExtraction((win_size//2+1)//2, d_model, dropout)
        self.phase_FeatureEx = FeatureExtraction((win_size//2+1)//2, d_model, dropout)
        # self.feature_extraction = FeatureExtraction(win_size//step_size, d_model, dropout)
        self.re_linear = nn.Linear(self.win_size, 1)

    def forward(self, input):
        B, L, D = input.shape
        input = self.embedding(input)
        frequency_comp = torch.fft.rfft(input, dim=1)
        real_comp = frequency_comp.real
        imag_comp = frequency_comp.imag
        amp = self.mse(real_comp, imag_comp)
        phase = torch.arctan(imag_comp/(real_comp + 1e-5))
        amp_c = self.amp_linear(amp.permute(0, 2, 1)).permute(0, 2, 1)
        phase_c = self.phase_linear(phase.permute(0, 2, 1)).permute(0, 2, 1)
        cos_comp = torch.cos(phase_c)
        sin_comp = torch.sin(phase_c)
        phase_new = torch.complex(cos_comp, sin_comp)
        frequency_new = torch.mul(amp_c, phase_new)
        rec = torch.fft.irfft(frequency_new, dim=1)
        rec = self.projection_out(rec)
        # rec = self.re_linear(rec.permute(0, 2, 1)).permute(0, 2, 1)
        return rec


# class FrequenceReconstructor(nn.Module):
#     def __init__(self, win_size, dim, d_model, step_size=2, dropout=0.5, activation="relu"):
#         super(FrequenceReconstructor, self).__init__()
#         self.win_size = win_size
#         self.dim = dim
#         self.d_model = d_model
#         self.step_size = step_size
#
#         # Encoding
#         self.embedding = DataEmbedding(dim, d_model, dropout)
#         self.projection_out = nn.Linear(d_model, dim, bias=True)
#         self.split = Splitting(step_size)
#         self.feature_list = nn.ModuleList()
#         self.mse = torch.nn.MSELoss(reduction='none')
#         for i in range(step_size):
#             self.feature_list.append(FeatureExtraction(win_size // step_size, d_model, dropout))
#         self.amp_linear = nn.Linear(win_size//2+1, (win_size//2+1), bias=True)
#         self.phase_linear = nn.Linear(win_size//2+1, (win_size//2+1), bias=True)
#         self.amp_FeatureEx = FeatureExtraction((win_size//2+1)//2, d_model, dropout)
#         self.phase_FeatureEx = FeatureExtraction((win_size//2+1)//2, d_model, dropout)
#         # self.feature_extraction = FeatureExtraction(win_size//step_size, d_model, dropout)
#         self.re_linear = nn.Linear(self.win_size, 1)
#
#     def forward(self, input):
#         B, L, D = input.shape
#         input = self.embedding(input)
#         frequency_comp = torch.fft.rfft(input, dim=1)
#         real_comp = frequency_comp.real
#         imag_comp = frequency_comp.imag
#
#         # amp = self.mse(real_comp, imag_comp)
#         # phase = torch.arctan(imag_comp/(real_comp + 1e-5))
#         # amp_c = self.amp_linear(amp.permute(0, 2, 1)).permute(0, 2, 1)
#         # phase_c = self.phase_linear(phase.permute(0, 2, 1)).permute(0, 2, 1)
#         # # amp_c = self.amp_FeatureEx(amp_c)
#         # # phase_c = self.phase_FeatureEx(phase_c)
#         # cos_comp = torch.cos(phase_c)
#         # sin_comp = torch.sin(phase_c)
#         # phase_new = torch.complex(cos_comp, sin_comp)
#         # frequency_new = torch.mul(amp_c, phase_new)
#
#         amp_c = self.amp_linear(real_comp.permute(0, 2, 1)).permute(0, 2, 1)
#         phase_c = self.phase_linear(imag_comp.permute(0, 2, 1)).permute(0, 2, 1)
#         frequency_new = torch.complex(amp_c, phase_c)
#         rec = torch.fft.irfft(frequency_new, dim=1)
#         rec = self.projection_out(rec)
#
#         # rec = self.re_linear(rec.permute(0, 2, 1)).permute(0, 2, 1)
#         return rec