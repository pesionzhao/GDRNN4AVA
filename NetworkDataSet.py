"""
@File :NetworkDataSet.py
@Author :Pesion
@Date :2024/9/19
@Desc : custom dataset
"""
import scipy.io as scio
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

class M_train_dataset(Dataset):
    def __init__(self, data_path, traces, layers, train_traces: dict, train_layers: dict):
        super(M_train_dataset, self).__init__()
        self.Train_Data = scio.loadmat(data_path)
        self.traces = traces
        self.layers = layers
        self.start = train_traces['start'] if train_traces['start'] is not None else 0
        self.stop = train_traces['stop'] if train_traces['stop'] is not None else self.traces
        self.step = train_traces['step'] if train_traces['step'] is not None else 1
        self.layerstart = train_layers['start'] if train_layers['start'] is not None else 0
        self.layerstop = train_layers['stop'] if train_layers['stop'] is not None else self.layers
        self.layerstep = train_layers['step'] if train_layers['step'] is not None else 1

    def __len__(self):
        return int((self.stop - self.start) / self.step)

    def __getitem__(self, index):
        index = index + self.start
        # [layers]
        vp_label = torch.tensor(self.Train_Data['vp_aug'][self.layerstart:self.layerstop, index], dtype=torch.float32)
        vs_label = torch.tensor(self.Train_Data['vs_aug'][self.layerstart:self.layerstop, index], dtype=torch.float32)
        den_label = torch.tensor(self.Train_Data['rho_aug'][self.layerstart:self.layerstop, index], dtype=torch.float32)
        # [3, layers]
        M_sample = torch.stack([vp_label, vs_label, den_label], dim=0)
        # [layers, angles]
        seis_near = torch.tensor(self.Train_Data['seis_s'][index,self.layerstart:self.layerstop,:], dtype=torch.float32)
        seis_mid = torch.tensor(self.Train_Data['seis_m'][index,self.layerstart:self.layerstop,:], dtype=torch.float32)
        seis_far = torch.tensor(self.Train_Data['seis_l'][index,self.layerstart:self.layerstop,:], dtype=torch.float32)
        #  [3, layers, angles]
        seis = torch.stack([seis_near, seis_mid, seis_far], dim=0)
        # [layers]
        vp_sample_initial = torch.tensor(self.Train_Data['vp_back'][self.layerstart:self.layerstop, index], dtype=torch.float32)
        vs_sample_initial = torch.tensor(self.Train_Data['vs_back'][self.layerstart:self.layerstop, index], dtype=torch.float32)
        den_sample_initial = torch.tensor(self.Train_Data['rho_back'][self.layerstart:self.layerstop, index], dtype=torch.float32)
        # [3, layers]
        M_sample_initial = torch.stack([vp_sample_initial, vs_sample_initial, den_sample_initial], dim=0)

        return M_sample, torch.sum(seis,-1), M_sample_initial

class M_test_dataset(Dataset):
    def __init__(self, data_path, traces, layers, train_traces: dict, train_layers: dict):
        super(M_test_dataset, self).__init__()
        self.Train_Data = scio.loadmat(data_path)
        self.traces = traces
        self.layers = layers
        self.start = train_traces['start'] if train_traces['start'] is not None else 0
        self.stop = train_traces['stop'] if train_traces['stop'] is not None else self.traces
        self.step = train_traces['step'] if train_traces['step'] is not None else 1
        self.layerstart = train_layers['start'] if train_layers['start'] is not None else 0
        self.layerstop = train_layers['stop'] if train_layers['stop'] is not None else self.layers
        self.layerstep = train_layers['step'] if train_layers['step'] is not None else 1
        self.use_layers = (self.layerstop - self.layerstart) // self.layerstep

    def __len__(self):
        return int((self.stop - self.start) / self.step)

    def __getitem__(self, index):
        index = index + self.start
        # [layers, angles]
        seis_near = torch.tensor(self.Train_Data['seis_s'][index, self.layerstart:self.layerstop, :], dtype=torch.float32)
        seis_mid = torch.tensor(self.Train_Data['seis_m'][index, self.layerstart:self.layerstop, :], dtype=torch.float32)
        seis_far = torch.tensor(self.Train_Data['seis_l'][index, self.layerstart:self.layerstop, :], dtype=torch.float32)
        # [3, layers, angles]
        seis = torch.stack([seis_near, seis_mid, seis_far], dim=0)
        # [layers]
        vp_initial = torch.tensor(self.Train_Data['vp_back'][self.layerstart:self.layerstop, index], dtype=torch.float32)
        vs_initial = torch.tensor(self.Train_Data['vs_back'][self.layerstart:self.layerstop, index], dtype=torch.float32)
        den_initial = torch.tensor(self.Train_Data['rho_back'][self.layerstart:self.layerstop, index], dtype=torch.float32)
        # [layers, angles]
        M_initial = torch.stack([vp_initial, vs_initial, den_initial], dim=0)

        return torch.sum(seis,-1), M_initial

