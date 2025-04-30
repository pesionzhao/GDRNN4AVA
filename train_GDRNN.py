"""
@File    : Train_modelWithGD.py
@Author  : Pesion
@Date    : 2023/9/25
@Desc    : 
"""
import logging
import os
import psutil
import time

import torch
from torch.utils.data import DataLoader
import scipy.io as scio
from NetworkDataSet import M_train_dataset, M_test_dataset
from UNetWithAll import UNetWithAll
from ForwardModel.Zoeppritz import MyZoeppritzOneTheta
import numpy as np
from util.utils import read_yaml
import argparse


parser = argparse.ArgumentParser(description='dual inversion train script',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--device', '-d', help='device id', default="cuda:4")
parser.add_argument('--resume', '-r', help='resume path', default=None)
parser.add_argument('--epoch', '-e', type=int, help='epochs', default=150)
parser.add_argument('--cfg', '-c', help='yaml', default='config/m_data.yaml')
parser.add_argument('--step', '-t', help='RNNs time step', type=int, default=None)
args = parser.parse_args()
device = torch.device(args.device)

cfg = read_yaml(args.cfg)
cfg = argparse.Namespace(**cfg)

LearningRate = 1e-3
epoch = args.epoch if args.epoch is not None else int(cfg.epoch)

# RNNs time step
time_step = args.step if args.step is not None else int(cfg.time_step)

zoeppritz = MyZoeppritzOneTheta.apply
testmat = scio.loadmat(cfg.augdata_path)

# Angular data in radians
Theta1 = np.radians(testmat['theta'][0])
Theta2 = np.radians(testmat['theta'][1])
Theta3 = np.radians(testmat['theta'][2])

# we can inverse part of dataset by modifying uselayers
input_len = cfg.augdata_uselayers['stop']

wavemat = testmat['wavemat'][:input_len, :input_len]

def print_resource_usage(tag=""):
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1024 ** 2
    gpu_mem = torch.cuda.memory_allocated(device) / 1024 ** 2 if torch.cuda.is_available() else 0
    gpu_mem_peak = torch.cuda.max_memory_allocated(device) / 1024 ** 2 if torch.cuda.is_available() else 0

    print(f"[{tag}] CPU Memory Use: {cpu_mem:.2f} MB")
    if torch.cuda.is_available():
        print(f"[{tag}] Current GPU Memory Use: {gpu_mem:.2f} MB, peak: {gpu_mem_peak:.2f} MB")

# elastic parameter to seismic data
def seismic_forward(M_param):
    """

    Args:
        M_param: [(batchsize), 3, input_len]
    Returns:
        seis: [(batchsize), 3, input_len]

    """
    assert M_param.shape !=3 and M_param.shape != 2, "check the shape of M_param"
    if len(M_param.shape) == 3: # batchsize > 1

        vp = M_param[:, 0, :].T
        vs = M_param[:, 1, :].T
        rho = M_param[:, 2, :].T
        ntraces = vp.shape[-1]

        seis_near = zoeppritz(vp, vs, rho, Theta1, wavemat, input_len, ntraces)
        seis_mid = zoeppritz(vp, vs, rho, Theta2, wavemat, input_len, ntraces)
        seis_far = zoeppritz(vp, vs, rho, Theta3, wavemat, input_len, ntraces)
        seis = torch.torch.stack([seis_near, seis_mid, seis_far], dim=1).to(device)  # (batchsize, 3, input_len)

    else: # batchsize == 1

        vp = M_param[0, 0, :]
        vs = M_param[0, 1, :]
        rho = M_param[0, 2, :]

        seis_near = zoeppritz(vp, vs, rho, Theta1, wavemat, input_len, 1)
        seis_mid = zoeppritz(vp, vs, rho, Theta2, wavemat, input_len, 1)
        seis_mid = zoeppritz(vp, vs, rho, Theta3, wavemat, input_len, 1)
        seis = torch.unsqueeze(torch.stack([seis_near, seis_mid, seis_far], dim=0), dim=0).to(device)  # (3, input_len)

    return seis


# loss function
def loss(obs, label, pre_all, pre_label):
    """
    Args:
        obs: obs seismic data
        label: log-well label 
        pre_label: predicted label for data-driven
        pre_all: predicted all elastic parameters for physics-driven

    Returns:
        loss
    """
    pre_label_ntrace = pre_label.shape[0] if len(pre_label.shape)==3 else 1
    pre_all_ntrace = pre_all.shape[0] if len(pre_all.shape)==3 else 1
    # 数据约束 data-driven
    data_loss = cfg.miu * torch.norm(pre_label - label, p=2)
    # 模型约束 physics-driven
    Model_loss2 = (1 - cfg.miu) * torch.norm(seismic_forward(pre_all) - obs, p=2)

    # # TV约束 TV regularization
    # pre_real_diff = torch.diff(pre_all,1,-1)
    # pre_sample_diff = torch.diff(pre_label,1,-1)
    # tv_loss3 = TV_loss(pre_real_diff, torch.zeros_like(pre_real_diff))+TV_loss(pre_sample_diff,torch.zeros_like(pre_sample_diff))

    logging.info('data_loss:{}  Model_loss2:{}'.format(data_loss / pre_label_ntrace, Model_loss2 / pre_all_ntrace))

    loss_value = data_loss + Model_loss2
    return loss_value


# GDRNN
def gradient_descent(M, obs):
    M1 = M.clone()
    M1.requires_grad_()

    ntraces = M.shape[0] if len(M.shape)==3 else 1

    # GDRNN Cell
    for layer_num_iter in range(time_step):
        start_time = time.time()
        seis = seismic_forward(M1)
        J_m = torch.norm(seis - obs, p=2)

        logging.info(f"time_step:{layer_num_iter + 1}: J = {J_m / ntraces}")

        dJ_dm = torch.autograd.grad(outputs=J_m, inputs=M1, retain_graph=True)[0]
        new_grad = model(dJ_dm)  # input old gradient ,output new gradient
        M1 = M1 - new_grad  # gradient descent

        # TODO: M1 = M1 - Par(dj_dm + lamda* new_grad)

        # end_time = time.time()
        # print(f"execution time: {end_time - start_time:.3f} s")
        # print_resource_usage("program end")

    logging.info("-------------")

    return M1


lossLoader = []
if __name__ == '__main__':
    if args.resume is not None:
        model = torch.load(args.resume).to(device)
    else:
        model = UNetWithAll().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LearningRate, weight_decay=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.98)

    # dataset for data-driven
    train_dataset1 = M_train_dataset(cfg.augdata_path, cfg.augdata_traces, cfg.augdata_layers, cfg.augdata_usetraces,
                                     cfg.augdata_uselayers)
    train_dataloader1 = DataLoader(dataset=train_dataset1, batch_size=cfg.batchsize_data_driven, shuffle=False, num_workers=0)
    # dataset for physics-driven
    train_dataset2 = M_test_dataset(cfg.realdata_path, cfg.realdata_traces, cfg.realdata_layers, cfg.realdata_usetraces,
                                    cfg.realdata_uselayers)
    train_dataloader2 = DataLoader(dataset=train_dataset2, batch_size=cfg.batchsize_physics_driven, shuffle=False, num_workers=0)

    # create save_dir
    if os.path.exists(os.path.join(cfg.save_path, 'weights')):
        save_num = 1
        while True:
            save_dir = os.path.join(cfg.save_path, 'weights' + str(save_num))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                break
            elif len(os.listdir(save_dir)) == 0:
                break
            save_num += 1
    else:
        save_dir = os.path.join(cfg.save_path, 'weights')
        os.makedirs(save_dir)
        
    # set logging
    logging.basicConfig(
        level=logging.DEBUG,           
        format="%(asctime)s - %(levelname)s - %(message)s",  
        handlers=[
            logging.FileHandler(os.path.join(save_dir, f"train.log"), mode='w'), 
            logging.StreamHandler() 
        ]
    )

    model.train()
    best_loss = float('inf')
    for i in range(epoch):
        epoch_loss = 0
        logging.info(f"---------Training: epoch = {i + 1} ---------")
        t0 = time.time()

        dataloader_iter = iter(train_dataloader1)

        for S_real, M_initial in train_dataloader2:
            torch.cuda.reset_peak_memory_stats(device=device)
            S_real, M_initial = S_real.to(device), M_initial.to(device)
            try:
                M_sample, S_sample, M_sample_initial = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(train_dataloader1)
                M_sample, S_sample, M_sample_initial = next(dataloader_iter)
            M_sample, S_sample, M_sample_initial = M_sample.to(device), S_sample.to(device), M_sample_initial.to(device)

            # data-driven
            pre_label = gradient_descent(M_sample_initial, S_sample)
            # physics-driven
            pre_all = gradient_descent(M_initial, S_real)

            loss_value = loss(S_real, M_sample, pre_all, pre_label)
            epoch_loss = epoch_loss + loss_value

            # optimizer step
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            scheduler.step()

        lossLoader.append((epoch_loss / len(train_dataset1)).item())
        logging.info(f"epoch{i + 1} loss is {(epoch_loss / len(train_dataset1)).item()}")
        t1 = time.time()
        logging.info(f"epoch{i + 1} execution time is {t1 - t0}")

        if epoch_loss / len(train_dataset1) < best_loss:
            best_loss = epoch_loss / (len(train_dataset1))
            save_path = os.path.join(save_dir, f"GDRNNbest_{i}.pth")
            torch.save(model, save_path)
            logging.info(f"save best model in {save_path}")
        # save model every 10 epochs
        elif i%10==0:
            torch.save(model, os.path.join(save_dir, f"GDRNNlast_{i}.pth"))

    # save loss
    scio.savemat(os.path.join(save_dir, 'loss.mat'), {'loss': lossLoader})
