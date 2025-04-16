"""
@File :util.py
@Author :Pesion
@Date :2023/9/18
@Desc : Some functions for generating dataset 
"""
import os
import struct
import yaml
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import scipy.signal as signal
# from Forward.Zoeppritz import Zoeppritz
from Forward.Zoeppritz_Complex import ZoeppritzComp
import math


def read_yaml(config_path, method=None):
    """

    Args:
        config_path: path of yaml file
        method: subclass name

    Notes:
        read yaml file

    """
    with open(config_path, 'r', encoding='utf-8') as file:
        # data = file.read()
        cfg = yaml.safe_load(file)
    if method is not None:
        return cfg[method]
    else:
        return cfg

def plot_result(pre, vp, vs, rho, vp_back, vs_back, rho_back):
    # single trace
    if len(vp.shape) == 1:
        layers = len(vp)
        vp_cal = pre[:layers]
        vs_cal = pre[layers:2 * layers]
        rho_cal = pre[2 * layers:]
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='all')
        ax1.plot(vp_cal, label='inv_vp')
        ax1.plot(vp[:], label='vp')
        ax1.plot(vp_back, label='vp_back')
        ax1.set_title('vp curve', loc='left')
        ax1.set_xlabel('layers')
        ax1.legend()

        ax2.plot(vs_cal, label='inv_vs')
        ax2.plot(vs[:], label='vs')
        ax2.plot(vs_back, label='mean_vs')
        ax2.set_title('vs curve', loc='left')
        ax2.set_xlabel('layers')
        ax2.legend()

        ax3.plot(rho_cal, label='inv_rho')
        ax3.plot(rho[:], label='rho')
        ax3.plot(rho_back, label='mean_rho')
        ax3.set_title('rho curve', loc='left')
        ax3.set_xlabel('layers')
        ax3.legend()
    # multi traces
    else:
        ntrace = vp.shape[-1]
        layers = vp.shape[0]
        vpnorm = plt.Normalize(vp.min(), vp.max())  # 用于统一标签与预测值色标
        vsnorm = plt.Normalize(vs.min(), vs.max())
        rhonorm = plt.Normalize(rho.min(), rho.max())
        vp_cal = pre[:layers]
        vs_cal = pre[layers:2 * layers]
        rho_cal = pre[2 * layers:]
        fig, axes = plt.subplots(2, 3)
        imvpcal = axes[0, 0].imshow(vp_cal, aspect='auto', norm=vpnorm)
        axes[0, 0].set_title('vp inv')
        fig.colorbar(imvpcal, ax=[axes[0, 0], axes[1, 0]])
        imvp = axes[1, 0].imshow(vp, aspect='auto', norm=vpnorm)
        axes[1, 0].set_title('vp')
        # fig.colorbar(imvp)

        imvscal = axes[0, 1].imshow(vs_cal, aspect='auto', norm=vsnorm)
        axes[0, 1].set_title('vs inv', )
        fig.colorbar(imvscal, ax=[axes[0, 1], axes[1, 1]])
        imvs = axes[1, 1].imshow(vs, aspect='auto', norm=vsnorm)
        axes[1, 1].set_title('vs')
        # fig.colorbar(imvs)

        imrhocal = axes[0, 2].imshow(rho_cal, aspect='auto', norm=rhonorm)
        axes[0, 2].set_title('rho inv')
        fig.colorbar(imrhocal, ax=[axes[0, 2], axes[1, 2]])
        imrho = axes[1, 2].imshow(rho, aspect='auto', norm=rhonorm)
        axes[1, 2].set_title('rho')
        # fig.colorbar(imrho)


def plot_single_trace(pre, vp, vs, rho, back):
    cut = 30
    layers = int(len(pre) / 3)
    vp_cal = pre[cut:layers]
    vs_cal = pre[layers + cut:2 * layers]
    rho_cal = pre[2 * layers + cut:]
    vp_back = back[cut:layers]
    vs_back = back[layers + cut:2 * layers]
    rho_back = back[2 * layers + cut:]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='all')
    ax1.plot(vp_cal, label='inv_vp')
    ax1.plot(vp[cut:], label='vp')
    ax1.plot(np.exp(vp_back), label='mean_vp')
    ax1.set_title('vp curve', loc='left')
    ax1.set_xlabel('layers')
    ax1.legend()

    ax2.plot(vs_cal, label='inv_vs')
    ax2.plot(vs[cut:], label='vs')
    ax2.plot(np.exp(vs_back), label='mean_vs')
    ax2.set_title('vs curve', loc='left')
    ax2.set_xlabel('layers')
    ax2.legend()

    ax3.plot(rho_cal, label='inv_rho')
    ax3.plot(rho[cut:], label='rho')
    ax3.plot(np.exp(rho_back), label='mean_rho')
    ax3.set_title('rho curve', loc='left')
    ax3.set_xlabel('layers')
    ax3.legend()
    plt.show()


def plot_multi_traces(pre, vp, vs, rho):
    """

    Args:
        pre: [3*layers, traces]
        vp: [layers, traces]
        vs: [layers, traces]
        rho: [layers, traces]

    """
    ntrace = vp.shape[-1]
    layers = vp.shape[0]
    vpnorm = plt.Normalize(vp.min(), vp.max())
    vsnorm = plt.Normalize(vs.min(), vs.max())
    rhonorm = plt.Normalize(rho.min(), rho.max())
    vp_cal = pre[:layers]
    vs_cal = pre[layers:2 * layers]
    rho_cal = pre[2 * layers:]

    fig, axes = plt.subplots(2, 3)
    imvpcal = axes[0, 0].imshow(vp_cal, aspect='auto', norm=vpnorm)
    axes[0, 0].set_title('vp inv')
    fig.colorbar(imvpcal, ax=[axes[0, 0], axes[1, 0]])
    imvp = axes[1, 0].imshow(vp, aspect='auto', norm=vpnorm)
    axes[1, 0].set_title('vp')
    # fig.colorbar(imvp)

    imvscal = axes[0, 1].imshow(vs_cal, aspect='auto', norm=vsnorm)
    axes[0, 1].set_title('vs inv', )
    fig.colorbar(imvscal, ax=[axes[0, 1], axes[1, 1]])
    imvs = axes[1, 1].imshow(vs, aspect='auto', norm=vsnorm)
    axes[1, 1].set_title('vs')
    # fig.colorbar(imvs)

    imrhocal = axes[0, 2].imshow(rho_cal, aspect='auto', norm=rhonorm)
    axes[0, 2].set_title('rho inv')
    fig.colorbar(imrhocal, ax=[axes[0, 2], axes[1, 2]])
    imrho = axes[1, 2].imshow(rho, aspect='auto', norm=rhonorm)
    axes[1, 2].set_title('rho')
    # fig.colorbar(imrho)
    plt.show()


def augdata(vp, vs, rho, index: list, addtrace=9):
    """

    Args:
        addtrace: number of traces to add
        vp: [layers, traces]
        vs:
        rho:
        index: well index

    Returns:
        label augmentation: vp_aug, vs_aug, rho_aug

    Notes:
        Due to the small amount of logging data, it is necessary to increase the number of samples through data augmenting. 
        Here, the method of adding noise is used to expand, and the simulated logging data is obtained through the index index of the original data

    """
    chose_vp = []
    chose_vs = []
    chose_rho = []
    for i in index:
        chose_vp.append(vp[:, i - 1])
        chose_vs.append(vs[:, i - 1])
        chose_rho.append(rho[:, i - 1])

    fig1, ax_vp = plt.subplots(1, len(index), sharey='all')
    ax_vp[0].set_ylabel('vp')
    fig2, ax_vs = plt.subplots(1, len(index), sharey='all')
    ax_vs[0].set_ylabel('vs')
    fig3, ax_rho = plt.subplots(1, len(index), sharey='all')
    ax_rho[0].set_ylabel('rho')
    fig1.suptitle(f'vp label')
    fig2.suptitle(f'vs label')
    fig3.suptitle(f'rho label')
    for i in range(len(index)):
        ax_vp[i].plot(chose_vp[i])
        ax_vp[i].set_title(f'trace {index[i]}')
        ax_vp[i].set_xlabel('layers')
        ax_vs[i].plot(chose_vs[i])
        ax_vs[i].set_title(f'trace {index[i]}')
        ax_vs[i].set_xlabel('layers')
        ax_rho[i].plot(chose_rho[i])
        ax_rho[i].set_title(f'trace {index[i]}')
        ax_rho[i].set_xlabel('layers')

    noise = np.random.normal(0, 0.01, (vp.shape[0], addtrace))
    aug_vp = None
    aug_vs = None
    aug_rho = None
    for i in range(len(index)):
        if i == 0:
            aug_vp = np.concatenate([chose_vp[i][:, None], chose_vp[i][:, None] + noise], axis=1)
            aug_vs = np.concatenate([chose_vs[i][:, None], noise + chose_vs[i][:, None]], axis=1)
            aug_rho = np.concatenate([chose_rho[i][:, None], noise + chose_rho[i][:, None]], axis=1)
        else:
            aug_vp = np.concatenate([aug_vp, chose_vp[i][:, None], chose_vp[i][:, None] + noise], axis=1)
            aug_vs = np.concatenate([aug_vs, chose_vs[i][:, None], noise + chose_vs[i][:, None]], axis=1)
            aug_rho = np.concatenate([aug_rho, chose_rho[i][:, None], noise + chose_rho[i][:, None]], axis=1)
    fig, ax = plt.subplots(1, 3)
    ax[0].plot(aug_vp)
    ax[0].set_title('aug_vp')
    ax[1].plot(aug_vs)
    ax[1].set_title('aug_vs')
    ax[2].plot(aug_rho)
    ax[2].set_title('aug_rho')
    plt.show()

    return aug_vp, aug_vs, aug_rho


def back_model(vp, vs, rho, f0=5):
    """

    Args:
        vp:
        vs:
        rho:
        f0: cut-off frequency

    Returns:
        background model [vp_back, vs_back, rho_back]

    """
    N = 2
    dt = 0.002
    Wn = 2 * f0 / (1 / (dt))  # Normalized cutoff frequency
    b, a = signal.butter(N, Wn, btype='low')
    vp_back = signal.filtfilt(b, a, vp, axis=0)
    vs_back = signal.filtfilt(b, a, vs, axis=0)
    rho_back = signal.filtfilt(b, a, rho, axis=0)
    fig, ax = plt.subplots(1, 3)
    if len(vp_back.shape) == 2:
        ax[0].imshow(vp_back, aspect='auto')
        ax[0].set_title('vp_back')
        ax[1].imshow(vs_back, aspect='auto')
        ax[1].set_title('vs_back')
        ax[2].imshow(rho_back, aspect='auto')
        ax[2].set_title('rho_back')
    elif len(vp_back.shape) == 1:
        ax[0].plot(vp_back)
        ax[0].set_title('vp_back')
        ax[1].plot(vs_back)
        ax[1].set_title('vs_back')
        ax[2].plot(rho_back)
        ax[2].set_title('rho_back')
    return vp_back, vs_back, rho_back


def v2seis(vp, vs, rho, wavemat, theta1, theta2, theta3):
    """

    Args:
        vp: [layers, traces]
        vs:
        rho:
        wavemat: [layers, layers]
        theta1: degree!! not radian
        theta2:
        theta3:

    Returns:
        siemic data
    """
    # changeable parameter
    forward = ZoeppritzComp(vp.shape[-1], vp.shape[0])

    seismic1 = forward.forward(vp, vs, rho, np.radians(theta1), wavemat)
    seismic2 = forward.forward(vp, vs, rho, np.radians(theta2), wavemat)
    seismic3 = forward.forward(vp, vs, rho, np.radians(theta3), wavemat)
    return seismic1, seismic2, seismic3


def generate_ricker(layers, f0, dt0):
    t = (np.arange(layers + 1) * dt0)[1:]
    w = (1 - 2 * (np.pi * f0 * (t - 1 / f0)) ** 2) * np.exp(-((np.pi * f0 * (t - 1 / f0)) ** 2))  # wavelet sequence
    npad = layers - 1
    h_mtx = np.vstack([np.pad(w, (i, npad - i), 'constant', constant_values=(0, 0)) for i in
                       range(layers)]).T  # [2*layers-1, layers]
    h_mtx = h_mtx[math.floor(1 / f0 / dt0): math.floor(1 / f0 / dt0) + layers, :]
    return h_mtx


def generate_dataset(vp, vs, rho, index: list, cutoff_f=2, name='custom'):
    """

    Args:
        vp:
        vs:
        rho:
        index: traces used as log-well
        cutoff_f: cut-off frequency
        name: dataset name

    Notes:
        get name_train_dataset.mat and name_test_dataset.mat

    """
    train_dataset_path = os.path.join('../dataset', name + '_train_dataset.mat')
    test_dataset_path = os.path.join('../dataset', name + '_test_dataset.mat')
    train_data = generate_train_dataset(vp, vs, rho, cutoff_f, index)
    scio.savemat(train_dataset_path, train_data)
    test_data = generate_test_dataset(vp, vs, rho, cutoff_f)
    scio.savemat(test_dataset_path, test_data)



def generate_train_dataset(vp, vs, rho, cutoff_f, index):
    """

    Args:
        vp:
        vs:
        rho:
        cutoff_f: cut-off frequency
        index: traces used as log-well
    Note:
        Generate a training set using real logging data which is defined by index
    Returns:
        train_data = {vp_back, vs_back, rho_back, vp_aug, vs_aug, rho_aug, seis_s, seis_m, seis_l, wavemat, theta}

    """
    # parameter
    # TODO The frequency, sampling interval, incidence Angle, and number of tracks extracted are controlled externally as variable parameters
    f0 = 30
    dt0 = 0.002
    theta1 = np.arange(1, 16)
    theta2 = np.arange(16, 31)
    theta3 = np.arange(31, 46)
    vp, vs, rho = augdata(vp, vs, rho, index)

    vp_back, vs_back, rho_back = back_model(vp, vs, rho, cutoff_f)
    wavemat = generate_ricker(vp.shape[0], f0, dt0)
    seismic1, seismic2, seismic3 = v2seis(vp, vs, rho, wavemat, theta1, theta2, theta3)
    dataset = {}
    dataset.update({'vp_back': vp_back})
    dataset.update({'vs_back': vs_back})
    dataset.update({'rho_back': rho_back})
    dataset.update({'vp_aug': vp})
    dataset.update({'vs_aug': vs})
    dataset.update({'rho_aug': rho})
    dataset.update({'seis_s': seismic1})
    dataset.update({'seis_m': seismic2})
    dataset.update({'seis_l': seismic3})
    dataset.update({'wavemat': wavemat})
    dataset.update({'theta': [theta1, theta2, theta3]})
    return dataset


def generate_test_dataset(vp, vs, rho, cutoff_f):
    """

    Args:
        vp: 
        vs:
        rho:
        cutoff_f: cut-off frequency

    Returns:
        dataset = {vp, vs, rho, vp_back, vs_back, rho_back, seis_s, seis_m, seis_l, wavemat, theta}

    """
    # parameter
    # TODO The frequency, sampling interval, incidence Angle, and number of tracks extracted are controlled externally as variable parameters
    f0 = 30
    dt0 = 0.002
    wavemat = generate_ricker(vp.shape[0], f0, dt0)
    theta1 = np.arange(1, 16)
    theta2 = np.arange(16, 31)
    theta3 = np.arange(31, 46)

    vp_back, vs_back, rho_back = back_model(vp, vs, rho, cutoff_f)
    seismic1, seismic2, seismic3 = v2seis(vp, vs, rho, wavemat, theta1, theta2, theta3)
    dataset = {}
    dataset.update({'vp': vp})
    dataset.update({'vs': vs})
    dataset.update({'rho': rho})
    dataset.update({'vp_back': vp_back})
    dataset.update({'vs_back': vs_back})
    dataset.update({'rho_back': rho_back})
    dataset.update({'seis_s': seismic1})
    dataset.update({'seis_m': seismic2})
    dataset.update({'seis_l': seismic3})
    dataset.update({'wavemat': wavemat})
    dataset.update({'theta': [theta1, theta2, theta3]})
    return dataset


def ibm_to_ieee(ibm_float_bytes):
    ibm_int = struct.unpack('>I', ibm_float_bytes)[0]

    sign = -1 if (ibm_int >> 31) & 0x1 else 1

    exponent = ((ibm_int >> 24) & 0x7F) - 64

    mantissa = ibm_int & 0xFFFFFF

    fraction = mantissa / float(0x1000000)

    ieee_float = sign * (16 ** exponent) * fraction

    return ieee_float
def read_SEGY(datapath, savepath):
    with open(datapath, 'rb') as f:

        # Read the binary file header
        bin_header = f.read(3600)

        # Get the relevant information in the file header
        sample_interval = struct.unpack_from(">h", bin_header, 3216)[0]
        num_samples = struct.unpack_from(">h", bin_header, 3220)[0]
        data_format = struct.unpack_from(">h", bin_header, 3222)[0]

        # Read all trace data
        traces = []
        while True:
            trace_header = f.read(240)
            if not trace_header:
                break
            cdp_cur = struct.unpack_from(">i", trace_header, 20)[0]
            num_samples = struct.unpack_from(">h", trace_header, 114)[0]
            dt = struct.unpack_from(">h", trace_header, 116)[0]
            if cdp_cur >= 0:
                amp = []
                for _ in range(num_samples):
                    # if unpack the IBM format
                    ibm = ibm_to_ieee(f.read(4))
                    amp.append(ibm)
                    # if unpack the IEEE format
                    # amp.append(struct.unpack_from(">f", f.read(4))[0])
                traces.append(amp)
            else:
                f.read(num_samples * 4)
                continue

        traces = np.array(traces)
        np.save(savepath, traces) 
