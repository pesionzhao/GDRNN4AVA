"""
@File    : predict_modelWithGD.py
@Author  : Pesion
@Date    : 2024/9/27
@Desc    : 
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
from torch.utils.data import DataLoader
import scipy.io as io
from NetworkDataSet import M_test_dataset
from ForwardModel.Zoeppritz import MyZoeppritzOneTheta
from util.utils import read_yaml
import argparse
import scipy.io as scio
import logging

parser = argparse.ArgumentParser(description='dual inversion predict script',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--weight', '-w', help='weight path', default="./weights_dir/weights/GDRNNbest_21.pth")
parser.add_argument('--output', '-o', help='output file name', default=None)
parser.add_argument('--device', '-d', help='device id', default="cuda:0")
parser.add_argument('--name', '-n', help='save folder name', default="custom_name")
parser.add_argument('--cfg', '-c', help='yaml', default='config/m_data.yaml')
parser.add_argument('--step', '-t', help='RNNs time step', type=int, default=9)
parser.add_argument('--batch', '-b', help='batchsize', type=int, default=None)
args = parser.parse_args()
# args = parser.parse_args(['--weights','./weights_dir/weights6/last.pth', '--device', 'cuda:0']) # custom_args
config_path = args.cfg
cfg = read_yaml(config_path=config_path) 
cfg = argparse.Namespace(**cfg)
weights_path = args.weight
save_name = args.output
time_step = args.step if args.step is not None else int(cfg.time_step)

# set save dir
if save_name is None:
    name_list = os.path.splitext(weights_path)[0].split('/')[-2:] 
    save_name = "_".join(name_list)

if not os.path.exists('./output'):
    os.mkdir('./output')
savedir = os.path.join('./output', args.name)
if not os.path.exists(savedir):
    os.mkdir(savedir)

# set logging
if not os.path.exists('log'):
    os.mkdir('log')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler('log/'+save_name+'.log', mode='w'),
        logging.StreamHandler() 
    ]
)

device = torch.device(args.device)  
batchsize = args.batch if args.batch is not None else cfg.batchsize_predict
test_dataset = M_test_dataset(cfg.testdata_path, cfg.testdata_traces, cfg.testdata_layers, cfg.testdata_usetraces,
                              cfg.testdata_uselayers)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batchsize, num_workers=0)

model = torch.load(weights_path).to(device) 
model.eval()

zoeppritz = MyZoeppritzOneTheta.apply
testmat = scio.loadmat(cfg.testdata_path)

# 
Theta1 = np.radians(testmat['theta'][0])
Theta2 = np.radians(testmat['theta'][1])
Theta3 = np.radians(testmat['theta'][2])
# input_len = testmat['vp_back'].shape[0]
input_len = test_dataset.use_layers

wavemat = testmat['wavemat'][:input_len, :input_len]

def seismic_obj_function(M_param, S, wavemat):

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

    J_m = torch.norm(seis - S, p=2)
    # TODO: J_m = norm(sys - S, p=2) + TV_loss
    return J_m


def model_driving(M, S, wavemat):
    M1 = M.clone()
    M1.requires_grad_()
    use_net = True # True: Use GDRNN, Flase: traditional gradient descent
    step = 0.25 # step_size for 
    for layer_num_iter in range(time_step+1):
        ntraces = M.shape[0] if len(M.shape)==3 else 1
        J_m = seismic_obj_function(M1, S, wavemat)
        logging.info("J_m:{}".format(J_m / ntraces))
        if layer_num_iter == time_step-1:
            break
        if use_net:
            if layer_num_iter != 0:
                if J_m > old_J: # new gradient is not better
                    M1 = old_M1 # use old_M1
                    use_net = False
                    break
        dJ_dm = torch.autograd.grad(outputs=J_m, inputs=M1, retain_graph=False)[0] 
        
        # stored in old_M1
        old_M1 = M1.clone()
        if use_net:
            with torch.no_grad():
                new_grad = model(dJ_dm)
            M1 = M1 - new_grad
        else:
            if layer_num_iter == 5:
                step/=2
            elif layer_num_iter == 10:
                step/=2
            if layer_num_iter and old_J - J_m < -0.1:
                break
            M1 = M1 - step*dJ_dm

        old_J = J_m.detach().clone()
    logging.info("-------------")

    return M1


output_list = []
i = 1
for S_sample, M_sample_initial in test_dataloader:
    logging.info('%d th traces inversing' % (i * batchsize))
    S_sample, M_sample_initial = S_sample.to(device), M_sample_initial.to(device)
    output1 = model_driving(M_sample_initial, S_sample, wavemat)
    output_list.append(output1)
    i = i + 1
output = torch.cat(output_list, dim=0)

Inv_vp = output[:, 0, :].cpu().detach().numpy()
Inv_vs = output[:, 1, :].cpu().detach().numpy()
Inv_rho = output[:, 2, :].cpu().detach().numpy()

scio.savemat(os.path.join(savedir, save_name + '.mat'), {'vp': Inv_vp, 'vs': Inv_vs, 'rho': Inv_rho})

