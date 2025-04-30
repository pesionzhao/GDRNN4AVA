"""
@File    : predict_original.py
@Author  : Pesion
@Date    : 2024/9/27
@Desc    : predict elastic parameter by original data-driven and physics-driven method
"""
import os
import torch
from NetworkDataSet import M_test_dataset
from util.utils import read_yaml
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import scipy.io as scio

parser = argparse.ArgumentParser(description='dual inversion predict script',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--weight', '-w', help='weight path', default="")
parser.add_argument('--output', '-o', help='output name', default=None)
parser.add_argument('--device', '-d', help='cuda: number', default="cuda:0")
parser.add_argument('--name', '-n', help='output dir name', default="custom_name")
parser.add_argument('--cfg', '-c', help='yaml', default="config/m_data.yaml")
args = parser.parse_args()
# args = parser.parse_args(['--weights','./weights_dir/weights6/last.pth', '--device', 'cuda:0']) # custom_args
config_path = args.cfg
weights_path = args.weight
save_name = args.output
if save_name is None:
    name_list = os.path.splitext(weights_path)[0].split('/')[-2:]
    save_name = "_".join(name_list)

if not os.path.exists('./output'):
    os.mkdir('./output')
savedir = os.path.join('./output', args.name)
if not os.path.exists(savedir):
    os.mkdir(savedir)
device = torch.device(args.device) 
cfg = read_yaml(config_path=config_path)
cfg = argparse.Namespace(**cfg)

test_dataset = M_test_dataset(cfg.testdata_path, cfg.testdata_traces, cfg.testdata_layers, cfg.testdata_usetraces,
                              cfg.testdata_uselayers)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=cfg.batchsize, num_workers=0)

model = torch.load(weights_path, map_location=device)
model.to(device)
model.eval()

output_list = []
inital_list = []
with torch.no_grad():
    for S_sample, M_sample_initial in test_dataloader:
        S_sample, M_sample_initial = S_sample.to(device), M_sample_initial.to(device)
        output1 = model(S_sample)
        output_list.append(output1)
        inital_list.append(M_sample_initial)

    output = torch.cat(output_list, dim=0)
    M_sample_initial = torch.cat(inital_list, dim=0)

output = cfg.lam * output + M_sample_initial

Inv_vp = output[:, 0, :].cpu().detach().numpy()
Inv_vs = output[:, 1, :].cpu().detach().numpy()
Inv_rho = output[:, 2, :].cpu().detach().numpy()

scio.savemat(os.path.join(savedir, save_name + '.mat'), {'vp': Inv_vp, 'vs': Inv_vs, 'rho': Inv_rho})
