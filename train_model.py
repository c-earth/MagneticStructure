import time
import os

import torch
from e3nn.o3 import Irreps, Irrep

from utils.config import Config
from utils.backend import project_work_space, create_folders, backup, random_hyper, non_linear_selector
from utils.data import load_data_sets
from utils.model import GraphNetwork
from utils.loss import AddLosses, MSEMagcoLoss, L1MagcoLoss
from utils.train import train
from utils.basis import BsplineBasis, FourierBasis, GaussianBasis, ReLUBasis, PWLBasis
from utils.hs_extract import hs_extract
from utils.tpe import TPE


cf = Config()
run_time = time.strftime('%y%m%d_%H%M%S', time.localtime())

cf.section('Project directory structure')
cf.data_name = '250313_103742'
cf.train_file, cf.project_dir = project_work_space()
cf.utils_dir = os.path.join(cf.project_dir, 'utils')
cf.data_dir = os.path.join(cf.project_dir, 'data')
cf.models_dir = os.path.join(cf.project_dir, 'models')
if cf.data_name == '':
    cf.data_name = run_time
cf.model_data_dir = os.path.join(cf.data_dir, cf.data_name)
cf.raw_data_file_dir = os.path.join(cf.data_dir, 'raw_small')
cf.data_file_dir = os.path.join(cf.model_data_dir, 'pkl')
cf.data_dict_file_dir = os.path.join(cf.model_data_dir, 'dict')
cf.data_sets_file_dirs = [os.path.join(cf.model_data_dir, f'sets/{partition}') for partition in ['tr_sets', 'va_sets', 'te_sets']]
create_folders(cf.models_dir, cf.model_data_dir, cf.data_file_dir, cf.data_dict_file_dir, *cf.data_sets_file_dirs)

cf.section('General Information')
cf.project_name = os.path.basename(cf.project_dir)
cf.new_name = f'{cf.project_name}_{run_time}'
cf.old_name = ''
cf.model_label = ''
cf.device = 'cuda' if torch.cuda.is_available() else 'cpu'
cf.seed = 10939963
if cf.model_label == '':
    cf.model_label = cf.new_name

hs_fields = ['lmax', 'r_max', 'radial_basis_num', 'node_feat_mul_list', 'node_attr_mul_list', 'edge_length_mul_list', 'node_hid_mul_list', 'basis', 'basis_kwargs']
bases_inv = {'None': [0, 1, 2, 3, 4], 'BsplineBasis': [5], 'FourierBasis': [6], 'GaussianBasis': [7], 'ReLUBasis': [8], 'PWLBasis': [9]}

cf.section('Control Switches')
cf.verbose = True
cf.use_backup = True
cf.use_old_model = False
if cf.use_old_model and cf.old_name != '':
    cf.run_name = cf.old_name
else:
    cf.run_name = cf.new_name
    backup([], cf.models_dir, cf.run_name)

cf.hyper_search = False
cf.tpe = False
priors = hs_extract(cf.models_dir, hs_fields, bases_inv, os.path.join(cf.models_dir, cf.run_name))
if cf.tpe:
    tpe = TPE(priors, 0.2, n_sample = 20, bandwidth = 2.0, bandwidth_decay = 0.985)
    tpe.segment()
    cf.next_hs_dict = tpe.next()
else:
    cf.next_hs_dict = dict(zip(priors.columns.values[1:-1], [None] * len(priors.columns.values[1:-1])))
cf.use_radial_one_hot = random_hyper([True, False], True, cf.hyper_search) if not cf.tpe else random_hyper(list(range(5, 21)) + [1], 10, cf.hyper_search, cf.next_hs_dict['radial_basis_num']) != 1
cf.load_best = False
cf.reset_checkpoint = False

print('start training: ', cf.run_name)

cf.section('Model`s Hyperparameters')
cf.max_iter = 64
cf.lmax = random_hyper([0, 1, 2], 1, cf.hyper_search)
cf.r_max = random_hyper([4., 5., 6.], 4., cf.hyper_search)
cf.radial_basis_num = random_hyper(list(range(5, 21)), 10, cf.hyper_search) if cf.use_radial_one_hot else 1
cf.irreps_edge_direct = Irreps.spherical_harmonics(cf.lmax)
cf.descriptors = ['number']
cf.irrep_node_feat_list = ['0e'] * len(cf.descriptors)
cf.node_feat_mul_list_num_hid = random_hyper([0, 1], 0, cf.hyper_search)
cf.node_feat_mul_list_hid_mul = random_hyper(list(range(4, 101)), 64, cf.hyper_search)
cf.node_feat_mul_list_out_mul = random_hyper(list(range(4, 101)), 64, cf.hyper_search)
cf.node_feat_mul_list = [118] + [cf.node_feat_mul_list_hid_mul] * cf.node_feat_mul_list_num_hid + [cf.node_feat_mul_list_out_mul]
cf.irrep_node_attr_list = ['0e']
cf.node_attr_mul_list_num_hid = random_hyper([0, 1], 0, cf.hyper_search)
cf.node_attr_mul_list_hid_mul = random_hyper(list(range(4, 101)), 64, cf.hyper_search)
cf.node_attr_mul_list_out_mul = random_hyper(list(range(4, 101)), 64, cf.hyper_search)
cf.node_attr_mul_list = [118] + [cf.node_attr_mul_list_hid_mul] * cf.node_attr_mul_list_num_hid + [cf.node_attr_mul_list_out_mul]
cf.irrep_edge_length_list = ['0e']
cf.edge_length_mul_list_num_hid = random_hyper([0, 1, 2], 1, cf.hyper_search)
cf.edge_length_mul_list_hid_mul = random_hyper(list(range(1, 201)), 100, cf.hyper_search)
cf.edge_length_mul_list = [cf.radial_basis_num] + [cf.edge_length_mul_list_hid_mul] * cf.edge_length_mul_list_num_hid
cf.irrep_node_hid_list = [Irrep(l, p) for l in range(cf.lmax + 1) for p in [-1, 1]]
cf.node_hid_mul_list_num_hid = random_hyper([1, 2, 3], 2, cf.hyper_search)
cf.node_hid_mul_list_hid_mul = random_hyper(list(range(1, 51)), 32, cf.hyper_search)
cf.node_hid_mul_list = [cf.node_hid_mul_list_hid_mul] * cf.node_hid_mul_list_num_hid + [1]
cf.irrep_out_list = ['1e']
cf.out_mul_list = [1]

cf.section('Nonlinear Layer Setting')
bases = {0: None, 1: None, 2: None, 3: None, 4: None, 5: BsplineBasis, 6: FourierBasis, 7: GaussianBasis, 8: ReLUBasis, 9: PWLBasis}
cf.basis_idx = random_hyper(list(range(10)), 0, cf.hyper_search)
cf.basis = bases[cf.basis_idx]
cf.basis_num = random_hyper([3, 5, 7, 9, 11, 13], 9, cf.hyper_search)
cf.k = random_hyper(range(1, cf.basis_num), 3, cf.hyper_search)
if cf.basis in [BsplineBasis, FourierBasis, GaussianBasis, ReLUBasis, PWLBasis]:
    cf.basis_kwargs = {'basis_num': cf.basis_num, 'grid_range': [0, 1]}
else:
    cf.basis_kwargs = dict()

if cf.basis in [BsplineBasis, GaussianBasis, ReLUBasis, PWLBasis]:
    cf.basis_kwargs['k'] = cf.k

layer_num_dict = {'node_feat_emb': len(cf.node_feat_mul_list) - 1,
                  'node_attr_emb': len(cf.node_attr_mul_list) - 1,
                  'edge_length_emb': len(cf.edge_length_mul_list),
                  'conv': len(cf.node_hid_mul_list),
                  'out_emb': len(cf.out_mul_list) - 1}

cf.act_list_dict_list_dict, cf.act_kwargs_list_dict_list_dict = non_linear_selector(layer_num_dict, cf.use_radial_one_hot, [] if cf.basis is None else [cf.basis], [] if cf.basis is None else [cf.basis_kwargs])

cf.section('Data')
cf.batch_size = 1
cf.tr_va_te_ratios = [0.8, 0.1, 0.1]
data_loaders = load_data_sets(
    cf.data_sets_file_dirs,
    cf.data_dict_file_dir,
    cf.data_file_dir,
    cf.raw_data_file_dir,
    cf.r_max,
    cf.radial_basis_num,
    cf.irreps_edge_direct,
    cf.descriptors,
    cf.tr_va_te_ratios,
    cf.batch_size,
    cf.seed
)

cf.section('Model')
model = GraphNetwork(
    cf.irrep_node_feat_list,
    cf.node_feat_mul_list,
    cf.irrep_node_attr_list,
    cf.node_attr_mul_list,
    cf.irrep_edge_length_list,
    cf.edge_length_mul_list,
    cf.irrep_node_hid_list,
    cf.node_hid_mul_list,
    cf.irrep_out_list,
    cf.out_mul_list,
    cf.irreps_edge_direct,
    cf.act_list_dict_list_dict,
    cf.act_kwargs_list_dict_list_dict
)
cf.model = str(model)
cf.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

cf.section('Optimization')
tr_loss_fn = AddLosses([MSEMagcoLoss()], reduction = 'mean')
te_loss_fn = AddLosses([MSEMagcoLoss()])
cf.tr_loss_fn = tr_loss_fn.__repr__()
cf.te_loss_fn = te_loss_fn.__repr__()
cf.lr = 0.005
cf.weight_decay = 0.05
opt = torch.optim.AdamW(model.parameters(), lr = cf.lr, weight_decay = cf.weight_decay)
cf.opt = str(opt)
cf.schedule_gamma = 0.96
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma = cf.schedule_gamma)
cf.schduler = scheduler.__class__.__name__

if cf.use_backup and not cf.use_old_model:
    backup([cf.train_file, cf.utils_dir], cf.models_dir, cf.run_name)
cf.save(os.path.join(cf.models_dir, cf.run_name))

log_file = open(os.path.join(cf.models_dir, cf.run_name, 'log.txt'), 'a')
print(cf, flush = True, file = log_file)

train(
    model,
    cf.model_label,
    opt,
    data_loaders,
    tr_loss_fn,
    te_loss_fn,
    cf.max_iter,
    scheduler,
    cf.device,
    cf.batch_size,
    cf.verbose,
    os.path.join(cf.models_dir, cf.run_name),
    cf.load_best,
    cf.reset_checkpoint,
    log_file
)

log_file.close()
