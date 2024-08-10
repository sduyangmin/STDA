import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default= 1000,
                        help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='training batch size')
    parser.add_argument('--lr', type=float, default=1e-4, # for DeepLGR, lr= 0.005
                        help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.9,
                        help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.9,
                        help='adam: decay of second order momentum of gradient')
    parser.add_argument('--n_channels', type=int, default=128,
                        help='C_2 in paper')
    parser.add_argument('--channels', type=int, default=1,
                        help='number of flow image channels')
    parser.add_argument('--seed', type=int, default=2017, help='random seed')
    parser.add_argument('--use_exf', action='store_true', default=True,
                        help='External influence factors')
    parser.add_argument('--height', type=int, default=16,
                        help='height of the input map')
    parser.add_argument('--width', type=int, default=16,
                        help='weight of the input map')
    parser.add_argument('--scale_factor', type=int, default=4,
                        help='upscaling factor')
    parser.add_argument('--model', type=str, default='STDA',
                        help='chose model to use', 
                        choices=['UrbanFM', 'FODE', 'UrbanODE', 'DeepLGR', 'UrbanPy', 'CUFAR','STDA'])
    parser.add_argument('--scaler_X', type=int, default=1500,
                        help='scaler of coarse-grained flows')
    parser.add_argument('--scaler_Y', type=int, default=100,
                        help='scaler of fine-grained flows')
    parser.add_argument('--c_map_shape', type= int, default= 16,
                        choices=[16, 32])
    parser.add_argument('--f_map_shape', type= int, default= 64,
                        choices=[64, 128])
    parser.add_argument('--ext_shape', type= int, default= 5)
    parser.add_argument('--dataset', type=str, default='XiAn', choices= ['XiAn', 'ChengDu','BeiJing'],
                        help='dataset name')
    parser.add_argument('--sub_region', type= int, default=8,
                        help= 'sub regions number H and W in the paper')


    # UrbanPy parameters
    parser.add_argument('--n_residuals', type=str, default='16,16',
                        help='number of residual units')
    parser.add_argument('--n_res_propnet', type=int, default=1,
                        help='number of res_layer in proposal net')
    parser.add_argument('--from_reso', type=int, choices=[8, 16, 32, 64, 128], default= 32,
                        help='coarse-grained input resolution')
    parser.add_argument('--to_reso', type=int, choices=[16, 32, 64, 128], default= 128,
                        help='fine-grained input resolution')
    parser.add_argument('--islocal', action='store_true', default = True,
                        help='whether to use external factors')
    parser.add_argument('--loss_weights', type=str, default='1,1,1', 
                        help='weights on multiscale loss')
    parser.add_argument('--alpha', type=float, default=1e-2,
                        help='alpha')
    parser.add_argument('--compress', type=int, default=2,
                        help='compress channels')
    parser.add_argument('--coef', type=float, default=0.5,
                        help='the weight of proposal net')
    parser.add_argument('--save_diff', default=False, action='store_true',
                        help='save differences')
    
    #STDA parameters
    parser.add_argument('--seq_len', type=int, default=4,
                        help='the length of historic coarse flow  map: L in paper')
    parser.add_argument('--fraction', type=int, default=5,
                        help='days for few-sample adaptation')
    parser.add_argument('--transfer_method', type=str, default='STDA',
                        help='chose transfer method to use',
                        choices=['STDA'])
    
    
    #data args
    parser.add_argument('--datapath', default='./datasets', type=str, help='datapath')
    parser.add_argument('--rec_cities', nargs='+', default=['XiAn','ChengDu'], help='List of resource cities')
    parser.add_argument('--tar_city', nargs='+', default=['BeiJing'], help='List of target city')# XiAn  BeiJing ChengDu

    #task args
    parser.add_argument('--source_spt_size', type=int, default=4,help='support size of source cities')
    parser.add_argument('--source_qry_size', type=int, default=4,help='query size of source cities')
    parser.add_argument('--task_num', type=int, default=2,help='num of task')


    # train args
    #元训练
    parser.add_argument('--meta_lr', default=1.1, type=float,help='gamma in paper')
    parser.add_argument('--source_epochs', default=30, type=int,help='epochs of meta-training') 
    parser.add_argument('--source_train_update_steps', default=10, type=int,help='update steps for each task') 
    parser.add_argument('--source_lr', default=1e-3, type=float,help='rho in paper')
    parser.add_argument('--beta', default=0.1, type=float,help='beta in paper')

    #微调
    parser.add_argument('--target_epochs', default=100, type=int,help='epochs of few-sample adaptation')
    parser.add_argument('--target_lr', default=1e-3, type=float,help='eta in paper')




 

    
    opt = parser.parse_args()
    if opt.dataset == 'BeiJing':
        opt.width, opt.height = 32, 32
        opt.scaler_dict = {32:1, 64:1, 128:1}
        opt.from_reso = 32
        opt.to_reso = 128
    else:
        opt.width, opt.height = 16, 16
        opt.scaler_dict = {16:1, 32:1, 64:1}
        opt.from_reso = 16
        opt.to_reso = 64    
    opt.c_map_shape = opt.width
    opt.f_map_shape = opt.width * opt.scale_factor
                    # {32:1500, 64:300, 128:100}
    opt.n_residuals = [int(_) for _ in opt.n_residuals.split(',')]
    opt.loss_weights = [float(_) for _ in opt.loss_weights.split(',')]
    opt.N = int(np.log(opt.to_reso / opt.from_reso)/np.log(2))

    opt.n_residuals = opt.n_residuals[:opt.N]
    assert opt.from_reso < opt.to_reso, 'invalid resolution, from {} to {}'.format(opt.from_reso, opt.to_reso)
    opt.scales = [opt.from_reso*2**i for i in range(opt.N+1)]
    opt.scalers= [opt.scaler_dict[key] for key in opt.scales]
    return opt
