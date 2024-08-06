import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import argparse
import time
from meta import STMAML
from tqdm import tqdm
from src.utils import get_task,get_dataloader_st
import itertools
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from src.args import get_args
args = get_args()

if __name__ == '__main__':

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        print("INFO: GPU")
    else:
        args.device = torch.device('cpu')
        print("INFO: CPU")

    torch.manual_seed(7)

    datapath_target = os.path.join('datasets',args.tar_city[0])
    train_dataloadert = get_dataloader_st(
    datapath_target, args.scaler_X, args.scaler_Y, args.source_spt_size,\
        seq_len = args.seq_len,fraction = args.fraction, mode = 'train')
    dataloder_target = itertools.cycle(iter(train_dataloadert))
    model = STMAML(args, model_name=args.model).to(device=args.device)
    

    loss_criterion = nn.MSELoss()

    spt_idx = 0
    qry_idx = 0
    for epoch in range(args.source_epochs):
        
        start_time = time.time()
        x_spt, y_spt,ext_spt, x_qry, y_qry, ext_qry = get_task( args.datapath , args.rec_cities, args.scaler_X,args.scaler_Y,\
                                                 args.source_spt_size, args.source_qry_size, args.seq_len, args.tar_city[0],\
                                                    args.fraction,args.transfer_method,spt_idx, qry_idx)
        spt_idx+=args.source_spt_size
        qry_idx+=args.source_qry_size
        x_sptt,ext_sptt, y_sptt = next(dataloder_target)


        if x_sptt.shape[0] != args.source_spt_size:
            x_sptt,ext_sptt, y_sptt = next(dataloder_target)

        if args.transfer_method == 'MAML':
            loss = model.meta_train_MAML(x_spt, y_spt, ext_spt, x_qry, y_qry, ext_qry)
        elif args.transfer_method == 'STDA':
            loss = model.meta_train_STDA(x_spt, y_spt, ext_spt, x_qry, y_qry, ext_qry,x_sptt, y_sptt,ext_sptt)

        end_time = time.time()
        if epoch % 1 == 0:
            print("[Source Train] epoch #{}/{}: loss is {}, training time is {}".format(epoch+1, args.source_epochs, loss, end_time-start_time))

    print("Source dataset meta-train finish.")
