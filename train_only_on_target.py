import os
device = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = device
import warnings
warnings.filterwarnings("ignore")
import time
import torch
import torch.nn.functional as F
from model.modules.ODE import *
from src.metrics import get_MSE, get_MAE, get_MAPE
from src.utils import print_model_parm_nums,get_dataloader_st
from src.args import get_args
from src.urbanpy_train import UrbanPy_train
from model.STDA import STDA
from model.CUFAR import CUFAR
from model.UrbanFM import UrbanFM
from model.DeepLGR import DeepLGR
from model.UrbanODE import UrbanODE
from model.FODE import FODE
import numpy as np

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('LSTM') == -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

torch.cuda.empty_cache()
args = get_args()
if args.model == 'UrbanPy':
    UrbanPy_train()
else:
    save_path = 'base_model/{}-{}-{}-{}'.format(args.model,
                                                            args.dataset,
                                                            args.fraction,args.seq_len)
    torch.manual_seed(args.seed)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "args.txt"), mode="a", encoding="utf-8") as f:
        f.write("{}".format(args).replace(', ', ',\n'))

    print("mk dir {}".format(save_path))
    os.makedirs(save_path, exist_ok=True)
    print('device:', device)
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def get_learning_rate(optimizer):
        lr=[]
        for param_group in optimizer.param_groups:
            lr +=[ param_group['lr'] ]
        return lr[-1]

    def choose_model():
        if args.model == 'CUFAR':
            model = CUFAR(height=args.height, width=args.width, use_exf=args.use_exf,
                        scale_factor=args.scale_factor, channels=args.n_channels, 
                        sub_region= args.sub_region, 
                        scaler_X=args.scaler_X, scaler_Y=args.scaler_Y, args= args)
        elif args.model == 'UrbanFM':
            model = UrbanFM(in_channels=1, out_channels=1, n_residual_blocks=1,
                        base_channels= args.n_channels, img_width= args.width, 
                        img_height= args.height, ext_flag= args.use_exf, 
                        scaler_X=args.scaler_X, scaler_Y=args.scaler_Y)
            model.apply(weights_init_normal)
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=5.0)
        elif args.model == 'FODE':
            model = FODE(in_channels=1, out_channels=1, n_residual_blocks=16,
                        base_channels= args.n_channels, img_width= args.width, 
                        img_height= args.height, ext_flag= args.use_exf, 
                        scaler_X=args.scaler_X, scaler_Y=args.scaler_Y)
        elif args.model == 'UrbanODE':
            model = UrbanODE(in_channels=1, out_channels=1, n_residual_blocks=16,
                        base_channels= args.n_channels, img_width= args.width, 
                        img_height= args.height, ext_flag= args.use_exf, 
                        scaler_X=args.scaler_X, scaler_Y=args.scaler_Y)
        elif args.model == 'DeepLGR':
            model = DeepLGR(in_channels=1, out_channels=1, n_residual_blocks=12,
                        base_channels= args.n_channels, img_width= args.width, 
                        img_height= args.height, ext_flag=  args.use_exf, predictor='td',
                        scaler_X=args.scaler_X, scaler_Y=args.scaler_Y)
        elif args.model == 'STDA':
            model = STDA(scale_factor=args.scale_factor, channels=args.n_channels, 
                        sub_region= args.sub_region, 
                        scaler_X=args.scaler_X, scaler_Y=args.scaler_Y, args= args)

        return model

    def load_model():
        load_path = '{}/best_epoch.pt'.format(save_path)
        print("load from {}".format(load_path))
        model_state_dict = torch.load(load_path)["model_state_dict"]

        model = choose_model()
        model.load_state_dict(model_state_dict)
        if cuda:
            model = model.cuda()
            return model
        else:
            return model

    criterion = F.mse_loss
    total_datapath = 'datasets'
    total_mses = [np.inf]
    best_epoch = 0

    start_time = time.time()

    print('='*15,'Start to train {}','='*15)

    model = choose_model()
    if cuda:
        model = model.cuda()

    print_model_parm_nums(model, args.model)    

    optimizer = torch.optim.Adam(
        model.parameters(), lr= args.lr, betas=(args.b1, args.b1))
    
    datapath = os.path.join(total_datapath, args.dataset)
    train_ds = get_dataloader_st(
    datapath, args.scaler_X, args.scaler_Y, args.batch_size,seq_len = args.seq_len,fraction = args.fraction, mode = 'train')
    valid_task = get_dataloader_st(
    datapath, args.scaler_X, args.scaler_Y, 8,seq_len = args.seq_len, fraction = None, mode = 'valid')
    test_ds = get_dataloader_st(
    datapath, args.scaler_X, args.scaler_Y, 8,seq_len = args.seq_len, fraction = None, mode = 'test')

    earlystop_count = 0
    for epoch in range(0, args.n_epochs):
        epoch_start_time = time.time()
        earlystop_count+=1
        train_loss = 0
        # training phase
        for i, (c_map, exf, f_map) in enumerate(train_ds):
            model.train()
            optimizer.zero_grad()
            if args.model == 'STDA':
                pred_f_map,region_x, region_y = model(c_map) 
                loss = criterion(pred_f_map * args.scaler_Y, f_map * args.scaler_Y) 
            elif args.model == 'CUFAR':
                pred_f_map,region_x, region_y = model(c_map, exf) 
                loss = criterion(pred_f_map * args.scaler_Y, f_map * args.scaler_Y) 
            elif args.model == 'UrbanFM':
                pred_f_map,region_x, region_y = model(c_map, exf) 
                loss = criterion(pred_f_map * args.scaler_Y, f_map * args.scaler_Y) 
            else:
                pred_f_map = model(c_map, exf) 
                loss = criterion(pred_f_map * args.scaler_Y, f_map * args.scaler_Y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(c_map)
        train_loss /= len(train_ds.dataset)

        # validating phase
        model.eval()
        val_mse, mse = 0, 0
        for j, (c_map, exf, f_map) in enumerate(valid_task):
            if args.model == 'STDA':
                pred_f_map,region_x, region_y = model(c_map) 
            elif args.model == 'CUFAR':
                pred_f_map,region_x, region_y = model(c_map, exf) 
            elif args.model == 'UrbanFM':
                pred_f_map,region_x, region_y = model(c_map, exf) 
            else:
                pred_f_map = model(c_map, exf)

            pred = pred_f_map.cpu().detach().numpy() * args.scaler_Y
            real = f_map.cpu().detach().numpy() * args.scaler_Y
            mse += get_MSE(pred=pred, real=real) * len(c_map)
        val_mse = mse / len(valid_task.dataset)

        if val_mse < np.min(total_mses):
            earlystop_count = 0
            state = {'model_state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            best_epoch = epoch
            torch.save(state, '{}/best_epoch.pt'.format(save_path))
        total_mses.append(val_mse)

        log = ('|Epoch:{}|Loss:{:.3f}|Val_MSE\t{:.3f}|Time_Cost:{:.2f}|Best_Epoch:{}|lr:{}'.format( 
                    epoch, train_loss, 
                    total_mses[-1],
                    time.time() - epoch_start_time, 
                    best_epoch, 
                    get_learning_rate(optimizer)))
        print(log)
        f = open('{}/train_process.txt'.format(save_path), 'a')
        f.write(log+'\n')
        if earlystop_count == 35:
            break
    model = load_model()
    model.eval()

    total_mse, total_mae, total_mape = 0, 0, 0
    for i, (c_map, exf, f_map) in enumerate(test_ds):
        if args.model == 'STDA':
            pred_f_map,region_x, region_y = model(c_map)
        elif args.model == 'CUFAR':
            pred_f_map,region_x, region_y = model(c_map, exf)  
        elif args.model == 'UrbanFM':
            pred_f_map,region_x, region_y = model(c_map, exf)
        else:
            pred_f_map = model(c_map, exf)

        pred = pred_f_map.cpu().detach().numpy() * args.scaler_Y
        real = f_map.cpu().detach().numpy() * args.scaler_Y 



        total_mse += get_MSE(pred=pred, real=real) * len(c_map)
        total_mae += get_MAE(pred=pred, real=real) * len(c_map)
        total_mape += get_MAPE(pred=pred, real=real) * len(c_map)


    mse = total_mse / len(test_ds.dataset)
    mae = total_mae / len(test_ds.dataset)
    mape = total_mape / len(test_ds.dataset)

    log = ('test: RMSE={:.6f}, MAE={:.6f}, MAPE={:.6f}'.format( np.sqrt(mse), mae, mape))
    f = open('{}/test_results.txt'.format(save_path), 'a')
    f.write(log+'\n')
    f.close()
    print(log)
    print('*' * 64)

log = (
    f'Total running time: {(time.time()-start_time)//60:.0f}mins {(time.time()-start_time)%60:.0f}s')
print(log)
f = open('{}/test_results.txt'.format(save_path), 'a')
f.write(log+'\n')
f.close()
print('*' * 64)
