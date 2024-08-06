import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.linear_model import Lasso
from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance
import random
import sys
def print_model_parm_nums(model, str):
    total_num = sum([param.nelement() for param in model.parameters()])
    print('{} params: {}'.format(str, total_num))
    return total_num

def get_gt_densities(flows, opt):
    inp = flows[0] * opt.scaler_dict[opt.scales[0]]
    scale0 = opt.scales[0]
    out, masks = [], []

    for i, f in enumerate(flows[1:]):
        scale_ = opt.scales[i+1]
        inp_ = F.upsample(inp, scale_factor=scale_//scale0)
        masks.append(inp_ != 0)
        f0 = inp_ + 1e-9
        f_ = f*opt.scaler_dict[opt.scales[i+1]]
        out.append(f_/f0)
    return out, masks

def get_dataloader(datapath, scaler_X, scaler_Y, batch_size, mode='train'):
    datapath = os.path.join(datapath, mode)
    print(datapath)
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    X = Tensor(np.expand_dims(np.load(os.path.join(datapath, 'X.npy')), 1)) / scaler_X
    # print(X.shape)
    Y = Tensor(np.expand_dims(np.load(os.path.join(datapath, 'Y.npy')), 1))/ scaler_Y
    ext = Tensor(np.load(os.path.join(datapath, 'ext.npy')))

    assert len(X) == len(Y)
    print('# {} samples: {}'.format(mode, len(X)))
    print(X.shape)
    data = torch.utils.data.TensorDataset(X, ext, Y)
    if mode == 'train':
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    else:
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    return dataloader

def get_lapprob_dataloader(datapath, args, batch_size=2, mode='train',fraction=1):

    datapath = os.path.join(datapath, args.dataset)
    datapath = os.path.join(datapath, mode)
    
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    Xs = list()
    if "XiAn" in datapath or "ChengDu" in datapath:
        for scale in args.scales:
            if scale == 16:
                Xs.append(Tensor(np.expand_dims(np.load(os.path.join(datapath, 'X.npy')), 1)) / args.scaler_dict[scale])
            elif scale == 32:
                Xs.append(Tensor(np.expand_dims(np.load(os.path.join(datapath, 'X_%d.npy'%scale)), 1)) / args.scaler_dict[scale])
            elif scale == 64:
                Xs.append(Tensor(np.expand_dims(np.load(os.path.join(datapath, 'Y.npy')), 1)) / args.scaler_dict[scale])
    else:
        for scale in args.scales:
            if scale == 16:
                Xs.append(Tensor(np.expand_dims(np.load(os.path.join(datapath, 'X_%d.npy'%scale)), 1)) / args.scaler_dict[scale])
            elif scale == 32:
                Xs.append(Tensor(np.expand_dims(np.load(os.path.join(datapath, 'X.npy')), 1)) / args.scaler_dict[scale])
            elif scale == 64:
                Xs.append(Tensor(np.expand_dims(np.load(os.path.join(datapath, 'X_%d.npy'%scale)), 1)) / args.scaler_dict[scale])
            elif scale == 128:
                Xs.append(Tensor(np.expand_dims(np.load(os.path.join(datapath, 'Y.npy')), 1)) / args.scaler_dict[scale])
    ext = Tensor(np.load(os.path.join(datapath, 'ext.npy')))
    Xs.append(ext)
    if fraction != None:
        if mode == 'train':
            if "XiAn" in datapath or "ChengDu" in datapath:
                for i, arr in enumerate(Xs):
                    Xs[i] = arr[:fraction*96]
            else:#Beijing
                for i, arr in enumerate(Xs):
 
                    Xs[i] = arr[:fraction*24]


    data = torch.utils.data.TensorDataset(*Xs)
    for scale in args.scales:
        if mode == 'train':
            dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
        else:
            dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False)
    return dataloader

def get_dataloader_st(datapath, scaler_X, scaler_Y, batch_size, seq_len, fraction, mode='train'):
    #fraction代表切割天数
    datapath = os.path.join(datapath, mode)
    print(datapath)
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    X_data = []
    Y_data = []
    ext_data = []
    # 加载数据
    if "XiAn" in datapath or "ChengDu" or "P1" in datapath :
        print(1)
        X = np.expand_dims(np.load(os.path.join(datapath, 'X.npy')),1) / scaler_X
        Y = np.expand_dims(np.load(os.path.join(datapath, 'Y.npy')),1) / scaler_Y
        ext = np.load(os.path.join(datapath, 'ext_best.npy'))
    else:
        
        X = np.load(os.path.join(datapath, 'X.npy')) / scaler_X
        Y = np.load(os.path.join(datapath, 'Y.npy')) / scaler_Y
        ext = np.load(os.path.join(datapath, 'ext_best.npy'))


    #构造小样本
    if fraction != None:
        if mode == 'train':
            if "XiAn" in datapath or "ChengDu" in datapath:
                X = X[:fraction*96]
                Y = Y[:fraction*96]
                ext = ext[:fraction*96]

            else:#北京
                X = X[:fraction*24]
                Y = Y[:fraction*24]
                ext = ext[:fraction*24]

    assert len(X) == len(Y)
    print('# {} samples: {}'.format(mode, len(X)))
    print(X.shape)

    # 创建滑动窗口样本
    for i in range(len(X) - seq_len):
        X_window = X[i:i+seq_len]     # 当前时刻与先前 seq_len 个时刻的图像
        Y_window = Y[i+seq_len-1]     # 当前时刻的图像
        ext_window = ext[i:i+seq_len]

        X_data.append(X_window)
        Y_data.append(Y_window)
        ext_data.append(ext_window)

    # 将列表转换为张量
    X_data = Tensor(np.array(X_data))
    Y_data = Tensor(np.array(Y_data))
    ext_data = Tensor(np.array(ext_data))

    print(X_data.shape)
    print(Y_data.shape)

    # 创建数据集和数据加载器
    data = torch.utils.data.TensorDataset(X_data, ext_data, Y_data)
    if mode == 'train':
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    else:
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    return dataloader



def get_dataloader_st_baseline(datapath, scaler_X, scaler_Y, batch_size, seq_len, fraction, mode='train'):
    #fraction代表切割天数
    datapath = os.path.join(datapath, mode)
    print(datapath)
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    X_data = []
    Y_data = []
    ext_data = []
    # 加载数据
    if "XiAn" in datapath or "ChengDu" or "BeiJing" in datapath :
        print(1)
        X = np.expand_dims(np.load(os.path.join(datapath, 'X.npy')),1) / scaler_X
        Y = np.expand_dims(np.load(os.path.join(datapath, 'Y.npy')),1) / scaler_Y
        ext = np.load(os.path.join(datapath, 'ext.npy'))
    else:
        
        X = np.load(os.path.join(datapath, 'X.npy')) / scaler_X
        Y = np.load(os.path.join(datapath, 'Y.npy')) / scaler_Y
        ext = np.load(os.path.join(datapath, 'ext.npy'))
    #构造小样本
    if fraction != None:
        if mode == 'train':
            if "XiAn" in datapath or "ChengDu" in datapath:
                X = X[:fraction*96]
                Y = Y[:fraction*96]
                ext = ext[:fraction*96]
            else:#北京
                X = X[:fraction*24]
                Y = Y[:fraction*24]
                ext = ext[:fraction*24]


    assert len(X) == len(Y)
    print('# {} samples: {}'.format(mode, len(X)))
    print(X.shape)

    # 创建滑动窗口样本
    for i in range(len(X) - seq_len):
        X_window = X[i:i+seq_len]     # 当前时刻与先前 seq_len 个时刻的图像
        Y_window = Y[i+seq_len-1]     # 当前时刻的图像
        ext_window = ext[i:i+seq_len]

        X_data.append(X_window)
        Y_data.append(Y_window)
        ext_data.append(ext_window)

    # 将列表转换为张量
    X_data = Tensor(np.array(X_data))
    Y_data = Tensor(np.array(Y_data))
    ext_data = Tensor(np.array(ext_data))

    print(X_data.shape)
    print(Y_data.shape)

    # 创建数据集和数据加载器
    data = torch.utils.data.TensorDataset(X_data, ext_data, Y_data)
    if mode == 'train':
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    else:
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    return dataloader


def get_task(datapath, cities, scaler_X, scaler_Y, spts, qrys, seq_len, target_city, fraction,transfer_mode, spt_idx,qry_idx): #只生成源城市的数据
    # datapath: 不包含城市名
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    cities_current = cities

    X_cities_support = {} #储存城市suopport采样样本
    Y_cities_support = {}
    ext_cities_support = {}

    X_cities_query = {}#储存城市quert采样样本
    Y_cities_query = {}
    ext_cities_query = {}

#####计算目标城市均值分布
    datapath_tc = os.path.join(datapath ,target_city, 'train') 
    # print(datapath_current)
    X_tc = np.expand_dims(np.load(os.path.join(datapath_tc, 'X.npy')), 1) / scaler_X

    if "XiAn" in datapath or "ChengDu" in datapath:
        X_tc = X_tc[:fraction*96]

    else:#北京
        X_tc = X_tc[:fraction*24]

    mean_target_X = np.mean(X_tc)





    # random.seed(random_state)
####### 生成元训练数据
    for i, city in enumerate(cities_current):# 构建元任务的support set
        datapath_current = os.path.join(datapath ,city, 'train') 
        # print(datapath_current)
        X = np.expand_dims(np.load(os.path.join(datapath_current, 'X.npy')), 1) / scaler_X
        # print(X.shape)
        Y = np.expand_dims(np.load(os.path.join(datapath_current, 'Y.npy')), 1) / scaler_Y
        ext = np.load(os.path.join(datapath_current, 'ext.npy'))
        assert len(X) == len(Y)



    ##计算最佳样本索引
        sample_dict = {}
        if transfer_mode == 'STDA':
            for i in range(X.shape[0] - seq_len):
                x_curr = X[i].flatten()
                statistic = np.count_nonzero(x_curr)
                sample_dict[i] = statistic
            sample_index = sorted(sample_dict, key=sample_dict.get)[:int((X.shape[0] - seq_len)*0.5)]

        else:
            sample_index = [i for i in range(1, X.shape[0] - seq_len)]

    #构造小样本
        X= Tensor(X)
        Y = Tensor(Y)
        ext = Tensor(ext)

        # idx = random.sample(sample_index, spts)
        if  spt_idx+spts > len(sample_index):
            spt_idx=0
        idx = sample_index[spt_idx:spt_idx+spts]

        # 保存样本  
        X_city = []
        Y_city = []
        ext_city = []
        for j in idx:
            # 取出当前时间步以及前面的 n 个时间步作为一个样本
            X_sample = X[j:j+seq_len]
            Y_sample = Y[j+seq_len-1]
            ext_sample = ext[j:j+seq_len]
           
            X_city.append(X_sample)
            Y_city.append(Y_sample)
            ext_city.append(ext_sample)

        X_city = torch.stack(X_city, dim= 0)
        Y_city = torch.stack(Y_city, dim= 0)
        ext_city = torch.stack(ext_city, dim= 0)
       
        X_cities_support[city] = X_city
        Y_cities_support[city] = Y_city
        ext_cities_support[city] = ext_city

    for i, city in enumerate(cities_current):# 构建元任务的query set
        datapath_current = os.path.join(datapath,city, 'valid') 
        X = np.expand_dims(np.load(os.path.join(datapath_current, 'X.npy')), 1) / scaler_X
        # print(X.shape)
        Y = np.expand_dims(np.load(os.path.join(datapath_current, 'Y.npy')), 1) / scaler_Y
        ext = np.load(os.path.join(datapath_current, 'ext.npy'))
        assert len(X) == len(Y)


    ##计算最佳样本索引
        count_t = 0
        count_f = 0
        sample_dict = {}
        if transfer_mode == 'STDA':
            sample_index = []
            for i in range(X.shape[0] - seq_len):
                x_curr = X[i].flatten()
                statistic = np.count_nonzero(x_curr)
                sample_dict[i] = statistic

            sample_index = sorted(sample_dict, key=sample_dict.get)[:int((X.shape[0] - seq_len)*0.5)]

        else:
            sample_index = [i for i in range(1, X.shape[0] - seq_len)]

    #构造小样本
        X= Tensor(X)
        Y = Tensor(Y)
        ext = Tensor(ext)
        # idx = random.sample(sample_index, qrys)
        if  qry_idx+qrys > len(sample_index):
            qry_idx=0
        idx = sample_index[qry_idx:qry_idx+qrys]


        # 保存样本  
        X_city = []
        Y_city = []
        ext_city = []
        for j in idx:
            # 取出当前时间步以及前面的 n 个时间步作为一个样本
            X_sample = X[j:j+seq_len]
            Y_sample = Y[j+seq_len - 1]
            ext_sample = ext[j:j+seq_len]

            X_city.append(X_sample)
            Y_city.append(Y_sample)
            ext_city.append(ext_sample)

        X_city = torch.stack(X_city, dim= 0)
        Y_city = torch.stack(Y_city, dim= 0)
        ext_city = torch.stack(ext_city, dim= 0)
       
        X_cities_query[city] = X_city
        Y_cities_query[city] = Y_city
        ext_cities_query[city] = ext_city



    return X_cities_support, Y_cities_support, ext_cities_support, X_cities_query, Y_cities_query, ext_cities_query
    #返回任务集合



# 定义 MMD 损失函数
def compute_mmd_linear(x, y):
    # 将两个张量展平为二维矩阵
    x_flat = x.view(x.size(0), -1)
    y_flat = y.view(y.size(0), -1)
    
    # 计算每个样本的均值
    x_mean = torch.mean(x_flat, dim=1, keepdim=True)
    y_mean = torch.mean(y_flat, dim=1, keepdim=True)
    
    # 计算两个分布的协方差
    x_cov = torch.matmul(x_flat - x_mean, (x_flat - x_mean).t())
    y_cov = torch.matmul(y_flat - y_mean, (y_flat - y_mean).t())
    
    # 计算 MMD 损失
    mmd_loss = torch.norm(x_cov - y_cov, p='fro')
    
    return mmd_loss

# 计算MMD距离
def compute_mmd_guasssian(x, y, sigma=1.0):
    # 计算内核矩阵
    K_xx = torch.exp(-torch.cdist(x, x, p=2) ** 2 / (2 * sigma ** 2))
    K_yy = torch.exp(-torch.cdist(y, y, p=2) ** 2 / (2 * sigma ** 2))
    K_xy = torch.exp(-torch.cdist(x, y, p=2) ** 2 / (2 * sigma ** 2))

    # 计算MMD
    mmd = K_xx.mean(dim=1) + K_yy.mean(dim=1) - 2 * K_xy.mean(dim=1)
    return mmd





def rsa_loss(region_xs, region_xt, region_ys, region_yt):
    
    hs_x , ht_x = region_xs.shape[1],region_xt.shape[1]
    hs_y , ht_y = region_ys.shape[1],region_yt.shape[1]
    #对齐网格大小
    if ht_x > hs_x:
        region_xt = F.adaptive_avg_pool2d(region_xt,(hs_x,hs_x))
        region_yt = F.adaptive_avg_pool2d(region_yt,(hs_y,hs_y))

    else:
        region_xs = F.adaptive_avg_pool2d(region_xs,(ht_x,ht_x))
        region_ys = F.adaptive_avg_pool2d(region_ys,(ht_y,ht_y))

    #将最后的h， w维平铺
    region_xs = region_xs.reshape(region_xs.shape[0], -1)
    region_xt = region_xt.reshape(region_xt.shape[0], -1)
    region_ys = region_ys.reshape(region_ys.shape[0], -1)
    region_yt = region_yt.reshape(region_yt.shape[0], -1)

    w = compute_mmd_linear(region_xs, region_xt)
    y_diff = compute_mmd_linear(region_ys, region_yt)
    loss = torch.sum(w * y_diff)
    return loss


