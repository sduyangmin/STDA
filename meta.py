import os
import torch
from torch import nn
from torch import optim
import numpy as np
from tqdm import tqdm
from src.utils import print_model_parm_nums,rsa_loss
from model.STDA import STDA
from model.CUFAR import CUFAR
from model.UrbanFM import UrbanFM
import copy

class STMAML(nn.Module):
    """
    MAML-based Few-sample learning architecture 
    """
    def __init__(self, args,model_name='UrbanFM_conv'):
        super(STMAML, self).__init__()
        self.args = args

        self.source_update_lr = args.source_lr #源任务内部学习率
        self.target_update_lr = args.target_lr #源任务内部学习率
        self.meta_lr = args.meta_lr     #调整初始化参数用的学习率


        self.update_step = args.source_train_update_steps

        self.task_num = args.task_num
        self.source_cities = args.rec_cities
        self.model_name = model_name
        self.scaler_Y = args.scaler_Y
        self.scaler_X = args.scaler_X
        if args.model == 'STDA':
            self.model = STDA(
                        scale_factor=args.scale_factor, channels=args.n_channels, 
                        sub_region= args.sub_region, 
                        scaler_X=args.scaler_X, scaler_Y=args.scaler_Y, args= args)

            print("Meta Model: STDA")
        elif args.model == 'CUFAR':
            self.model = CUFAR(height=args.height, width=args.width, use_exf=args.use_exf,
                        scale_factor=args.scale_factor, channels=args.n_channels, 
                        sub_region= args.sub_region, 
                        scaler_X=args.scaler_X, scaler_Y=args.scaler_Y, args= args)
            print("Meta Model: CUFAR")
        elif args.model == 'UrbanFM':
            self.model = UrbanFM(in_channels=1, out_channels=1, n_residual_blocks=1,
                        base_channels= args.n_channels, img_width= args.width, 
                        img_height= args.height, ext_flag= args.use_exf, 
                        scaler_X=args.scaler_X, scaler_Y=args.scaler_Y)
            
            print("Meta Model: UrbanFM")
        self.meta_model_path = 'meta_train_model/{}-{}/{}/{}/{}'.format(self.args.model,self.args.seq_len,
                                                            self.args.tar_city[0],
                                                            self.args.fraction,self.args.transfer_method)
        os.makedirs(self.meta_model_path, exist_ok=True)

        with open(os.path.join(self.meta_model_path, "args.txt"), mode="a", encoding="utf-8") as f:
            f.write("{}".format(args).replace(', ', ',\n'))
        print_model_parm_nums(self.model, model_name)
        print(self.args.transfer_method)

        self.meta_optim = optim.Adam(filter(lambda p: p.requires_grad,self.model.parameters()), lr=self.meta_lr)#定义元学习优化器
        self.loss_criterion = nn.MSELoss()# 定义元学习损失函数
        self.model.train()

    def meta_train_STDA(self, x_spt, y_spt, ext_spt, x_qry, y_qry, ext_qry,x_sptt, y_sptt, ext_sptt): # 更新元学习器参数 返回元学习器在元训练任务的测试集上的损失

        meta_param = self.model.state_dict()
        gradient = {name : 0 for name in meta_param}
        beta = self.args.beta
        for j in self.source_cities:
            learner = copy.deepcopy(self.model) # 第一次copy 用来计算更新初始化参数所需的方向
            optim = torch.optim.Adam(learner.parameters(), lr = self.source_update_lr) # 训练任务的优化器和学习率
            learner.train()

            #train each task for learner
            for k in range(self.update_step):
                if self.args.model == 'STDA':
                    score,region_xs,region_ys = learner(x_spt[j])
                    outt ,region_xt,region_yt = learner(x_sptt)
                else:
                    score,region_xs,region_ys = learner(x_spt[j],ext_spt[j])
                    outt ,region_xt,region_yt = learner(x_sptt, ext_sptt) 

                loss =  (1-beta) * self.loss_criterion(score* self.args.scaler_Y , y_spt[j]* self.args.scaler_Y) \
                        + beta * rsa_loss(region_xs, region_xt, region_ys, region_yt)
        
                optim.zero_grad()
                loss.backward()
                optim.step()            
            
            #Theta = Theta + epsilon * (1/batch size) * sigma(W - Theta), Equation in paper 
            learner_param = learner.state_dict()
            for name in gradient:
                gradient[name] = gradient[name] + (learner_param[name] - meta_param[name])
                
        self.model.load_state_dict(({name : meta_param[name] + self.args.meta_lr * (gradient[name] / self.args.task_num) for name in meta_param}))
        state = {'model_state_dict': self.model.state_dict(), 'optimizer': self.meta_optim.state_dict()}
        torch.save(state, '{}/best_epoch.pt'.format(self.meta_model_path))

        return  loss.detach().cpu().numpy()
    
    def meta_train_MAML(self, x_spt, y_spt, ext_spt, x_qry, y_qry, ext_qry): # 更新元学习器参数 返回元学习器在元训练任务的测试集上的损失


        meta_param = self.model.state_dict()
        gradient = {name : 0 for name in meta_param}
        
        for j in self.source_cities:
            learner = copy.deepcopy(self.model) # 第一次copy 用来计算更新初始化参数所需的方向
            optim = torch.optim.Adam(learner.parameters(), lr = self.source_update_lr) # 训练任务的优化器和学习率
            learner.train()
            
            #train each task for learner
            for k in range(self.update_step):
            
                if self.args.model == 'STDA':
                    score,region_xs,region_ys = learner(x_spt[j])
                else:
                    score,region_xs,region_ys = learner(x_spt[j],ext_spt[j])

                loss = self.loss_criterion(score * self.args.scaler_Y , y_spt[j]* self.args.scaler_Y) 
                    
                optim.zero_grad()
                loss.backward()
                optim.step()            
            
            #Theta = Theta + epsilon * (1/batch size) * sigma(W - Theta), Equation in paper 
            learner_param = learner.state_dict()
            for name in gradient:
                gradient[name] = gradient[name] + (learner_param[name] - meta_param[name])
                
        self.model.load_state_dict(({name : meta_param[name] + self.args.meta_lr * (gradient[name] / self.args.task_num) for name in meta_param}))
        state = {'model_state_dict': self.model.state_dict(), 'optimizer': self.meta_optim.state_dict()}

        torch.save(state, '{}/best_epoch.pt'.format(self.meta_model_path))
        return  loss.detach().cpu().numpy()
    def forward(self, x):
        out = self.model(x)
        return out

