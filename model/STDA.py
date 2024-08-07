import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import sys
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.meta_cin = 1
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.meta_dim = 16*16
        self.in_channels = self.input_dim + self.hidden_dim
        self.out_channels = 4 * self.hidden_dim
        self.stride = 1

        self.d = self.meta_dim
        self.w1_linear = nn.Linear(self.meta_dim, self.in_channels * self.d)
        self.w2_linear = nn.Linear(self.d, self.out_channels * self.kernel_size[0] * self.kernel_size[1])
        self.b_linear = nn.Linear(self.meta_dim, self.out_channels)

        self.p_conv = nn.Conv2d(self.meta_cin, 2*kernel_size[0]*kernel_size[1], kernel_size=3, padding=1, stride=1)
        nn.init.constant_(self.p_conv.weight, 0)
        self.zero_padding = nn.ZeroPad2d(self.padding)

    def forward(self, input_tensor, cur_state, meta_info,meta_info_offset):
        b, c_in, H, W = input_tensor.shape

        # [B, 1, d_mi] -> [B, d_mi] -> [B, C_in*d] -> [B, C_in, d] -> [B, C_in, C_out*kernel_size*kernel_size] -> [B, C_out, C_in, kernel_size,kernel_size] -> [B*C_out, C_in, 1, kernel_size]
        meta_info = torch.reshape(meta_info, (-1, self.meta_dim))#  [B, d_mi]
        w_meta = self.w1_linear(meta_info)#  [B, C_in*d]
        w_meta = torch.reshape(w_meta, (-1, self.in_channels, self.d))# [B, C_in, d]
        w_meta = self.w2_linear(w_meta)#[B, C_in, C_out*kernel_size*kernel_size]
        w_meta = torch.reshape(w_meta, (-1, self.in_channels, self.out_channels, self.kernel_size[0],self.kernel_size[1])).permute(0, 2, 1, 3,4)# [B, C_out, C_in, kernel_size,kernel_size]
        w_meta = torch.reshape(w_meta, (-1, self.in_channels, self.kernel_size[0],self.kernel_size[1]))#[B*C_out, C_in, 1, kernel_size]
        b_meta = self.b_linear(meta_info).view(-1)

        h_cur, c_cur = cur_state
        x = torch.cat([input_tensor, h_cur], dim=1)  
        offset = self.p_conv(meta_info_offset)
        dtype = offset.data.type()
        ks = self.kernel_size[0]
        N = offset.size(1) // 2
        if self.padding:
            x = self.zero_padding(x)
 
        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)
  
        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
 
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
 
        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)
 
        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
 
        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
 
        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt
 
 
        x_offset = self._reshape_x_offset(x_offset, ks)
        x_offset = x_offset.reshape(-1, x_offset.shape[2], x_offset.shape[3]).unsqueeze(0)

        combined_conv = F.conv2d(x_offset, weight=w_meta, bias=b_meta, groups=b,stride = 3)

        combined_conv = combined_conv.squeeze(0).reshape(b, -1, H, W)

        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.w1_linear.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.w1_linear.weight.device))
    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size[0]-1)//2, (self.kernel_size[0]-1)//2+1),
            torch.arange(-(self.kernel_size[0]-1)//2, (self.kernel_size[0]-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)
 
        return p_n
 
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
 
        return p_0
    
    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)
 
        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p
 
    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)
 
        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
 
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
 
        return x_offset
 
    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)
 
        return x_offset



class ConvLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, meta_info, meta_info_offset,hidden_state=None):

        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c], meta_info = meta_info,meta_info_offset = meta_info_offset)
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class regionalFI_model(nn.Module):
    def __init__(self, n_channel, scale_factor, in_channel, kernel_size, padding, groups):
        super(regionalFI_model, self).__init__()
        self.n_channels = n_channel
        self.scale_factor = scale_factor
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.groups = groups
        self.conv1 = nn.Conv2d(self.in_channel, self.n_channels, self.kernel_size, 1, padding, 
                                    groups= self.groups)
        self.relu = nn.ReLU(inplace= True)
        self.conv2 = nn.Conv2d(self.n_channels, self.scale_factor ** 2 * self.n_channels, 
                                    3, 1, 1, groups= self.groups)
        self.pixelshuffle = nn.PixelShuffle(upscale_factor= self.scale_factor)

        self.ccn_layer = CCN_layer(self.n_channels, 0.3)

    def forward(self, x,ssta,tsta):

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.pixelshuffle(x))
        x = self.ccn_layer(x,ssta,tsta)
        return x

class CCN(nn.Module): #cross-city normalization

    def __init__(self, planes):
        super(CCN, self).__init__()

        self.IN = nn.InstanceNorm2d(planes, affine=False)
        self.BN = nn.BatchNorm2d(planes, affine=False)
        self.alpha = nn.Parameter(torch.FloatTensor([0.0]), requires_grad=True)
        self.avg = torch.nn.AdaptiveAvgPool2d((16,16))
        self.conv = nn.Conv2d(2, 1, 1, 1)
        self.linear = nn.Linear(16*16, 1)
    def forward(self, x,ssta,tsta):
        src_dip = self.avg(ssta.unsqueeze(0))
        tgt_dip = self.avg(tsta.unsqueeze(0))
        tgt_dip = tgt_dip.float().to(src_dip.device)
        offset = self.conv(torch.cat([src_dip,tgt_dip],dim=0))
        t=F.relu(self.linear(offset.flatten()))
        out_in = self.IN(x)
        out_bn = self.BN(x)
        out = t * out_in + (1 - t) * out_bn
        return out
    

class CCN_layer(nn.Module):
    def __init__(self, n_channel, drop_rate):
        super(CCN_layer, self).__init__()
        self.norm = CCN(n_channel)
        self.drop_rate = nn.Dropout(drop_rate)

    def forward(self, x,ssta,tsta):
        x = self.norm(x,ssta,tsta)
        x = self.drop_rate(x)
        return x


class STDA(nn.Module):
    def __init__(self, scale_factor=4, 
                    channels=128, sub_region = 4, scaler_X=1, scaler_Y=1, args= None):
        super(STDA, self).__init__()


        self.scale_factor = scale_factor
        self.out_channel = 1
        self.sub_region = sub_region
        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y
        self.args = args
        self.kernel_size = 3
        self.in_channel=1
        self.meta_dim = 64
        self.C_1 = 1
        self.C_2 = channels
        self.meta_conv = nn.Conv2d(in_channels=self.in_channel,
                                    out_channels= self.in_channel,
                                    kernel_size=3,
                                    padding=1)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 16))
        self.mu_T = torch.tensor(np.load('./datasets/statistics/' + args.tar_city[0]+ str(args.fraction)+ '.npy')).to(self.meta_conv.weight.device)

        self.inter_region_model = ConvLSTM(1, self.C_1, (3,3), 1, True, True, False)
        self.intra_region_model = regionalFI_model(self.C_2 * (sub_region ** 2), 
                            self.scale_factor, 1 * (sub_region **2), 3, 1, sub_region **2)

        self.relu = nn.ReLU()

        self.decoder_conv = nn.Sequential(
            nn.Conv2d(self.C_2, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.out_channel, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        
    def normalization(self, x, save_x): #N^2-normalzation
        w = (nn.AvgPool2d(self.scale_factor)(x)) * self.scale_factor ** 2
        w = nn.Upsample(scale_factor= self.scale_factor, mode='nearest')(w)
        w = torch.divide(x, w + 1e-7)
        up_c = nn.Upsample(scale_factor=self.scale_factor, mode='nearest')(save_x)
        x = torch.multiply(w, up_c)
        return x
    

    def forward(self, x):
        save_x = x[:,-1,:,:]
        mu_k = torch.mean(save_x, dim = (0,1))
 
        #generate Z
        meta_info = self.meta_conv(save_x)# Z in paper
        meta_info_offset = meta_info
        meta_info = self.adaptive_pool(meta_info)

        ##ST-learner with adaptive kernel
        #city_level spatio-temporal learning
        _, last_states = self.inter_region_model(x, meta_info,meta_info_offset)
        hidden_state = last_states[0][0]  # H_t in paper

        #region-level fine-grained inference and cross-city normalization
        patchs_c = rearrange(save_x + hidden_state, 'b c (ph h) (pw w) -> b (ph pw c) h w', 
                                    ph= self.sub_region, pw= self.sub_region)
    
        F_norm = self.intra_region_model(patchs_c,ssta = mu_k, tsta = self.mu_T)

        #ST-decoder
        F_norm_combine = rearrange(F_norm , 'b (ph pw c) h w -> b c (ph h) (pw w)', 
                                    ph= self.sub_region, pw= self.sub_region)
        x = torch.relu(F_norm_combine)
        x = self.decoder_conv(x)
        x = self.normalization(x, save_x * self.scaler_X / self.scaler_Y)

        return x, patchs_c.reshape(-1, patchs_c.shape[2], patchs_c.shape[3]), F_norm.reshape(-1, F_norm.shape[2], F_norm.shape[3]) 
                    #patchs_c.reshape(-1, patchs_c.shape[2], patchs_c.shape[3]), F_norm.reshape(-1, F_norm.shape[2], F_norm.shape[3])  is  used for ragional structure alignment

    def _ini_hidden(self, bs, c, w, h):
        return torch.ones((bs, c, w, h)).to(self.device)
