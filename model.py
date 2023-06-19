import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np

def layer_init(layer, std=.1, bias_zero = False):
    torch.nn.init.orthogonal_(layer.weight, std)
    if bias_zero:
        torch.nn.init.constant_(layer.bias, 0.)
    return layer


class ResLayer(nn.Module):

    def __init__(self, in_f, red_l, start_k = 1, mid_k = 3, bias = True, use_relu = True):

        super(ResLayer, self).__init__()
        padding_s = 0 if start_k == 1 else (start_k - 1) //  2
        padding_m = 0 if mid_k == 1 else (mid_k - 1) //  2
        self.use_relu = use_relu
        self.conv1 = layer_init(nn.Conv2d(in_f, red_l, start_k, bias = bias, padding= padding_s))
        self.conv2 = layer_init(nn.Conv2d(red_l, red_l, mid_k, bias = bias, padding = padding_m))
        self.conv3 = layer_init(nn.Conv2d(red_l, in_f, start_k, bias = bias, padding= padding_s))

    def forward(self, x):

        y = self.conv1(x)
        if self.use_relu:
            y = F.relu(y,inplace=True)
        y = self.conv2(y)
        if self.use_relu:
            y = F.relu(y,inplace=True)
        y = self.conv3(y)
        if self.use_relu:
            y = F.relu(y+x,inplace=True)
        return y + x
    
class BoardEmbed(nn.Module):

    def __init__(self):
        super(BoardEmbed, self).__init__()
        self.embedding = nn.Embedding(3,2)
        self.conv1 = layer_init(nn.Conv2d(2,32,1))
    def forward(self, x):
        x = self.embedding(x).view(-1,48,48,2)
        x = x.permute(0,3,1,2)
        x = F.relu(self.conv1(x), inplace=True)
        return x

class PlayerEmbed(nn.Module):
    def __init__(self):
        super(PlayerEmbed, self).__init__()
        self.embedding = nn.Embedding(6,3)
        self.conv1 = layer_init(nn.Conv2d(3,32,1))
    def forward(self, x):
        x = self.embedding(x).view(-1,48,48,3)
        x = x.permute(0,3,1,2)
        x = F.relu(self.conv1(x))
        return x

class ActionSharing(nn.Module):

    def __init__(self, in_f, lc):

        super(ActionSharing, self).__init__()

        self.seq_layers = nn.Sequential(
            *[ResLayer(in_f,in_f,start_k=5,mid_k= 5,bias = False,use_relu=False) for _ in range(lc)]
        )

    def forward(self, x):
        return self.seq_layers(x)

    
class Model(nn.Module):

    def __init__(self, unit_act_sp, factory_act_sp, net_n = 256, res_c = 8, red_l= 32):

        super(Model, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f'Using {self.device} device')

        self.u_act_s = (4,5,19,2)
        self.f_act_s = (3,2)

        self.board_embedding = BoardEmbed()
        self.player_0_embedding = PlayerEmbed()
        self.player_1_embedding = PlayerEmbed()
        self.remain_embedding = nn.Embedding(21,1)

        self.rubble_conv = layer_init(nn.Conv2d(1,32,1))
        self.player_0_conv = layer_init(nn.Conv2d(6,32,1))
        self.player_1_conv = layer_init(nn.Conv2d(6,32,1))

        self.investigate_disc = nn.Sequential(
            *[ResLayer(96,32) for _ in range(2)]
        )

        self.investigate_rubble = nn.Sequential(
            *[ResLayer(32,16) for _ in range(2)]
        )

        self.investigate_player_0 = nn.Sequential(
            *[ResLayer(32,16) for _ in range(4)]
        )

        self.investigate_player_1 = nn.Sequential(
            *[ResLayer(32,16) for _ in range(4)]
        )

        self.add_fc = layer_init(nn.Linear(2,16))

        self.net_n = net_n

        self.result_conv = layer_init(nn.Conv2d(192,net_n,1))

        self.res = nn.Sequential(
            *[ResLayer(net_n,red_l) for _ in range(res_c)]
        )

        self.fc = layer_init(nn.Linear(256,128))

        self.last_fc = layer_init(nn.Linear(144,128))

        self.lstm = nn.LSTM(128,128, batch_first = True)
        for name, param in self.lstm.named_parameters():
            if name.startswith("weight"): 
                nn.init.orthogonal_(param, .1)
            elif name.startswith("bias"):
                nn.init.constant_(param, 0.)

        self.act_u0 = layer_init(nn.Linear(128,48*48*unit_act_sp), std = .01, bias_zero=True)
        self.act_u1 = ActionSharing(30,3)
        self.act_f = layer_init(nn.Linear(128,48*48*factory_act_sp), std = .01, bias_zero=True)
        self.critic = layer_init(nn.Linear(128,1), std = 1, bias_zero=True)

    def forward(self, disc, cont, timestamp, remain, h_s, c_s, validate = False):

        if not validate:

            board, player_0, player_1 = torch.split(disc, (1,1,1), dim = 1)
            disc2 = torch.cat((board, player_1, player_0), dim = 1)
            rubble, player_0, player_1 = torch.split(cont,(1,6,6), dim = 1)
            cont2 = torch.cat((rubble, player_1, player_0), dim = 1)
            
            disc = torch.cat((disc, disc2))
            cont = torch.cat((cont, cont2))
            timestamp = torch.cat((timestamp, timestamp))
            remain = torch.cat((remain, remain))

        board, player_0, player_1 = torch.split(disc,(1,1,1),dim = 1)
        board = self.board_embedding(board)
        player_0 = self.player_0_embedding(player_0)
        player_1 = self.player_1_embedding(player_1)

        disc = torch.cat((board, player_0, player_1), dim = 1)
        disc = self.investigate_disc(disc)

        rubble, player_0, player_1 = torch.split(cont, (1,6,6), dim = 1)
        rubble = F.relu(self.rubble_conv(rubble), inplace = True)
        rubble = self.investigate_rubble(rubble)

        player_0 = F.relu(self.player_0_conv(player_0), inplace=True)
        player_0 = self.investigate_player_0(player_0)
        
        player_1 = F.relu(self.player_1_conv(player_1), inplace=True)
        player_1 = self.investigate_player_1(player_1)

        result = torch.cat((disc, rubble, player_0, player_1), dim = 1)
        result = F.relu(self.result_conv(result), inplace=True)

        result = self.res(result)
        result = nn.AdaptiveMaxPool2d(1)(result).view(-1,256)
        result = F.relu(self.fc(result))

        remain = self.remain_embedding(remain).view(-1,1)
        add = torch.cat((timestamp, remain), dim = 1)
        add = F.relu(self.add_fc(add), inplace=True)

        result = torch.cat((result, add), dim = 1)
        result = self.last_fc(result).view(-1,1,128)

        if not validate:

            h_s0 = h_s[0]
            c_s0 = c_s[0]
            h_s1 = h_s[1]
            c_s1 = c_s[1]

            split_v = result.shape[0] // 2

            result_0, result_1 = torch.split(result,split_v)

            o_0, (h_0, c_0) = self.lstm(result_0, (h_s0, c_s0))
            o_1, (h_1, c_1) = self.lstm(result_1, (h_s1, c_s1))

            return (h_0, c_0), (h_1, c_1)
        
        o, (hs, cs) = self.lstm(result, (h_s, c_s))
        return hs
    
    def get_action(self, disc, cont, timestamp, remain, h_s, c_s, u_masks, f_masks, action = None, validate = False):

        if not validate:
            (next_hs0, next_cs0),(next_hs1, next_cs1) = self.forward(disc, cont, timestamp, remain, h_s, c_s)

            next_hs0 = next_hs0.view(-1,128)
            next_hs1 = next_hs1.view(-1,128)
            next_cs = torch.stack((next_cs0, next_cs1))
            next_hs = torch.cat((next_hs0, next_hs1))

            logi_u = self.act_u0(next_hs).view(-1,30,48,48)
            logi_u = self.act_u1(logi_u).permute(0,2,3,1).reshape(-1,30)
            logi_f = self.act_f(next_hs).view(-1,5)

            next_hs = torch.stack((next_hs0, next_hs1))
        else:
            next_hs = self.forward(disc, cont, timestamp, remain, h_s, c_s, validate= validate).view(-1,128)
            logi_u = self.act_u0(next_hs).view(-1,30,48,48)
            logi_u = self.act_u1(logi_u).permute(0,2,3,1).reshape(-1,30)
            logi_f = self.act_f(next_hs).view(-1,5)

        logi_u_s = torch.split(logi_u, self.u_act_s, dim = 1)
        logi_f_s = torch.split(logi_f, self.f_act_s, dim = 1) 

        u_masks = u_masks.view(-1,u_masks.shape[-1])
        f_masks = f_masks.view(-1,f_masks.shape[-1])

        if not validate:
            copy_u_mask = u_masks.clone().detach()
            copy_f_mask = f_masks.clone().detach()

            u_masks = torch.cat((u_masks, copy_u_mask))
            f_masks = torch.cat((f_masks, copy_f_mask))

        mask_u_s = torch.split(u_masks, self.u_act_s, dim = 1)
        mask_f_s = torch.split(f_masks, self.f_act_s, dim = 1)

        if action != None:
            action_u, action_f = action
            action_u = action_u.view(-1,action_u.shape[-1]).T
            action_f = action_f.view(-1,action_f.shape[-1]).T
            m_cate_u = [MaskedCategorical(logits=logi,masks=mask) for mask,logi in zip(mask_u_s,logi_u_s)]
            m_cate_f = [MaskedCategorical(logits=logi,masks=mask) for mask,logi in zip(mask_f_s,logi_f_s)]
        else:
            m_cate_u = [MaskedCategorical(logits=logi,masks=mask) for mask,logi in zip(mask_u_s,logi_u_s)]
            m_cate_f = [MaskedCategorical(logits=logi,masks=mask) for mask,logi in zip(mask_f_s,logi_f_s)]
            action_u = torch.stack([categorical.sample() for categorical in m_cate_u])
            action_f = torch.stack([categorical.sample() for categorical in m_cate_f])

        logprob_u = torch.stack([categorical.log_prob(action) for action,categorical in zip(action_u,m_cate_u)])
        logprob_f = torch.stack([categorical.log_prob(action) for action,categorical in zip(action_f,m_cate_f)])
        logprob_u = logprob_u.T.view(-1,48*48,4)
        logprob_f = logprob_f.T.view(-1,48*48,2)
        if validate:
            entropy_u = torch.stack([categorical.entropy() for categorical in m_cate_u])
            entropy_f = torch.stack([categorical.entropy() for categorical in m_cate_f])
            entropy_u  = entropy_u.T.view(-1,48*48,4)
            entropy_f = entropy_f.T.view(-1,48*48,2)
            return logprob_u.sum(1).sum(1), logprob_f.sum(1).sum(1), entropy_u.sum(1).sum(1), entropy_f.sum(1).sum(1)
        action_f = action_f.T.view(-1,48*48,2)
        action_u = action_u.T.view(-1,48*48,4)
        u_masks = u_masks.view(-1,48*48,30)
        f_masks = f_masks.view(-1,48*48,5)
        return action_f, action_u, logprob_u.sum(1).sum(1), logprob_f.sum(1).sum(1),\
        next_hs, next_cs, u_masks, f_masks
    
    def get_value(self, disc, cont, timestamp, remain, h_s, c_s, validate = False):

        if validate:
            next_hs = self.forward(disc, cont, timestamp, remain, h_s, c_s, validate = validate).view(-1,128)
            return self.critic(next_hs)

        (hs0 , cs0), (hs1, cs1) = self.forward(disc, cont, timestamp, remain, h_s, c_s) 
        next_hs = torch.stack((hs0, hs1)).view(-1,128)
        return self.critic(next_hs)

class MaskedCategorical(Categorical):

    def __init__(self, logits, masks):

        self.masks = masks.bool()
        logits = torch.where(self.masks, logits, torch.tensor(-1e+8, device="cuda"))
        super(MaskedCategorical, self).__init__(logits = logits)

    def entropy(self):
        
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0. , device = "cuda"))
        return -p_log_p.sum(-1)
    

"""device = "cuda" if torch.cuda.is_available() else "cpu" 
model = Model(30,5).to(device)

print(f'Using {device} device')

board = torch.randint(0,3,size=(1,48,48)).to(device)
player_d = torch.randint(0,6,size=(2,48,48)).to(device)
disc = torch.cat((board, player_d)).view(1,3,48,48)

cont = torch.rand((1,13,48,48)).to(device)

timestamp = torch.rand((1,1,)).to(device)

remain = torch.randint(0,21,(1,1,)).to(device)

h_s = torch.zeros(1,1,128).to(device)
c_s = torch.zeros(1,1,128).to(device)

forw = model(disc, cont, timestamp, remain, h_s, c_s)"""

