import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np

def layer_init(layer, std= np.sqrt(2), bias_zero = True):
    torch.nn.init.orthogonal_(layer.weight, std)
    if bias_zero:
        torch.nn.init.constant_(layer.bias, 0.)
    return layer


class ResLayer(nn.Module):

    def __init__(self, in_f, red_l, end_f, start_k = 1, mid_k = 3, bias = True, use_relu = True, use_BN = False):

        super(ResLayer, self).__init__()
        padding_s = 0 if start_k == 1 else (start_k - 1) //  2
        padding_m = 0 if mid_k == 1 else (mid_k - 1) //  2
        self.use_relu = use_relu
        self.use_BN = use_BN
        self.conv1 = layer_init(nn.Conv2d(in_f, red_l, start_k, bias = bias, padding= padding_s))
        self.conv2 = layer_init(nn.Conv2d(red_l, red_l, mid_k, bias = bias, padding = padding_m))
        self.conv3 = layer_init(nn.Conv2d(red_l, end_f, start_k, bias = bias, padding= padding_s))
        self.correction = None
        if in_f != end_f:
            self.conv4 = layer_init(nn.Conv2d(in_f, end_f, start_k, bias = bias, padding = padding_s))
            self.correction = nn.Sequential(
                self.conv4,
                nn.ReLU()
            )
        if use_BN:

            self.bn1 = nn.BatchNorm2d(red_l)
            self.bn2 = nn.BatchNorm2d(red_l)
            self.bn3 = nn.BatchNorm2d(in_f)

    def forward(self, x):

        if self.use_relu and self.use_BN:
            y = self.conv1(x)
            y = F.relu(self.bn1(y),inplace=True)
            y = self.conv2(y)
            y = F.relu(self.bn2(y),inplace=True)
            y = self.conv3(y)
            y = F.relu(self.bn3(y)+x,inplace=True)
        elif self.use_relu and not self.use_BN:
            y = F.relu(self.conv1(x),inplace=True)
            y = F.relu(self.conv2(y),inplace=True)
            if self.correction != None:
                x = self.correction(x)
            y = F.relu(self.conv3(y) + x,inplace=True)
        else:
            y = self.conv1(x)
            y = self.conv2(y)
            y = self.conv3(y) + x
        return y
    
class Transpose(nn.Module):

    def __init__(self, permute):

        super(Transpose, self).__init__()
        self.permute = permute
    def forward(self ,x):

        return x.permute(self.permute)
    
class Model(nn.Module):

    def __init__(self, u_act, f_act, obs_space):

        super(Model, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f'Using {self.device} device')

        self.u_act_s = u_act
        self.f_act_s = f_act
        self.obs_space = obs_space

        self.conv1 = nn.Conv2d(15,32,3,padding = 1)
        self.conv2 = nn.Conv2d(32,64,3,padding = 1)
        self.conv3 = nn.Conv2d(64,128,3,padding = 1)
        self.conv4 = nn.Conv2d(128,256,3,padding = 1)

        self.actor = nn.Sequential(
            layer_init(nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)),
            nn.LeakyReLU(),
            layer_init(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)),
            nn.LeakyReLU(),
            layer_init(nn.ConvTranspose2d(64, 35, 3, stride=2, padding=1, output_padding=1), std = .01),
            Transpose((0,2,3,1))
        )

        self.critic = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            layer_init(nn.Linear(256,128), std= 1.),
            nn.LeakyReLU(),
            layer_init(nn.Linear(128,1), std= 1.)
        )
        

    def forward(self, x):

        x = F.leaky_relu(self.conv1(x), inplace = True)
        x = nn.MaxPool2d(3,stride=2,padding=1)(x)
        x = F.leaky_relu(self.conv2(x), inplace = True)
        x = nn.MaxPool2d(3,stride=2,padding=1)(x)
        x = F.leaky_relu(self.conv3(x), inplace = True)
        x = nn.MaxPool2d(3,stride=2,padding=1)(x)
        x = F.leaky_relu(self.conv4(x), inplace = True)
        return x

    
    def get_action(self, x, masks, action = None, validate = False):

        x = x.reshape((-1,) + self.obs_space)
        logits = self.actor(self.forward(x))
        grid_logits = logits.reshape(-1, sum(self.u_act_s + self.f_act_s))
        split_logits = torch.split(grid_logits, self.u_act_s + self.f_act_s, dim=1)
        grid_masks = masks.view(-1, masks.shape[-1])
        split_masks = torch.split(grid_masks, self.u_act_s + self.f_act_s,dim=1)
        if action is None:
            multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits, split_masks)]
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        else:
            action = action.view(-1, action.shape[-1]).T
            multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits, split_masks)]

        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        logprob = logprob.T.view(-1, 48*48, len(self.u_act_s + self.f_act_s))
        if not validate:
            action = action.T.view(-1, 48*48, len(self.u_act_s + self.f_act_s))
            return action, logprob.sum(1).sum(1)
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        entropy = entropy.T.view(-1, 48*48, len(self.u_act_s + self.f_act_s))
        return logprob.sum(1).sum(1), entropy.sum(1).sum(1)
    
    def get_value(self, x):

        x = x.view((-1,) + self.obs_space)
        
        return self.critic(self.forward(x)) # Batch * env, values


class CategoricalMasked(Categorical):

    def __init__(self, logits, masks):

        self.masks = masks.bool()
        logits = torch.where(self.masks, logits, torch.tensor(-1e+8, device="cuda"))
        super(CategoricalMasked, self).__init__(logits = logits)

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

