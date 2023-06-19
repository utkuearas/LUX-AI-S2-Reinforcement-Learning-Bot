import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
from torch.cuda.amp import autocast

def layer_init(layer, std= np.sqrt(2), bias_zero = True, convert = True):
    if convert:
        torch.nn.init.orthogonal_(layer.weight, std)
        if bias_zero:
            torch.nn.init.constant_(layer.bias, 0.)
    return layer


class ResLayer(nn.Module):

    def __init__(self, f, mid_k = 3, bias = True):

        super(ResLayer, self).__init__()
        padding_m = 0 if mid_k == 1 else (mid_k - 1) //  2
        self.conv1 = nn.Conv2d(f, f, mid_k, bias = bias, padding= padding_m)
        self.conv2 = nn.Conv2d(f, f, mid_k, bias = bias, padding = padding_m)

    def forward(self, x):

        y = self.conv1(x)
        y = F.leaky_relu(y)
        y = self.conv2(y)
        return F.leaky_relu(y + x)

    
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

        self.conv1 = nn.Conv2d(15,16,3,padding = 1)

        self.residuals = nn.Sequential(
            *[ResLayer(16) for _ in range(4)]
        )

        self.actor = nn.Sequential(
            layer_init(nn.Conv2d(16,35,3,padding = 1), std = .01),
            Transpose((0,2,3,1))
        )

        self.critic = nn.Sequential(
            nn.Conv2d(15,16,3,padding=1),
            nn.LeakyReLU(),
            ResLayer(16),
            ResLayer(16),
            nn.Conv2d(16,32,3,stride=2,padding=1),
            nn.LeakyReLU(),
            ResLayer(32),
            ResLayer(32),
            nn.Conv2d(32,64,3,stride=2,padding=1),
            nn.LeakyReLU(),
            ResLayer(64),
            ResLayer(64),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            layer_init(nn.Linear(64,1), std= 1.)
        )
        

    def forward(self, x):

        x = F.leaky_relu(self.conv1(x))
        x = self.residuals(x)
        return x

    
    def get_action(self, x, masks, action = None, validate = False):

        x = x.reshape((-1,) + self.obs_space)
        x = x.to(memory_format = torch.channels_last)
        with autocast():
            logits = self.actor(self.forward(x))

            with autocast(enabled=False):
                logits = logits.float()
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
        x = x.to(memory_format = torch.channels_last)
        with autocast():
            critic = self.critic(x)
        return critic # Batch * env, values


class CategoricalMasked(Categorical):

    def __init__(self, logits, masks):

        self.masks = masks.bool()
        number = -1e+8
        logits = torch.where(self.masks, logits, torch.tensor(number, device="cuda"))
        super(CategoricalMasked, self).__init__(logits = logits)

    def entropy(self):
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0. , device = "cuda"))
        return -p_log_p.sum(-1)
    
"""model = Model((4,5,19,2),(3,2),(15,48,48))
print(sum(p.numel() for p in model.parameters()))"""

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

