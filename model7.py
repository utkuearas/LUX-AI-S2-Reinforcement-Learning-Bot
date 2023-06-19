import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
from torch.cuda.amp import autocast

def layer_init(layer, std = np.sqrt(2), bias_zero = True, convert = True):
    if convert:
        torch.nn.init.orthogonal_(layer.weight, std)
        if bias_zero:
            torch.nn.init.constant_(layer.bias, 0.0)
    return layer

@torch.jit.script
def fused_swish(x):
    return x / (1.0 + torch.exp(-x))
    
class FusedSwish(nn.Module):

    def __init__(self):
        super(FusedSwish, self).__init__()
    def forward(self, x):

        return fused_swish(x)

class Transpose(nn.Module):

    def __init__(self, transpose):

        super(Transpose, self).__init__()
        self.transpose = transpose

    def forward(self, x):
        
        return x.permute(self.transpose)

class ValueFunc(nn.Module):

    def __init__(self):
        super(ValueFunc, self).__init__()

        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            layer_init(nn.Linear(128,64),1.),
            FusedSwish(),
            layer_init(nn.Linear(64,1),1.)
        )

    def forward(self, x):

        return self.model(x)
    
class Model(nn.Module):

    def __init__(self, u_act, f_act, obs_space):

        
        super(Model, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f'Using {self.device} device')

        self.u_act_s = u_act
        self.f_act_s = f_act
        self.obs_space = obs_space

        self.encode0 = layer_init(nn.Conv2d(obs_space[0],16,1))
        self.encode1 = layer_init(nn.Conv2d(16,32,3,padding=1))
        self.encode2 = layer_init(nn.Conv2d(32,64,3,padding=1))
        self.encode3 = layer_init(nn.Conv2d(64,128,3,padding=1))

        self.decode0 = layer_init(nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=1))
        self.decode01 = layer_init(nn.Conv2d(128,64,1))
        self.decode1 = layer_init(nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1))
        self.decode11 = layer_init(nn.Conv2d(64,32,1))
        self.decode2 = layer_init(nn.ConvTranspose2d(32,16,3,stride=2,padding=1,output_padding=1))
        self.decode21 = layer_init(nn.Conv2d(32,16,1))
        self.decode3 = layer_init(nn.Conv2d(16,sum(f_act+u_act),1))

        self.critic = ValueFunc()

    def action(self, x):

        #ENCODE

        x = x.reshape((-1,) + self.obs_space)
        x = x.to(memory_format = torch.channels_last)

        x1 = self.encode0(x) #16,48,48
        x = fused_swish(x1)
        x2 = nn.MaxPool2d(3,stride=2,padding=1)(self.encode1(x)) #32,24,24
        x = fused_swish(x2)
        x3 = nn.MaxPool2d(3,stride=2,padding=1)(self.encode2(x)) #64,12,12
        x = fused_swish(x3)
        x4 = nn.MaxPool2d(3,stride=2,padding=1)(self.encode3(x)) #128,6,6

        #DECODE
        y = torch.cat((self.decode0(x4),x3), dim=1) #128,12,12
        y = fused_swish(self.decode01(y)) #64,12,12
        y = torch.cat((self.decode1(y),x2), dim=1) #64,24,24
        y = fused_swish(self.decode11(y)) #32,24,24
        y = torch.cat((self.decode2(y),x1), dim=1) #32,48,48
        y = fused_swish(self.decode21(y)) #16,48,48
        y = self.decode3(y) 

        return y

    def action_and_value(self, x):

        #ENCODE
        x = x.reshape((-1,) + self.obs_space)
        x = x.to(memory_format = torch.channels_last)

        x1 = self.encode0(x) #16,48,48
        x = fused_swish(x1)
        x2 = nn.MaxPool2d(3,stride=2,padding=1)(self.encode1(x)) #32,24,24
        x = fused_swish(x2)
        x3 = nn.MaxPool2d(3,stride=2,padding=1)(self.encode2(x)) #64,12,12
        x = fused_swish(x3)
        x4 = nn.MaxPool2d(3,stride=2,padding=1)(self.encode3(x)) #128,6,6

        value = self.critic(x4)

        #DECODE
        y = torch.cat((self.decode0(x4),x3), dim=1) #128,12,12
        y = fused_swish(self.decode01(y)) #64,12,12
        y = torch.cat((self.decode1(y),x2), dim=1) #64,24,24
        y = fused_swish(self.decode11(y)) #32,24,24
        y = torch.cat((self.decode2(y),x1), dim=1) #32,48,48
        y = fused_swish(self.decode21(y)) #16,48,48
        y = self.decode3(y) 

        return y, value
    
    def encode(self, x):

        x = x.reshape((-1,) + self.obs_space)
        x = x.to(memory_format = torch.channels_last)
        x1 = fused_swish(self.encode0(x)) #16,48,48
        x2 = fused_swish(nn.MaxPool2d(3,stride=2,padding=1)(self.encode1(x1))) #32,24,24
        x3 = fused_swish(nn.MaxPool2d(3,stride=2,padding=1)(self.encode2(x2))) #64,12,12
        x4 = nn.MaxPool2d(3,stride=2,padding=1)(self.encode3(x3)) #128,6,6
        return x4

    def get_action_and_value(self, x, masks, action = None, phase = "rollout"):
            
        logits, value = self.action_and_value(x)
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
        if phase == "rollout":
            action = action.T.view(-1, 48*48, len(self.u_act_s + self.f_act_s))
            return action, logprob.sum(1).sum(1), value
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        entropy = entropy.T.view(-1, 48*48, len(self.u_act_s + self.f_act_s))
        return logprob.sum(1).sum(1), entropy.sum(1).sum(1), value
    
    def get_value(self, x):

        critic = self.critic(x)
        return critic # Batch * env, values


class CategoricalMasked(Categorical):

    def __init__(self, logits, masks):

        self.masks = masks.bool()
        logits = torch.where(self.masks, logits, torch.tensor(-1e+8, device="cuda"))
        super(CategoricalMasked, self).__init__(logits = logits)

    def entropy(self):
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0. , device = "cuda"))
        return -p_log_p.sum(-1)
    


"""model = Model((36,),(4,),(31,48,48))
print(sum(p.numel() for p in model.parameters()))"""

"""data = torch.rand((5,32,48,48))
encoded_data = model.encode(data)
value = model.get_aux_value(encoded_data)
print(value)"""

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

