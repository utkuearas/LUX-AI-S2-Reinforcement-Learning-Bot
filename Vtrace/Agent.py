import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np

def layer_init(layer, std =.01):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, 0.)
    return layer

@torch.jit.script
def fused_leaky_relu(x):
    return torch.max(.01*x,x)

@torch.jit.script
def fused_sigmoid(x):
    return 1 / (1 + torch.exp(-x))
    
class FusedLeakyRelu(nn.Module):

    def __init__(self):
        super(FusedLeakyRelu, self).__init__()
    def forward(self, x):

        return fused_leaky_relu(x)

class Transpose(nn.Module):

    def __init__(self, transpose):

        super(Transpose, self).__init__()
        self.transpose = transpose

    def forward(self, x):
        
        return x.permute(self.transpose)

class ReciprocalBlock(nn.Module):

    def __init__(self, in_f, kernel=7):

        super(ReciprocalBlock, self).__init__()
        if kernel == 7:
            pad = 3
        elif kernel == 3:
            pad = 1
        self.conv1 = nn.Conv2d(in_f, in_f, kernel, padding=pad)
        self.conv2 = nn.Conv2d(in_f, in_f, kernel, padding=pad)

    def forward(self, x):

        y = fused_leaky_relu(self.conv1(x))
        y = self.conv2(y)
        return fused_leaky_relu(y+x)
    
class Agent(nn.Module):

    def __init__(self, u_act, f_act, obs_space):
        
        super(Agent, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f'Using {self.device} device')

        self.u_act_s = u_act
        self.f_act_s = f_act
        self.obs_space = obs_space

        self.action_type_unit = nn.Sequential(
            nn.Conv2d(obs_space[0], 32, 7, padding=3),
            FusedLeakyRelu(),
            *[ReciprocalBlock(32) for _ in range(1)],
            layer_init(nn.Conv2d(32, u_act[0], 7, padding=3)),
            Transpose((0,2,3,1))
        )
        self.move_dir = nn.Sequential(
             nn.Conv2d(obs_space[0], 32, 7, padding=3),
             FusedLeakyRelu(),
            *[ReciprocalBlock(32) for _ in range(1)],
            layer_init(nn.Conv2d(32, u_act[1], 7, padding=3)),
            Transpose((0,2,3,1))
        )
        self.transfer_type = nn.Sequential(
             nn.Conv2d(obs_space[0], 32, 7, padding=3),
             FusedLeakyRelu(),
            *[ReciprocalBlock(32) for _ in range(1)],
            layer_init(nn.Conv2d(32, u_act[2], 7, padding=3)),
            Transpose((0,2,3,1))
        )
        self.transfer_dir = nn.Sequential(
             nn.Conv2d(obs_space[0], 32, 7, padding=3),
             FusedLeakyRelu(),
            *[ReciprocalBlock(32) for _ in range(1)],
            layer_init(nn.Conv2d(32, u_act[3], 7, padding=3)),
            Transpose((0,2,3,1))
        )
        self.pick_up_amount = nn.Sequential(
             nn.Conv2d(obs_space[0], 32, 7, padding=3),
             FusedLeakyRelu(),
            *[ReciprocalBlock(32) for _ in range(1)],
            layer_init(nn.Conv2d(32, u_act[4], 7, padding=3)),
            Transpose((0,2,3,1))
        )
        self.factory_action = nn.Sequential(
             nn.Conv2d(obs_space[0], 32, 7, padding=3),
             FusedLeakyRelu(),
            *[ReciprocalBlock(32) for _ in range(1)],
            layer_init(nn.Conv2d(32, f_act[0], 7, padding=3)),
            Transpose((0,2,3,1))
        )

        self.critic = nn.Sequential(
            nn.Conv2d(obs_space[0], 16, 3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1), # 24
            FusedLeakyRelu(),
            *[ReciprocalBlock(16,kernel=3) for _ in range(2)],
            nn.Conv2d(16, 16, 3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1), # 12
            FusedLeakyRelu(),
            *[ReciprocalBlock(16,kernel=3) for _ in range(2)],
            nn.Conv2d(16, 32, 3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1), # 6
            FusedLeakyRelu(),
            *[ReciprocalBlock(32,kernel=3) for _ in range(2)],
            nn.Conv2d(32, 32, 3, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1), # 3
            FusedLeakyRelu(),
            *[ReciprocalBlock(32,kernel=3) for _ in range(2)],
            nn.Flatten(),
            nn.Linear(288,1)
        )

    def forward(self, x):
        _1 = self.action_type_unit(x)
        _2 = self.move_dir(x)
        _3 = self.transfer_type(x)
        _4 = self.transfer_dir(x)
        _5 = self.pick_up_amount(x)
        _6 = self.factory_action(x)
        return torch.cat((_1,_2,_3,_4,_5,_6), dim=3)

    def get_action_and_value(self, x, masks, action = None, phase = "rollout", no_value = True, skip_obs = None):

        x = x.reshape((-1,) + self.obs_space)
        if phase == "rollout":
            logits = self.forward(x)
        elif phase == "update":
            logits = self.forward(x[:-skip_obs])

        if not no_value:
            value = self.get_value(x)

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
            return action, logprob.sum(1).sum(1)
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        entropy = entropy.T.view(-1, 48*48, len(self.u_act_s + self.f_act_s))
        return logprob.sum(1).sum(1), entropy.sum(1).sum(1), value
    
    def get_value(self, x):

        critic = self.critic(x)
        return critic # Batch * env, values


class CategoricalMasked(Categorical):

    INF_TENSOR = torch.tensor(torch.finfo(torch.float32).min, device="cuda")

    def __init__(self, logits, masks):

        self.masks = masks
        logits = torch.lerp(CategoricalMasked.INF_TENSOR, logits, masks)
        super(CategoricalMasked, self).__init__(logits = logits)

    def entropy(self):

        p_log_p = self.logits * self.masks 
        p_log_p = p_log_p * self.probs
        return -p_log_p.sum(-1)
    
"""model = Agent((5,6,3,6,20),(4,),(23,48,48))
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

