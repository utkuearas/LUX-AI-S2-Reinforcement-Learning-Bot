import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Encoder(nn.Module):
    def __init__(self, input_channels):

        super().__init__()
        self._encoder = nn.Sequential(
            layer_init(nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 128, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(),
            layer_init(nn.Conv2d(128, 256, kernel_size=3, padding=1)),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

    def forward(self, x):
        return self._encoder(x)

class Decoder(nn.Module):
    def __init__(self, output_channels):
        super().__init__()

        self.deconv = nn.Sequential(
            layer_init(nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(32, output_channels, 3, stride=2, padding=1, output_padding=1))
        )

    def forward(self, x):
        return self.deconv(x)
    

device = "cuda"
class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()

        self.encoder_actor = Encoder(34)
        self.encoder_value = Encoder(34)

        self.actor_factory = Decoder(33)
        self.actor_unit = Decoder(33)

        self.critic = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(2304, 128), std=1),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1),
        )

    def forward_actor_factory(self, x):
        return self.actor_factory(self.encoder_actor(x)) 
    
    def forward_actor_unit(self, x):
        return self.actor_unit(self.encoder_actor(x)) 
    
    def forward_value(self, x):
        return self.critic(self.encoder_value(x))

    def get_action(self, x, action=None, invalid_action_masks=None):

        logits_factory = self.forward_actor_factory(x).reshape(-1,1,33,48,48)
        logits_units = self.forward_actor_unit(x).reshape(-1,1,33,48,48)
        logits = torch.cat((logits_factory, logits_units), 1).to(device)
        invalid_action_masks = torch.tensor(invalid_action_masks).int().to(device)
        if action is None:
            multi_categoricals = [CategoricalMasked(logits=logit, masks=mask) for logit, mask in zip(logits, invalid_action_masks)]
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        else:
            multi_categoricals = [CategoricalMasked(logits=logit, masks=iam) for (logit, iam) in
                                  zip(logits, invalid_action_masks)]
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        return action, logprob.sum(1).sum(1).sum(1), entropy.sum(1).sum(1).sum(1), invalid_action_masks
    
    def get_value(self, x):
        return self.forward_value(x)

class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[], sw=None):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = torch.moveaxis(masks, 1, 3).bool()
            logits = torch.moveaxis(logits, 1, 3)
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8, device=device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(device))
        return -p_log_p.sum(-1)

"""agent = Agent().to(device)
test = torch.rand(8,34,48,48).to(device)
print(agent.forward_value(test).size())"""

