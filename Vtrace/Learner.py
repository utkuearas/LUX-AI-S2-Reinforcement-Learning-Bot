import torch
from torch.optim import Adam
from Actor import Actor
import time
import threading
from Agent import Agent
from copy import deepcopy
from random import uniform
import time
import os
class Memory:

    def __init__(self, num_step, env_amount, gym_id, buffer_size, actor_amount):

        self.env_amount = env_amount
        self.num_step = num_step
        self.Actors = [Actor(num_step, gym_id) for _ in range(actor_amount)]
        self.obs_space = self.Actor.obs_space
        self.act_space = self.Actor.act_space
        self.obs = torch.zeros((buffer_size, num_step ,) + self.obs_space)
        self.bootstrap_obs = torch.zeros((buffer_size ,) + self.obs_space)
        self.actions = torch.zeros((buffer_size , num_step,) + (48,48,6))
        self.logprobs = torch.zeros((buffer_size , num_step))
        self.masks = torch.zeros((buffer_size, num_step, 48, 48, 44))
        self.dones = torch.zeros((buffer_size, num_step))
        self.rewards = torch.zeros((buffer_size, num_step))
        self.index = 0
        self.isFull = False
        self.put_index = -1
        self.sync_background = True
        self.multiplier = buffer_size

    def sample_data(self):

        if not self.isFull:
            return None
        
        data = self.obs[self.index].to("cuda"), self.actions[self.index].to("cuda"), self.logprobs[self.index].to("cuda"), self.masks[self.index].to("cuda"),\
        self.dones[self.index].to("cuda"), self.rewards[self.index].to("cuda"), self.bootstrap_obs[self.index].to("cuda")

        self.index += 1
        if self.index == self.multiplier:
            self.index = 0
        return data
    
    def sync_data(self):
        while self.sync_background:
            if self.Actor.isReady():
                obs, actions, logprobs, rewards, dones, masks, bootstrap_obs = self.Actor.get_items()
                obs = obs
                actions = actions
                logprobs = logprobs
                rewards = rewards
                dones = dones
                masks = masks
                bootstrap_obs = bootstrap_obs
                for i in range(obs.shape[0]):
                    self.put_index += 1
                    if self.put_index == self.multiplier:
                        self.put_index = 0
                        self.isFull = True
                    self.obs[self.put_index] = obs[i]
                    self.actions[self.put_index] = actions[i]
                    self.logprobs[self.put_index] = logprobs[i]
                    self.rewards[self.put_index] = rewards[i]
                    self.dones[self.put_index] = dones[i]
                    self.masks[self.put_index] = masks[i]
                    self.bootstrap_obs[self.put_index] = bootstrap_obs[i]
                self.Actor.run_once()
            time.sleep(1)
    
    def sync_data_thread(self):
            
        th = threading.Thread(target = self.sync_data)
        th.start()

    def kill(self):
        self.sync_background = False

class Learner:

    def __init__(self, num_steps, env_amount, gym_id, multiplier):
        
         self.target_net = Agent((5,6,3,6,20),(4,),(23,48,48)).to("cuda")
         if not os.path.isfile("./agent.pt"):
            torch.save(self.target_net.state_dict(), "./agent.pt")
            print("Saved")
         self.memory = Memory(num_steps, env_amount, gym_id, multiplier)
         self.learning_state = False
         self.clip_vtrace_params = 1.
         self.gamma = .99
         self.lambda_ = .95
         self.env_amount = env_amount
         self.ent_coef = .01
         self.vf_coef = 1.
         self.min_aug = .99
         self.max_aug = 1.01
         self.num_steps = num_steps
         self.device="cuda"
         self.lr = 1e-6
         self.optimizer = Adam(self.target_net.parameters(), self.lr)
         self.m = 0
         self.total_time = 115200
         self.calc_lr = lambda t: ((self.total_time - t) / self.total_time) * self.lr
         self.t = time.time()
         self.global_step = 1

    def get_trajectory(self):

        data = self.memory.sample_data()
        return data
    
    def set_config(self):
        pass

    def start_learning_thread(self):

        self.memory.sync_data_thread()

        while self.learning_state and self.total_time > time.time() - self.t:

            data = self.memory.sample_data()
            if data == None:
                time.sleep(1)
                continue

            new_lr = self.calc_lr(time.time() - self.t)
            self.optimizer.param_groups[0]["lr"] = new_lr

            obs, actions, oldlogprobs, masks, dones, rewards, bootstrap_obs = data
            newlogprobs, entropy, value = self.prepare_data(obs, bootstrap_obs, masks, actions)

            oldlogprobs = oldlogprobs.reshape(-1)
            newlogprobs = newlogprobs.reshape(-1)
            value = value.reshape(self.m+1,-1).permute((1,0))

            with torch.no_grad():
                ratio = (newlogprobs - oldlogprobs).exp()
                approx_kl = (ratio - 1 - ratio.log()).mean()
                rhos = self.calc_rhos(newlogprobs, oldlogprobs)
                returns_ = self.calc_returns(value, dones, rewards, rhos)
                advantages = self.calc_advantages(torch.cat((returns_[1:], value[-1:])), value[:-1], rewards, rhos)

            self.memory.Actor.writer.add_scalar("optimization/kl",approx_kl.item(),self.global_step)
            self.memory.Actor.writer.add_scalar("optimization/lr",new_lr,self.global_step)
            self.memory.Actor.writer.add_scalar("optimization/adv",advantages.mean(),self.global_step)
            self.memory.Actor.writer.add_scalar("optimization/log",newlogprobs.mean(),self.global_step)
            
            pg = -(newlogprobs * advantages).mean()
            ent = (-entropy * self.ent_coef).mean()
            v = ((value[:-1,0] - returns_[:,0]).square() * .5 * self.vf_coef).mean()
            loss = pg + ent + v

            self.memory.Actor.writer.add_scalar("optimization/pg",pg.item(),self.global_step)
            self.memory.Actor.writer.add_scalar("optimization/ent",ent.item(),self.global_step)
            self.memory.Actor.writer.add_scalar("optimization/v",v.item(),self.global_step)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            torch.save(self.target_net.state_dict(), "./agent.pt")
            self.global_step += 1

        self.memory.kill()
                    

    def start_learning(self):

        self.learning_state = True
        self.memory.Actor.run_once()
        th = threading.Thread(target=self.start_learning_thread)
        th.start()

    def pause_learning(self):

        self.learning_state = False
        self.memory.kill()

    def prepare_data(self, obs, bootstrap_obs, masks, actions):

        real_obs = torch.cat((obs, bootstrap_obs.reshape((1,) + bootstrap_obs.shape))).to(self.device)
        real_obs = self.augment_data(real_obs)
        masks = masks.to(self.device)
        actions = actions.to(self.device)
        skip_obs = 1 + self.m * (self.num_steps + 1)
        logprob, entropy, value = self.target_net.get_action_and_value(real_obs, masks, actions, phase = "update", no_value = False, skip_obs=skip_obs)
        return logprob, entropy, value
    
    def calc_rhos(self, newlogprob, oldlogprob):

        return torch.clamp((newlogprob - oldlogprob).exp(), max = self.clip_vtrace_params)
    
    def calc_returns(self, values, dones, rewards, rhos):

        values_t = values[:-1]
        values_t_plus_1 = values[1:]
        weights = (1.0 - dones).reshape(self.num_steps,-1).expand_as(values_t) * self.gamma
        rhos = rhos.reshape(self.num_steps,-1).expand_as(values_t)
        rewards = rewards.reshape(self.num_steps,-1).expand_as(values_t)
        
        deltas = rhos * (rewards + weights * values_t_plus_1 - values_t)

        factor = self.lambda_ * self.gamma

        returns = values_t.clone()
        vtrace_item = 0.
        for t in reversed(range(len(values_t))):

            vtrace_item = deltas[t] + factor * rhos[t] * vtrace_item
            returns[t] += vtrace_item

        return returns
    
    def calc_advantages(self, returns, values, rewards, rhos):

        rhos = rhos.reshape(self.num_steps,-1).expand_as(returns)
        rewards = rewards.reshape(self.num_steps,-1).expand_as(returns)
        return (rhos * (rewards + self.gamma * returns - values)).mean(dim=1)
        
    def augment_data(self, obs):

        obs = obs.reshape((-1,) + obs.shape)
        real_obs = deepcopy(obs)

        for _ in range(self.m):

            number = uniform(self.min_aug,self.max_aug)
            real_obs = torch.cat((real_obs, obs * number),dim=0)

        return real_obs.reshape((-1,) + real_obs.shape[2:])

            






