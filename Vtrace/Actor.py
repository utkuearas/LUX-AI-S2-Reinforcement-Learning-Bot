import torch
from stable_baselines3.common.vec_env import VecEnvWrapper, SubprocVecEnv, VecVideoRecorder
import luxai_s2
from newEnv import TessEnv
from VecEnvWrappers import VecNormalize, VecMonitor
import gym
from Agent import Agent
import numpy as np
import threading
import cProfile
import time
from torch.utils.tensorboard import SummaryWriter
import os

class Memory:

    def __init__(self, num_step, env_action_space, env_obs_space):

        self.num_step = num_step
        self.obs = torch.zeros((num_step,) + env_obs_space)
        self.actions = torch.zeros((num_step,) + (48, 48, 6))
        self.logprobs = torch.zeros((num_step))
        self.rewards = torch.zeros((num_step))
        self.dones = torch.zeros((num_step))
        self.masks = torch.zeros((num_step, 48, 48, 44))
        self.index = 0
        self.ready = False
    
    def step(self, obs, action, logprob, reward, done, mask, bootstrap_obs = None):

        self.obs[self.index] = obs
        self.actions[self.index] = action
        self.logprobs[self.index] = logprob
        self.rewards[self.index] = reward
        self.dones[self.index] = done
        self.masks[self.index] = mask
        if bootstrap_obs != None:
            self.bootstrap_obs = bootstrap_obs
        self.index += 1
        if self.index == self.num_step:
            self.ready = True

    def get_items(self):

        self.ready = False
        self.index = 0
        return self.obs, self.actions, self.logprobs, self.rewards, self.dones, self.masks, self.bootstrap_obs
    

class Actor:

    def __init__(self, num_step, gym_id):

        self.env = gym.make(gym_id) 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.obs_space = self.env.observation_space.shape
        self.act_space = self.env.action_space.shape
        self.memory = Memory(num_step, self.act_space, self.obs_space)
        self.Agent = Agent((5,6,3,6,20),(4,),self.obs_space).to(self.device)
        self.Agent.load_state_dict(torch.load("./agent.pt"))
        self.num_step = num_step
        self.next_obs = torch.from_numpy(self.env.reset()).float()
        import wandb
        
        self.run_wan = wandb.init(
            project="Tess", entity=None,
            name="Tess", monitor_gym=True, save_code=True)
        wandb.tensorboard.patch(save = False)
        self.writer = SummaryWriter(f"/tmp/Tess")
        self.CHECKPOINT_FREQUENCY = 30
        self.global_step = 1

    def run(self):
        while True:
            try:
                self.Agent.load_state_dict(torch.load("./agent.pt"))
                break
            except:
                time.sleep(.5)
        with torch.no_grad():

            while not self.memory.ready:

                mask0 = torch.from_numpy(self.env.valid_actions_0)
                mask1 = torch.from_numpy(self.env.valid_actions_1)
                masks = torch.stack((mask0, mask1), dim = 0).to(self.device)

                real_obs = self.next_obs.clone().detach()
                real_obs[5:14,:,:] = self.next_obs[14:,:,:]
                real_obs[14:,:,:] = self.next_obs[5:14,:,:]
                real_obs = torch.stack((self.next_obs, real_obs), dim = 0)

                action, logprob = self.Agent.get_action_and_value(real_obs.to(self.device), masks)

                action = action.reshape(self.act_space).cpu().numpy()
                logprob = logprob.to("cpu")

                obs, rews, dones, infos = self.env.step(action)

                rews = torch.from_numpy(rews)
                next_done = torch.from_numpy(dones)

                bootstrap_obs = None
                if self.memory.index + 1 == self.num_step:
                    bootstrap_obs = torch.from_numpy(obs).float()

                self.memory.step(real_obs[0,:,:,:].to("cpu"), torch.from_numpy(action)[0,:,:,:], logprob[0], rews[0], next_done, masks[0,:,:,:].to("cpu"), bootstrap_obs)

                self.next_obs = torch.from_numpy(obs).float()

                if dones:
                    for key,value in infos.items():
                        pass
                        #Summary writer here
                    self.env.reset()

    def isReady(self):

        return self.memory.ready
        
    def get_items(self):

        return self.memory.get_items()

    def run_once(self):

        th = threading.Thread(target=self.run)
        th.start()
