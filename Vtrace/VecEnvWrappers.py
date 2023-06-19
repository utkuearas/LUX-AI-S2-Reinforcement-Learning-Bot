from stable_baselines3.common.vec_env import VecEnvWrapper
from utils import RunningMeanStd
import numpy as np
import time

class VecNormalize(VecEnvWrapper):

    def __init__(self, venv, cliprew=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)
        
        self.ret_rms = RunningMeanStd(shape=())
        self.cliprew = cliprew
        self.ret = np.zeros((self.num_envs,2))
        self.gamma = gamma
        self.epsilon = epsilon
        self.rew_monitor = np.zeros((self.num_envs,2))

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        if self.ret_rms:
            self.ret_rms.update(self.ret.reshape(-1))
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
            self.rew_monitor += rews

        done_info = np.array(dones, dtype = bool).reshape(-1)
        
        for i in range(len(done_info)):
            if done_info[i]:
                infos[i]["episode"]["dev"] = self.ret_rms.var
                infos[i]["episode"]["norm_rew"] = np.mean(self.rew_monitor[i])
        self.rew_monitor[done_info,:] = 0
        self.ret[done_info,:] = 0.
        return obs, rews, dones, infos

    def reset(self):
        self.ret = np.zeros((self.num_envs,2))
        obs = self.venv.reset()
        return obs
    
class VecMonitor(VecEnvWrapper):

    def __init__(self, venv):

        VecEnvWrapper.__init__(self, venv)
        self.start = time.time()

    def reset(self):
        obs = self.venv.reset()
        self.e_len = np.zeros(self.num_envs, dtype=np.int32)
        self.e_rew_0 = np.zeros(self.num_envs, dtype=np.float32)
        self.e_rew_1 = np.zeros(self.num_envs, dtype=np.float32)
        self.gat_i_0 = np.zeros(self.num_envs, dtype=np.float32)
        self.gat_i_1 = np.zeros(self.num_envs, dtype=np.float32)
        self.tra_i_0 = np.zeros(self.num_envs, dtype=np.float32)
        self.tra_i_1 = np.zeros(self.num_envs, dtype=np.float32)
        self.pus_h_0 = np.zeros(self.num_envs, dtype=np.float32)
        self.pus_h_1 = np.zeros(self.num_envs, dtype=np.float32)
        self.tra_o_0 = np.zeros(self.num_envs, dtype=np.float32)
        self.tra_o_1 = np.zeros(self.num_envs, dtype=np.float32)
        self.gat_o_0 = np.zeros(self.num_envs, dtype=np.float32)
        self.gat_o_1 = np.zeros(self.num_envs, dtype=np.float32)
        self.bui_0 = np.zeros(self.num_envs, dtype=np.float32)
        self.bui_1 = np.zeros(self.num_envs, dtype=np.float32)
        return obs
    
    def step_wait(self):

        obs, rew, done, infos = self.venv.step_wait()

        self.e_rew_0 += rew[:,0]
        self.e_rew_1 += rew[:,1]
        self.e_len += 1

        for i in range(len(infos)):
            self.gat_i_0[i] += infos[i]["gat_i_0"]
            self.gat_i_1[i] += infos[i]["gat_i_1"]
            self.tra_i_0[i] += infos[i]["tra_i_0"]
            self.tra_i_1[i] += infos[i]["tra_i_1"]
            self.pus_h_0[i] += infos[i]["pus_h_0"]
            self.pus_h_1[i] += infos[i]["pus_h_1"]
            self.tra_o_0[i] += infos[i]["tra_o_0"]
            self.tra_o_1[i] += infos[i]["tra_o_1"]
            self.gat_o_0[i] += infos[i]["gat_o_0"]
            self.gat_o_1[i] += infos[i]["gat_o_1"]
            self.bui_0[i] += infos[i]["bui_0"]
            self.bui_1[i] += infos[i]["bui_1"]

        new_infos = list(infos[:])
        for i in range(len(done)):
            if done[i]:
                info = infos[i].copy()
                episode_data = dict(t = round(time.time() - self.start), e_len = self.e_len[i], e_rew_0 = self.e_rew_0[i], e_rew_1 = self.e_rew_1[i],\
                                    gat_i_0 = self.gat_i_0[i], gat_i_1 = self.gat_i_1[i], tra_i_0 = self.tra_i_0[i],\
                        tra_i_1 = self.tra_i_1[i], pus_h_0 = self.pus_h_0[i], pus_h_1 = self.pus_h_1[i], tra_o_0 = self.tra_o_0[i], tra_o_1 = self.tra_o_1[i],\
                            gat_o_0 = self.gat_o_0[i], gat_o_1 = self.gat_o_1[i], bui_0 = self.bui_0[i], bui_1 = self.bui_1[i], li_0 = infos[i]["li_0"],\
                                li_1 = infos[i]["li_1"])
                info["episode"] = episode_data
                new_infos[i] = info
                self.e_len[i] = 0
                self.e_rew_0[i] = 0
                self.e_rew_1[i] = 0
                self.gat_i_0[i] = 0
                self.gat_i_1[i] = 0
                self.tra_i_0[i] = 0
                self.tra_i_1[i] = 0
                self.pus_h_0[i] = 0
                self.pus_h_1[i] = 0
                self.tra_o_1[i] = 0
                self.tra_o_0[i] = 0
                self.gat_o_0[i] = 0
                self.gat_o_1[i] = 0
                self.bui_0[i] = 0
                self.bui_1[i] = 0

        return obs, rew, done, new_infos