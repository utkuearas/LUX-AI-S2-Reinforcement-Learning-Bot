
import torch
import luxai_s2
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper, SubprocVecEnv, VecVideoRecorder
import time
import argparse
from distutils.util import strtobool
import os
from model import Model as model1
from model2 import Model as model2
from model3 import Model as model3
from model4 import Model as model4
from model5 import Model as model5
from model6 import Model as model6
from model7 import Model as model7
from model8 import Model as model8
import gym
from newEnv.TessEnv import TessEnv
from torch.optim import Adam, AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import kl_divergence
import time
from scaling import RunningMeanStd
import cProfile

torch.backends.cudnn.benchmark = True



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
    
class Monitor(VecEnvWrapper):

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
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PPO agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default="Tess ESPO Learning",
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="TessEnv-v1",
                        help='the id of the gym environment')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=10_000_000,
                        help='total timesteps of the experiments')
    parser.add_argument('--cuda', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x: bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="Tess",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--n-minibatch', type=int, default=20)
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--num-steps', type=int, default=512)
    parser.add_argument('--gamma', type=float, default=.99)
    parser.add_argument('--gae-lambda', type=float, default=.9)
    parser.add_argument('--ent-coef', type=float, default=1e-2)
    parser.add_argument('--clip_coef', type=float, default=.2)
    parser.add_argument('--video-trigger', type=int, default= 15000, nargs = '?', const= True)
    parser.add_argument('--model-number', type=int, default=8, nargs = '?', const= True)
    parser.add_argument('--no-wandb', type=lambda x: bool(strtobool(x)), default = False, nargs='?', const=True)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps * 2)
    args.minibatch_size = int(args.batch_size // args.n_minibatch)
    args.num_updates = (args.total_timesteps) // args.batch_size

    print(args.batch_size)


    experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{experiment_name}")
    writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
        '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
    if args.prod_mode and not args.no_wandb:
        import wandb
        
        run = wandb.init(
            project=args.wandb_project_name, entity=args.wandb_entity,
            #sync_tensorboard=True,
            config=vars(args), name=experiment_name, monitor_gym=True, save_code=True)
        wandb.tensorboard.patch(save = False)
        writer = SummaryWriter(f"/tmp/{experiment_name}")
        CHECKPOINT_FREQUENCY = 30

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using {device} device')

    env = SubprocVecEnv([lambda: gym.make(args.gym_id) for _ in range(args.num_envs)])
    env.seed(seed = args.seed)
    env = Monitor(env)
    env = VecNormalize(env)

    if args.capture_video:
        env = VecVideoRecorder(env, f'videos/{run.id}',
                                record_video_trigger=lambda x: x % args.video_trigger == 0, video_length=1000)
        
    
    agent = model8((5,6,3,6,20),(4,),(23,48,48)).to(device)#, memory_format = torch.channels_last)
        
    print(f"Using model number : {args.model_number}")
    
    if "agent.pt" in os.listdir("./"):
        agent.load_state_dict(torch.load("./agent.pt"))
        print("Loaded Succesfully")
    else:
        pass

    print(sum(p.numel() for p in agent.parameters()))

    obs = torch.zeros((args.num_steps, args.num_envs, 2,) + env.observation_space.shape, device = device)
    actions = torch.zeros((args.num_steps, args.num_envs,) + env.action_space.shape, device = device)
    logprobs = torch.zeros((args.num_steps, args.num_envs, 2), device = device)
    rewards = torch.zeros((args.num_steps, args.num_envs, 2), device = device)
    dones = torch.zeros((args.num_steps, args.num_envs, 2), device = device)
    values = torch.zeros((args.num_steps, args.num_envs, 2), device = device)
    masks = torch.zeros((args.num_steps, args.num_envs, 2, 48, 48, 44), device = device)

    """obs = torch.zeros((args.num_steps, args.num_envs,) + env.observation_space.shape, device = device)
    actions = torch.zeros((args.num_steps, args.num_envs,) + env.action_space.shape[1:], device = device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device = device)
    rewards = torch.zeros((args.num_steps, args.num_env), device = device)
    dones = torch.zeros((args.num_steps, args.num_envs), device = device)
    values = torch.zeros((args.num_steps, args.num_envs), device = device)
    masks = torch.zeros((args.num_steps, args.num_envs, 48, 48, 44), device = device)"""

    policy_optimizer = Adam(agent.parameters(), args.learning_rate)

    global_step = 1

    next_obs = torch.from_numpy(env.reset()).float().to(device)
    next_done = torch.zeros((args.num_envs, 2), device = device)
    """next_done = torch.zeros((args.num_envs,), device = device)"""
    agents = ["player_0","player_1"]

    anneal_lr = lambda update: (args.num_updates - update) / args.num_updates * args.learning_rate
    anneal_ent = lambda update: (args.num_updates - update) / args.num_updates * args.ent_coef
    
    """profiler = cProfile.Profile()
    profiler.enable()"""

    for update in range(args.num_updates):

        begin = time.time()
        #new_ent_coef = anneal_ent(update)
        new_ent_coef = args.ent_coef
    
        new_lr = anneal_lr(update)
        policy_optimizer.param_groups[0]['lr'] = new_lr

        for step in range(args.num_steps):

            global_step += 1 
            dones[step] = next_done

            real_obs = next_obs.clone().detach()
            real_obs[:,5:14,:,:] = next_obs[:,14:,:,:]
            real_obs[:,14:,:,:] = next_obs[:,5:14,:,:]
            real_obs = torch.stack((next_obs, real_obs), dim = 1)
            obs[step] = real_obs

            mask0 = torch.from_numpy(np.array(env.get_attr("valid_actions_0")))
            mask1 = torch.from_numpy(np.array(env.get_attr("valid_actions_1")))
            mask = torch.stack((mask0, mask1), dim = 1).to(device)

            masks[step] = mask

            with torch.no_grad():

                action, logprob, value = agent.get_action_and_value(real_obs, mask, phase = "rollout")
            
            action = action.reshape((-1,) + env.action_space.shape)
            logprob = logprob.reshape(-1, 2)
            value = value.reshape(-1, 2)

            logprobs[step] = logprob
            values[step] = value
            actions[step] = action

            """logprobs[step] = logprob[:,0]
            values[step] = value[:,0]
            actions[step] = action[:,0]"""

            action = action.cpu().numpy() 

            next_obs, rew, done, info = env.step(action) 
            next_obs = torch.from_numpy(next_obs).float().to(device)

            rew = torch.from_numpy(rew).to(device)
            rewards[step] = rew
            next_done = torch.from_numpy(done).view(-1,1).expand(-1,2).to(device)
            """next_done = torch.from_numpy(done).view(-1,1).to(device)"""

            state = False
            episode_data_list = dict()
            for i in range(len(done)):
                if done[i]:
                    state = True
                    inf = info[i]
                    episode_data = inf["episode"]
                    t_d = episode_data["t"]
                    for key,value in episode_data.items():
                        if key == "t":
                            continue
                        if key not in episode_data_list:
                            episode_data_list[key] = [value]
                        else:
                            episode_data_list[key] += [value]
            if state:
                for key,value in episode_data_list.items():
                    writer.add_scalar(f"episode/{key}",np.mean(value),global_step)

        with torch.no_grad():

            real_obs = next_obs.clone().detach()
            real_obs[:,5:14,:,:] = next_obs[:,14:,:,:]
            real_obs[:,14:,:,:] = next_obs[:,5:14,:,:]
            real_obs = torch.stack((next_obs, real_obs), dim = 1)
            fea = agent.forward(real_obs)
            last_value = agent.get_value(fea) # env * player , values
            last_value = last_value.view(-1,2)
            advantages = torch.zeros_like(rewards, device=device) # add to device
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = (1.0 - next_done)
                    nextvalues = last_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.view((-1,) + env.observation_space.shape)
        b_actions = actions.view(-1,48,48,6)
        b_logprobs = logprobs.view(-1)
        b_rewards = rewards.view(-1)
        b_returns = returns.view(-1)
        b_advantages = advantages.view(-1)
        b_masks = masks.view(-1,48,48,44)
        b_values = values.view(-1)
        
        early_stop = False
        inds = np.arange(args.batch_size,)
        for i_epoch in range(args.epoch):
            np.random.shuffle(inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                minibatch_ind = inds[start:end]

                mb_advantages = b_advantages[minibatch_ind]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                mb_obs = b_obs[minibatch_ind]
                mb_masks = b_masks[minibatch_ind]
                mb_actions = b_actions[minibatch_ind]

                new_logprob, entropy, value = agent.get_action_and_value(mb_obs, mb_masks, mb_actions, phase = "update")          

                mb_logprob = b_logprobs[minibatch_ind]
                mb_returns = b_returns[minibatch_ind]
                mb_values = b_values[minibatch_ind]

                log = new_logprob - mb_logprob

                ratio = (log).exp()
                with torch.no_grad():
                    approx_kl = (ratio - 1 - log).mean()
                ratio_clip = torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(-mb_advantages * ratio_clip, -mb_advantages * ratio).mean()

                v_loss = (value - mb_returns).square().mean() * .5

                ent_loss = entropy.mean()
                loss =  pg_loss + v_loss - ent_loss * new_ent_coef

                policy_optimizer.zero_grad(set_to_none = True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), .5)
                policy_optimizer.step()

            

        if args.prod_mode and not args.no_wandb:
            if not os.path.exists(f"models/{experiment_name}"):
                os.makedirs(f"models/{experiment_name}")
                torch.save(agent.state_dict(), f"{wandb.run.dir}/agent.pt")
                wandb.save(f"agent.pt")
            else:
                if update % CHECKPOINT_FREQUENCY == 0:
                    torch.save(agent.state_dict(), f"{wandb.run.dir}/agent.pt")
        

        writer.add_scalar("charts/update", update, global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("optimization/entropy", ent_loss.item(), global_step)
        writer.add_scalar("optimization/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("charts/fps", int(args.batch_size / (time.time() - begin)), global_step)

        layers = agent.state_dict()
        for layer_name, tensor in layers.items():
            flattened_data = tensor.flatten()
            writer.add_histogram(f"model/{layer_name}",flattened_data,global_step,max_bins=512)


    """profiler.disable()
    profiler.print_stats()"""

        



        









