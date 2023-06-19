

import torch
import gym
import luxai_s2
from gym.spaces import MultiDiscrete, Box
import gym.spaces as spaces
from copy import deepcopy
import numpy as np
from map_setup import mapSetupTool
import time

class TessEnv(gym.Wrapper):

    def __init__(self, seed = None):

        #Initialize env
        env = gym.make("LuxAI_S2-v0", collect_stats = True)
        super(TessEnv, self).__init__(env)
        #Setup the config
        wrapped_env_config = self.__setup_cfg(self.env.env_cfg)
        wrapped_env_state = self.env.state
        wrapped_env_state.env_cfg = wrapped_env_config
        self.env.set_state(wrapped_env_state)
        #Action space and observation space
        single_action = torch.tensor((5,6,3,6,20,4)).view(1,1,1,6)
        action_space = single_action.expand(2,48,48,-1)
        self.action_space = MultiDiscrete(action_space)
        self.observation_space = Box(low = 0., high = 1., shape = (23,48,48), dtype = np.float32)
        self.saved_state = None
        self.mp_tool = mapSetupTool()
        self.coll_map = np.zeros((2,48,48))
        self.seed(seed)

    def __setup_cfg(self, cfg):
        
        cfg.verbose = 0
        cfg.validate_action_space = False
        return cfg

    def reset(self):
        
        #self.env.reset(seed = self.saved_seed)
        self.env.reset()

        self.total_reward_0 = 0
        self.total_reward_1 = 0
        self.coll_map = np.zeros((2,48,48))
        self.factory_map = np.ones((48,48)) * -1
        self.resource_map = np.zeros((48,48))
        self.dig_map = np.zeros((5,48,48))

        self.valid_actions_1 = np.zeros((48,48,44), dtype = np.float32)
        self.valid_actions_0 = np.zeros((48,48,44), dtype = np.float32)
        self.valid_actions_0[:,:,[0,5,11,14,20,40]] = 1
        self.valid_actions_1[:,:,[0,5,11,14,20,40]] = 1

        """if self.saved_state != None:
            self.env.set_state(self.saved_state)
            self.saved_state = deepcopy(self.env.state)
            obs = self.env.state.get_obs()
            obs = dict(player_0 = obs, player_1 = obs)
        else:"""
        obs = self.__skip_early_setup()

        self.unit_count_0 = 0
        self.factory_count_0 = len(obs["player_0"]["teams"]["player_0"]["factory_strains"])

        self.unit_count_1 = 0
        self.factory_count_1 = len(obs["player_0"]["teams"]["player_1"]["factory_strains"])

        self.total_factory_count = self.factory_count_1

        self.gather_reward_0 = 0
        self.gather_reward_1 = 0

        self.move_reward_0 = 0
        self.move_reward_1 = 0

        self.build_reward_0 = 0
        self.build_reward_1 = 0

        self.consumed_energy_0 = 0
        self.consumed_energy_1 = 0

        self.last_stats = deepcopy(self.env.state.stats)
        
        obs = self.__prepare_obs(obs)
        return obs
    
    def seed(self, seed = 0):

        self.saved_seed = seed
    
    def step(self, actions): # Design Required

        self.move_reward_0 = 0
        self.move_reward_1 = 0

        self.gather_reward_0 = 0
        self.gather_reward_1 = 0

        self.build_reward_0 = 0
        self.build_reward_1 = 0

        self.consumed_energy_0 = 0
        self.consumed_energy_1 = 0

        player_0_actions = actions[0]
        player_1_actions = actions[1]
        
        virtual_actions = dict(player_0 = dict(), player_1 = dict())

        state = self.env.state
        units = state.units
        factories = state.factories

        for agent in virtual_actions.keys():

            for unit in units[agent].values():

                x,y = unit.pos.pos
                unit_id = unit.unit_id
                cargo = unit.cargo
                if agent == "player_0":
                    action = player_0_actions[x][y][:5]
                elif agent == "player_1":
                    action = player_1_actions[x][y][:5]
                else:
                    print("Problem15")
                action = self.__parse_unit_action(action, cargo, [x,y], agent,unit.is_heavy())
                virtual_actions[agent][unit_id] = [action]

            for factory in factories[agent].values():

                x,y = factory.pos.pos
                unit_id = factory.unit_id
                if agent == "player_0":
                    action = player_0_actions[x][y][5:]
                else:
                    action = player_1_actions[x][y][5:]
                action = self.__parse_factory_action(action, agent)
                if action == None:
                    continue
                virtual_actions[agent][unit_id] = action

        obs, rew, done, _ = self.env.step(virtual_actions)

        transfer_dataice_0 = (self.env.state.stats["player_0"]["transfer"]["ice"] - self.last_stats["player_0"]["transfer"]["ice"]) 
        transfer_dataore_0 = (self.env.state.stats["player_0"]["transfer"]["ore"] - self.last_stats["player_0"]["transfer"]["ore"]) 
        transfer_dataice_1 = (self.env.state.stats["player_1"]["transfer"]["ice"] - self.last_stats["player_1"]["transfer"]["ice"])
        transfer_dataore_1 = (self.env.state.stats["player_1"]["transfer"]["ore"] - self.last_stats["player_1"]["transfer"]["ore"]) 
        gathered_dataice_0 = (self.env.state.stats["player_0"]["generation"]["ice"]["HEAVY"] - self.last_stats["player_0"]["generation"]["ice"]["HEAVY"]+\
                              self.env.state.stats["player_0"]["generation"]["ice"]["LIGHT"] - self.last_stats["player_0"]["generation"]["ice"]["LIGHT"])
        gathered_dataore_0 = (self.env.state.stats["player_0"]["generation"]["ore"]["HEAVY"] - self.last_stats["player_0"]["generation"]["ore"]["HEAVY"]+\
                              self.env.state.stats["player_0"]["generation"]["ore"]["LIGHT"] - self.last_stats["player_0"]["generation"]["ore"]["LIGHT"])
        gathered_dataore_1 = (self.env.state.stats["player_1"]["generation"]["ore"]["HEAVY"] - self.last_stats["player_1"]["generation"]["ore"]["HEAVY"]+\
                              self.env.state.stats["player_1"]["generation"]["ore"]["LIGHT"] - self.last_stats["player_1"]["generation"]["ore"]["LIGHT"])
        gathered_dataice_1 = (self.env.state.stats["player_1"]["generation"]["ice"]["HEAVY"] - self.last_stats["player_1"]["generation"]["ice"]["HEAVY"]+\
                              self.env.state.stats["player_1"]["generation"]["ice"]["LIGHT"] - self.last_stats["player_1"]["generation"]["ice"]["LIGHT"])
        destroyed_heavy_0 = (self.env.state.stats["player_0"]["destroyed"]["HEAVY"] - self.last_stats["player_0"]["destroyed"]["HEAVY"])
        destroyed_light_0 = (self.env.state.stats["player_0"]["destroyed"]["LIGHT"] - self.last_stats["player_0"]["destroyed"]["LIGHT"])
        destroyed_heavy_1 = (self.env.state.stats["player_1"]["destroyed"]["HEAVY"] - self.last_stats["player_1"]["destroyed"]["HEAVY"])
        destroyed_light_1 = (self.env.state.stats["player_1"]["destroyed"]["LIGHT"] - self.last_stats["player_1"]["destroyed"]["LIGHT"])
        build_0 = (self.state.stats["player_0"]["generation"]["built"]["HEAVY"] - self.last_stats["player_0"]["generation"]["built"]["HEAVY"])
        build_1 = (self.state.stats["player_1"]["generation"]["built"]["HEAVY"] - self.last_stats["player_1"]["generation"]["built"]["HEAVY"])

        if done["player_0"]:
            player_0_rew = rew["player_0"]
            player_1_rew = rew["player_1"]
            done = 1
            if player_0_rew == -1000:
                if player_1_rew == -1000:
                    done_rew_0 = 0
                    done_rew_1 = 0
                else:
                    done_rew_0 = -10
                    done_rew_1 = 10
            elif player_1_rew == -1000:
                done_rew_0 = 10
                done_rew_1 = -10
            else:
                done_rew_0 = 0
                done_rew_1 = 0
        else:
            done = 0
            done_rew_0 = 0
            done_rew_1 = 0

        factories_0 = len(obs["player_0"]["factories"]["player_0"].keys())
        factories_1 = len(obs["player_1"]["factories"]["player_1"].keys())

        factory_punishment_0, factory_punishment_1 = 0,0

        if factories_0 < self.factory_count_0:
            factory_punishment_0 = self.factory_count_0 - factories_0
            self.factory_count_0 = factories_0
        if factories_1 < self.factory_count_1:
            factory_punishment_1 = self.factory_count_1 - factories_1
            self.factory_count_1 = factories_1

        strain_ids_0 = self.env.state.teams["player_0"].factory_strains
        strain_ids_1 = self.env.state.teams["player_1"].factory_strains

        agent_lichen_mask_0 = np.isin(
                        self.env.state.board.lichen_strains, strain_ids_0
                    )
        agent_lichen_mask_1 = np.isin(
                        self.env.state.board.lichen_strains, strain_ids_1
                    )
        lichen_0 = self.state.board.lichen[agent_lichen_mask_0].sum()
        lichen_1 = self.state.board.lichen[agent_lichen_mask_1].sum()

        lichen_reward_0 = self.__calc_lichen_reward(self.env.state.real_env_steps, lichen_0)
        lichen_reward_1 = self.__calc_lichen_reward(self.env.state.real_env_steps, lichen_1)

        if self.env.state.real_env_steps < 1000:
            done_rew_0 = 0
            done_rew_1 = 0
            lichen_reward_0 = 0
            lichen_reward_1 = 0
        
        reward_0 = transfer_dataice_0 / 1000 + self.gather_reward_0 +\
            transfer_dataore_0 / 500\
                + lichen_reward_0 
        
        reward_1 = transfer_dataice_1 / 1000 + self.gather_reward_1 +\
            transfer_dataore_1 / 500\
                + lichen_reward_1 
    
        infos = dict(gat_i_0 = gathered_dataice_0, gat_i_1 = gathered_dataice_1, tra_i_0 = transfer_dataice_0,\
                     tra_i_1 = transfer_dataice_1, pus_h_0 = destroyed_heavy_0, pus_h_1 = destroyed_heavy_1,\
                    gat_o_0 = gathered_dataore_0, gat_o_1 = gathered_dataore_1, tra_o_1 = transfer_dataore_1,\
                        tra_o_0 = transfer_dataore_0, bui_0 = build_0 , bui_1 = build_1, li_0 = lichen_0, li_1 = lichen_1)
        
        self.factory_map = np.ones((48,48)) * -1
        self.resource_map = np.zeros((48,48))
        self.coll_map = np.zeros((2,48,48))
        self.dig_map = np.zeros((5,48,48))
        self.valid_actions_0 = np.zeros((48,48,44), dtype = np.float32)
        self.valid_actions_1 = np.zeros((48,48,44), dtype = np.float32)
        self.valid_actions_0[:,:,[0,5,11,14,20,40]] = 1
        self.valid_actions_1[:,:,[0,5,11,14,20,40]] = 1

        new_obs = self.__prepare_obs(obs)

        self.last_stats = deepcopy(self.env.state.stats)

        return new_obs, [reward_0 , reward_1] , done, infos
    
    def change_render(self):

        visualizer = self.env.py_visualizer
        def rubble(rubble_d):
            x = (100 - rubble_d) / 100
            x = round(255*x)
            return [x, x, x, 255]
        def ice(rubble_d):
            return [0,0,255,255]
        def ore(rubble_d):
            return [0,0,0,255]
        visualizer.rubble_color = rubble
        visualizer.ice_color = ice
        visualizer.ore_color = ore

    def __calc_lichen_reward(self, timesteps, lichen):

        return 1 / (timesteps - 1010) ** 2 * lichen
    
    def __skip_early_setup(self):

        action1 = dict(player_0 = dict(faction = "AlphaStrike", bid = 0), player_1 = dict(faction = "AlphaStrike", bid = 0))
        obs, _,_,_ = self.env.step(action1)

        board = obs["player_0"]["board"]

        r_map = board["rubble"]
        i_map = board["ice"]
        o_map = board["ore"]

        real_env_step = self.env.state.real_env_steps
        turn = 0
        while real_env_step < 0:
            if turn % 2 == 0:
                potential_spawns = np.array(list(zip(*np.where(obs["player_0"]["board"]["valid_spawns_mask"] == 1))))
                spawn_loc = self.mp_tool.findBestPlace(potential_spawns, i_map, o_map, r_map)
                action = dict(player_0 = dict(spawn=spawn_loc, metal=150, water=150), player_1 = dict())
                obs,_,_,_ = self.env.step(action)
            else:
                potential_spawns = np.array(list(zip(*np.where(obs["player_1"]["board"]["valid_spawns_mask"] == 1))))
                spawn_loc = self.mp_tool.findBestPlace(potential_spawns, i_map, o_map, r_map)
                action = dict(player_1 = dict(spawn=spawn_loc, metal=150, water=150), player_0 = dict())
                obs,_,_,_ = self.env.step(action)
            turn += 1
            real_env_step += 1

        #self.saved_state = deepcopy(self.env.state)
        
        self.last_stats = deepcopy(self.env.state.stats)
        return obs
    
    def __parse_unit_action(self, action, cargo, location, agent, unit_type):

        if action[0] == 0:
            print("BIG PROBLEM")
        elif action[0] == 1:
            direction = action[1] - 1
            return np.array([0, direction, 0, 0, 0, 1])
        elif action[0] == 2:
            if self.dig_map[0,location[0],location[1]] * 100 <= 20 and self.dig_map[0,location[0],location[1]] * 100 > 0:
                if agent == "player_0":
                    self.gather_reward_0 += 1 / 100
                elif agent == "player_1":
                    self.gather_reward_1 += 1 / 100
            return np.array((3,0,0,0,0,1))
        elif action[0] == 3:
            transfer_param = action[2] - 1
            direction = action[3] - 1
            if transfer_param == 0:
                resource_amount = cargo.ice
            elif transfer_param == 1:
                resource_amount = cargo.ore
            else:
                print("Problem6")
            return np.array((1,direction,transfer_param,resource_amount,0,1))
        elif action[0] == 4:
            pickup_param = action[4] * 150
            if not pickup_param > 0:
                print("Problem5")
            return np.array((2,0,4,pickup_param,0,1))
        else:
            print("Problem1")   

    def __parse_factory_action(self, action, agent):

        if action[0] == 0:
            print("Big Problem2")
            return None
        if action[0] == 1:
            return None
        elif action[0] == 2:
            if agent == "player_0":
                self.build_reward_0 += 1
            elif agent == "player_1":
                self.build_reward_1 += 1
            else:
                print("Problem10")
            return 1
        elif action[0] == 3:
            return 2
        else:
            print("Problem2")
    
    def __prepare_obs(self, obs):

        #When we prepare the obs in order to optimize our code we will
        #set necessary datas at first time we achieve the data.

        #Necessary datas dig_map, factory_map units_actions_mask 
        #factory_actions_mask resource_map

        obs = obs["player_0"]
        board = obs["board"]
        teams = obs["teams"]
        units = obs["units"]
        factories = obs["factories"]

        self.rubble_map = board["rubble"]

        #Board datas
        r_step = self.env.state.real_env_steps / 1001
        remain_t = 20 - ((r_step % 50) - 30)
        remain_t = 0 if remain_t < 0 or remain_t > 20 else remain_t / 20

        rubble_map = np.array(board["rubble"], dtype = np.float32) / 100
        timestep_map = np.ones((48,48), dtype = np.float32) * r_step
        remain_map = np.ones((48,48), dtype = np.float32) * remain_t
        ice_map = np.array(board["ice"], dtype = np.int32)
        ore_map = np.array(board["ore"], dtype = np.int32) 
        lichen_map_0, lichen_map_1 = self.__convert_lichen_map(teams, board["lichen"], board["lichen_strains"])

        self.dig_map[0] = rubble_map
        self.dig_map[1] = ice_map
        self.dig_map[2] = ore_map
        self.dig_map[3] = lichen_map_0
        self.dig_map[4] = lichen_map_1

        #First Factories because units' actions' masks depend on factory maps
        factory_water_map_0, factory_metal_map_0, factory_power_map_0 , factory_map_0, \
        factory_water_map_1, factory_metal_map_1, factory_power_map_1 , factory_map_1 = self.__convert_factory_map(factories)

        heavy_ice_map_0,heavy_ore_map_0,heavy_power_map_0,heavy_map_0,\
        heavy_ice_map_1,heavy_ore_map_1,heavy_power_map_1,heavy_map_1, = self.__convert_bot_map(units)
        
        new_obs = np.stack((rubble_map, ice_map, ore_map, timestep_map, remain_map,\
                             factory_water_map_0, factory_metal_map_0, factory_power_map_0 , factory_map_0, lichen_map_0,\
                             heavy_ice_map_0,heavy_ore_map_0,heavy_power_map_0,heavy_map_0,\
                             factory_water_map_1, factory_metal_map_1, factory_power_map_1 , factory_map_1, lichen_map_1,\
                             heavy_ice_map_1,heavy_ore_map_1,heavy_power_map_1,heavy_map_1))

        """per_1 = new_obs / 100
        noise = np.random.rand(31,48,48)
        noise_max = noise.max()
        noise_min = noise.min()
        noise = (noise - noise_max) / (noise_max - noise_min) * 2 - 1
        noise *= per_1
        new_obs += noise
        new_min = new_obs.min()
        new_max = new_obs.max()
        new_obs = (new_obs - new_min) / (new_max - new_min)
        """
        return new_obs
    
    def __convert_bot_map(self, units):

        player_0_units , player_1_units = units["player_0"] , units["player_1"]

        heavy_ice_map_0 = np.zeros((48,48), dtype = np.float32)
        heavy_ore_map_0 = np.zeros((48,48), dtype = np.float32)
        heavy_power_map_0 = np.zeros((48,48), dtype = np.float32)
        heavy_map_0 = np.zeros((48,48), dtype = np.float32)
        heavy_ice_map_1 = np.zeros((48,48), dtype = np.float32)
        heavy_ore_map_1 = np.zeros((48,48), dtype = np.float32)
        heavy_power_map_1 = np.zeros((48,48), dtype = np.float32)
        heavy_map_1 = np.zeros((48,48), dtype = np.float32)

        for unit in player_0_units.values():
            x,y = unit["pos"]
            ice = unit["cargo"]["ice"]
            ore = unit["cargo"]["ore"]
            power = unit["power"]
            if unit["unit_type"] == "HEAVY":
                heavy_ore_map_0[x][y] = ore / 1000
                heavy_power_map_0[x][y] = power / 3000
                heavy_ice_map_0[x][y] = ice / 1000
                heavy_map_0[x][y] = 1
                
        for unit in player_1_units.values():
            x,y = unit["pos"]
            ice = unit["cargo"]["ice"]
            ore = unit["cargo"]["ore"]
            power = unit["power"]
            if unit["unit_type"] == "HEAVY":
                heavy_ore_map_1[x][y] = ore / 1000
                heavy_power_map_1[x][y] = power / 3000
                heavy_ice_map_1[x][y] = ice / 1000
                heavy_map_1[x][y] = 1

        for unit in player_0_units.values():
            self.__find_invalid_actions(unit, "unit", 0,heavy_map_0, heavy_map_1)

        for unit in player_1_units.values():
            self.__find_invalid_actions(unit, "unit", 1,heavy_map_1, heavy_map_0)
            
        return heavy_ice_map_0,heavy_ore_map_0,heavy_power_map_0,heavy_map_0,\
        heavy_ice_map_1,heavy_ore_map_1,heavy_power_map_1,heavy_map_1,
    
    def __convert_lichen_map(self, teams, lichen_map, lichen_strains):

        player_0_strains = teams["player_0"]["factory_strains"]
        player_1_strains = teams["player_1"]["factory_strains"]
        result_lichen_map_0 = np.zeros((48,48), dtype = np.float32)
        result_lichen_map_1 = np.zeros((48,48), dtype = np.float32)
        lichen_map = np.array(lichen_map, dtype = np.float32) / 100
        lichen_strains = np.array(lichen_strains)

        mask0 = np.isin(lichen_strains, player_0_strains)
        lichens_for_0 = lichen_map[mask0]
        result_lichen_map_0[mask0] = lichens_for_0

        mask1 = np.isin(lichen_strains, player_1_strains)
        lichens_for_1 = lichen_map[mask1]
        result_lichen_map_1[mask1] = lichens_for_1 
        return result_lichen_map_0, result_lichen_map_1
    
    def __get_all_factory_cordinates(self, cordinate):

        cordinates = []
        for x in [-1,0,1]:
            for y in [-1,0,1]:
                cordinates += [[cordinate[0] + x, cordinate[1] + y]]
        return cordinates 
    
    def __convert_factory_map(self, factories):

        player_0_factories, player_1_factories = factories["player_0"], factories["player_1"]

        factory_power_map_0 = np.zeros((48,48), dtype = np.float32)
        factory_water_map_0 = np.zeros((48,48), dtype = np.float32)
        factory_metal_map_0 = np.zeros((48,48), dtype = np.float32)
        factory_map_0 = np.zeros((48,48), dtype = np.float32)
        factory_power_map_1 = np.zeros((48,48), dtype = np.float32)
        factory_water_map_1 = np.zeros((48,48), dtype = np.float32)
        factory_metal_map_1= np.zeros((48,48), dtype = np.float32)
        factory_map_1 = np.zeros((48,48), dtype = np.float32)

        for factory in player_0_factories.values():
            self.__find_invalid_actions(factory, "factory", 0,None,None)
            water = factory["cargo"]["water"]
            power = factory["power"]
            metal = factory["cargo"]["metal"]
            water = water / 10000 if water <= 10000 else 1.
            metal = metal / 10000 if metal <= 10000 else 1.
            pos = factory["pos"]
            all_cordinates = self.__get_all_factory_cordinates(pos)
            for x,y in all_cordinates:
                self.factory_map[x][y] = 0
                self.resource_map[x][y] = power
                power = power / 10000 if power <= 10000 else 1.
                factory_water_map_0[x][y] = water 
                factory_metal_map_0[x][y] = metal
                factory_power_map_0[x][y] = power 
                factory_map_0[x][y] = 1

        for factory in player_1_factories.values():
            self.__find_invalid_actions(factory, "factory", 1,None,None)
            water = factory["cargo"]["water"]
            power = factory["power"]
            metal = factory["cargo"]["metal"]
            water = water / 10000 if water <= 10000 else 1.
            metal = metal / 10000 if metal <= 10000 else 1.
            pos = factory["pos"]
            all_cordinates = self.__get_all_factory_cordinates(pos)
            for x,y in all_cordinates:
                self.factory_map[x][y] = 1
                self.resource_map[x][y] = -power
                power = power / 10000 if power <= 10000 else 1.
                factory_water_map_1[x][y] = water 
                factory_metal_map_1[x][y] = metal
                factory_power_map_1[x][y] = power 
                factory_map_1[x][y] = 1
            
        return factory_water_map_0, factory_metal_map_0, factory_power_map_0 , factory_map_0, \
        factory_water_map_1, factory_metal_map_1, factory_power_map_1 , factory_map_1
    
    def __check_nearby_enemy(self, location, enemy_map):

        x,y = location
        if (x < 47 and enemy_map[x+1][y]) or (x > 0 and enemy_map[x-1][y]) or (y < 47 and enemy_map[x][y+1]) or (y > 0 and enemy_map[x][y-1]):
            return True
        return False
    
    def __find_invalid_actions(self, unit, types, player, heavy_map_ally, heavy_map_enemy):
        
        x,y = unit["pos"]
        if types == "factory":
            actions = np.ones((4), dtype = int)
            actions[0] = 0
            heavy, light = self.__check_robot(unit)
            if not heavy:
                actions[2] = 0
            if player == 0:
                if actions[1] == 1:
                    self.coll_map[0][x][y] = 1
                self.valid_actions_0[x][y][-4:] = actions
            else:
                if actions[1] == 1:
                    self.coll_map[1][x][y] = 1
                self.valid_actions_1[x][y][-4:] = actions
        else:
            actions = np.ones((40), dtype = int)
            actions[[0,5]] = 0
            actions[7:11] = self.__check_move_actions(unit, heavy_map_ally)
            actions[2] = self.__check_dig_action(unit)
            actions[3], actions[11:14], actions[14:20] = self.__check_transfer_actions(unit)
            actions[4], actions[20:] = self.__check_pickup_actions(unit)
            if player == 0:
                if self.coll_map[0][x][y] == 1 or self.__check_nearby_enemy([x,y], heavy_map_enemy):
                    actions[6] = 0
                    if (actions[7:11] == 0).all():
                        self.valid_actions_0[x][y][42] = 0
                        actions[6] = 1
                    else:
                        actions[4] = 0
                        actions[21:] = 0
                        actions[20] = 1
                        actions[3] = 0
                        actions[11] = 1
                        actions[12:14] = 0
                        actions[14] = 1
                        actions[15:20] = 0
                self.valid_actions_0[x][y][:40] = actions
            else:
                if self.coll_map[1][x][y] == 1 or self.__check_nearby_enemy([x,y], heavy_map_enemy): 
                    actions[6] = 0
                    if (actions[7:11] == 0).all():
                        self.valid_actions_1[x][y][42] = 0
                        actions[6] = 1
                    else:
                        actions[4] = 0
                        actions[21:] = 0
                        actions[20] = 1
                        actions[3] = 0
                        actions[11] = 1
                        actions[12:14] = 0
                        actions[14] = 1
                        actions[15:20] = 0
                self.valid_actions_1[x][y][:40] = actions

    def __check_transfer_actions(self, unit):
        x,y = unit["pos"]
        team = unit["team_id"]

        action = np.ones((2,5), dtype=int)

        if self.factory_map[x][y] != team:
            action[:,0] = 0
        if y-1 < 0 or self.factory_map[x][y-1] != team:
            action[:,1] = 0
        if x+1 > 47 or self.factory_map[x+1][y] != team:
            action[:,2] = 0
        if y+1 > 47 or self.factory_map[x][y+1] != team:
            action[:,3] = 0
        if x-1 < 0 or self.factory_map[x-1][y] != team:
            action[:,4] = 0
        ice = unit["cargo"]["ice"]
        ore = unit["cargo"]["ore"]

        ice_t,ore_t = 1,1

        if ice < 1:
            action[0,:] = 0
            ice_t = 0
        if ore < 1:
            action[1,:] = 0
            ore_t = 0

        action = action.reshape(-1)
        if (action == 0).all():
            none_t = 1
            none_a_t = 0
            action = [0,0,0,0,0]
        else:
            none_t = 0
            none_a_t = 1
            if not (action[:5] == 0).all():
                action = action[:5].tolist()
            else:
                action = action[5:].tolist()
        return none_a_t, [none_t,ice_t,ore_t], [none_t] + action

    def __check_pickup_actions(self,unit):

        power = unit["power"]
        x,y = unit["pos"]
        unit_type = unit["unit_type"]
        actions = np.ones(19)
        team = -1 if unit["team_id"] == 1 else 1
        if (team == -1 and not self.resource_map[x][y] < 0) or (team == 1 and not self.resource_map[x][y] > 0):
            return 0 , [1] + [0]*19
        
        factory_power = abs(self.resource_map[x][y])
        max_capacity = 3000 if unit_type == "HEAVY" else 300
        possible_amount = max_capacity - power
        multiplier = 150
        
        max_count_index = np.floor(possible_amount / multiplier)
        max_count_index2 = np.floor(factory_power / multiplier)
        max_count_index = int(np.min((max_count_index, max_count_index2)))
        if max_count_index >= 19:
            return 1, [0] + [1]*19
        actions[max_count_index:] = 0
        if max_count_index == 0:
            return 0 , [1] + [0]*19
        return 1, [0] + actions.tolist()

    def __check_dig_action(self, unit):

        team = unit["team_id"]
        x,y = unit["pos"]
        dig_loc = self.dig_map[:,x,y]
        if team == 1 and dig_loc[-1] > 0.:
            return 0
        if team == 0 and dig_loc[-2] > 0.:
            return 0
        if (dig_loc == [0.,0.,0.,0.,0.]).all():
            return 0
        unit_type = unit["unit_type"]
        power = unit["power"]
        if unit_type == "HEAVY":
            dig_power = 60
        elif unit_type == "LIGHT":
            dig_power = 5
        if dig_power > power:
            return 0
        return 1

    def __check_move_actions(self, unit, heavy_map):

        x,y = unit["pos"]
        team = unit["team_id"]
        unit_type = unit["unit_type"]
        unit_power = unit["power"]

        if unit_type == "HEAVY":
            base_power = 20
            per_rubble = 1
        else:
            base_power = 1
            per_rubble = .05

        action1 = 1
        action2 = 1
        action3 = 1
        action4 = 1

        if (y-1) < 0 or (self.factory_map[x][y-1] != -1 and self.factory_map[x][y-1] != team) or\
            self.rubble_map[x][y-1] * per_rubble + base_power > unit_power or self.coll_map[team][x][y-1] == 1 or heavy_map[x][y-1] != 0:
            action1 = 0
        if (x+1) > 47 or (self.factory_map[x+1][y] != -1 and self.factory_map[x+1][y] != team) or\
            self.rubble_map[x+1][y] * per_rubble + base_power > unit_power or self.coll_map[team][x+1][y] == 1 or heavy_map[x+1][y] != 0:
            action2 = 0
        if (y+1) > 47 or (self.factory_map[x][y+1] != -1 and self.factory_map[x][y+1] != team) or\
            self.rubble_map[x][y+1] * per_rubble + base_power > unit_power or self.coll_map[team][x][y+1] == 1 or heavy_map[x][y+1] != 0:
            action3 = 0
        if (x-1) < 0 or (self.factory_map[x-1][y] != -1 and self.factory_map[x-1][y] != team) or\
            self.rubble_map[x-1][y] * per_rubble + base_power > unit_power or self.coll_map[team][x-1][y] == 1 or heavy_map[x-1][y] != 0:
            action4 = 0
        
        if action1 :
            self.coll_map[team][x][y-1] = 1
        if action2 :
            self.coll_map[team][x+1][y] = 1
        if action3 :
            self.coll_map[team][x][y+1] = 1
        if action4 :
            self.coll_map[team][x-1][y] = 1
        
        return action1, action2, action3, action4

    def __check_robot(self, unit):

        cargo = unit["cargo"]
        metal = cargo["metal"]
        power = unit["power"]
        if power >= 500 and metal >= 100:
            return 1,1
        elif power >= 50 and metal >= 10:
            return 0,1
        return 0,0
    
from gym.envs.registration import register

register("TessEnv-v1",
    entry_point="newEnv.TessEnv:TessEnv")