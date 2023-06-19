import torch

class mapSetupTool:


    def __init__(self):

        rubble_kernel = torch.tensor(((0.0,.11,.11,.11,.11,.11,.11,.11,.11,.11,.11,.11,0.0),
                                        (.11,.11,.25,.25,.25,.25,.25,.25,.25,.25,.25,.11,.11),
                                        (.11,.25,.25,.40,.40,.40,.40,.40,.40,.40,.25,.25,.11),
                                        (.11,.25,.40,.40,.64,.64,.64,.64,.64,.40,.40,.25,.11),
                                        (.11,.25,.40,.64,.64,1.0,1.0,1.0,.64,.64,.40,.25,.11),
                                        (.11,.25,.40,.64,1.0,0.0,0.0,0.0,1.0,.64,.40,.25,.11),
                                        (.11,.25,.40,.64,1.0,0.0,0.0,0.0,1.0,.64,.40,.25,.11),
                                        (.11,.25,.40,.64,1.0,0.0,0.0,0.0,1.0,.64,.40,.25,.11),
                                        (.11,.25,.40,.64,.64,1.0,1.0,1.0,.64,.64,.40,.25,.11),
                                        (.11,.25,.40,.40,.64,.64,.64,.64,.64,.40,.40,.25,.11),
                                        (.11,.25,.25,.40,.40,.40,.40,.40,.40,.40,.25,.25,.11),
                                        (.11,.11,.25,.25,.25,.25,.25,.25,.25,.25,.25,.11,.11),
                                        (0.0,.11,.11,.11,.11,.11,.11,.11,.11,.11,.11,.11,0.0))).reshape(1,1,13,13)
        ice_ore_kernel = torch.tensor(((0,0,0,0,0,0,0,0,0,0,0,0,0),
                                    (0,0,0,0,0,0,0,0,0,0,0,0,0),
                                    (0,0,0,0,0,0,0,0,0,0,0,0,0),
                                    (0,0,0,0,0,1,1,1,0,0,0,0,0),
                                    (0,0,0,0,1,1,1,1,1,0,0,0,0),
                                    (0,0,0,1,1,0,0,0,1,1,0,0,0),
                                    (0,0,0,1,1,0,0,0,1,1,0,0,0),
                                    (0,0,0,1,1,0,0,0,1,1,0,0,0),
                                    (0,0,0,0,1,1,1,1,1,0,0,0,0),
                                    (0,0,0,0,0,1,1,1,0,0,0,0,0),
                                    (0,0,0,0,0,0,0,0,0,0,0,0,0),
                                    (0,0,0,0,0,0,0,0,0,0,0,0,0),
                                    (0,0,0,0,0,0,0,0,0,0,0,0,0)), dtype=torch.float32).reshape(1,1,13,13)
        self.rubble_kernel = torch.nn.Conv2d(1,1,kernel_size = 13,bias = False)
        self.ice_ore_kernel = torch.nn.Conv2d(1,1,kernel_size = 13,bias = False,padding = 6)
        self.rubble_kernel.weight.data = rubble_kernel
        self.ice_ore_kernel.weight.data = ice_ore_kernel
    def findBestPlace(self, suitableCordinates, iceMap, oreMap, rubbleMap):

        ice_map = torch.tensor(iceMap, dtype = torch.float32).reshape(1,1,48,48)
        ore_map = torch.tensor(oreMap, dtype = torch.float32).reshape(1,1,48,48)
        rubble_map = torch.tensor(rubbleMap, dtype = torch.float32).reshape(1,1,48,48) / 100

        rubble_map = torch.nn.functional.pad(rubble_map, (6,6,6,6), 'constant', 1)
        
        ice_result = self.ice_ore_kernel(ice_map)
        ore_result = self.ice_ore_kernel(ore_map) 
        rubble_result = self.rubble_kernel(rubble_map)

        ice_map2 = torch.nn.functional.pad(ice_map, (6,6,6,6), 'constant', 0)
        ore_map2 = torch.nn.functional.pad(ore_map, (6,6,6,6), 'constant', 0)
        optional_ice_result = self.rubble_kernel(ice_map2)
        optional_ore_result = self.rubble_kernel(ore_map2)

        ice_mask = ice_result < 1
        ice_result[ice_mask] = -1000
        ice_mask = ice_result > 1
        ice_result[ice_mask] = 1
        ore_mask = ore_result < 1
        ore_result[ore_mask] = -500
        ore_mask = ore_result > 1
        ore_result[ore_mask] = 1

        result_map = (ice_result + ore_result - rubble_result +  optional_ice_result + optional_ore_result).reshape(48,48)

        max = -1e+8
        max_point = [0,0]

        for x,y in suitableCordinates:

            if result_map[x][y] > max:
                max = result_map[x][y]
                max_point = [x,y]

        return max_point