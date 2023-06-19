import collections
import itertools
import numpy as np
import ray
replay = collections.namedtuple("replay",("obs","rewards","actions"))

obs = np.random.rand(23,48,48)
reward = np.random.rand(1)
actions = np.random.rand(10)

transition = replay(obs,reward,actions)

print(transition)

class Buffer():

    def __init__(self):

        self.replay = collections.deque([])
        self.transition = collections.namedtuple("Transition",("obs","action","mask","logprob","reward","done"))

    def push(self, *args):

        self.replay.append(self.transition(*args))

    def sample(self, batch_size):

        return np.array(itertools.islice(self.replay, batch_size))
    
    def __len__(self):

        return len(self.replay)
    

memory = Buffer()

N_STEPS = 128
for step in range(N_STEPS):

    obs = np.random.rand(23,48,48)
    action = np.random.rand(48,48,6)
    mask = np.random.rand(48,48,46)
    logprob = np.random.rand(1)
    reward = np.random.rand(1)
    done = np.random.rand(1)
    memory.push(obs,action,mask,logprob,reward,done)

data = memory.sample(64)
print(data)


