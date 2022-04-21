from collections import deque
import random
import numpy as np


class Replaymemory:
    def __init__(self, buffer_size = 50000):
        self.buffer = deque()
        self.buffer_size = buffer_size


    def append(self, item):
        if len(self.buffer) > self.buffer_size:
            self.buffer.popleft()
        self.buffer.append(item)

    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        batch_state, batch_action, batch_reward, batch_state_new, batch_over = [], [], [], [], []
        
        for b in batch:
            batch_state.append(b[0][0])
            batch_action.append(b[1])
            batch_reward.append(b[2])
            batch_state_new.append(b[3][0])
            batch_over.append(float(b[4]))
        
        batch_state = np.stack(batch_state)
        batch_action = np.stack(batch_action)
        batch_reward = np.stack(batch_reward)
        batch_state_new = np.stack(batch_state_new)
        batch_over = np.stack(batch_over)
        
        return batch_state, batch_action, batch_reward, batch_state_new, batch_over
        
