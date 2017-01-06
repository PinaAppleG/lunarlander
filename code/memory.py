import numpy as np

class ReplayMemory(object):

    def __init__(self,N):
        self.data = []  # Stores the transitions
        self.n = 0  # Stores the number of elements in the Replay Memory
        self.N = N # Max number of elements in the Replay Memory
        self.terminal = []  # Stores if game is terminal or not
        
    def append(self, transition, terminal):
        if self.n<self.N:
            self.data.append(transition)
            self.terminal.append(terminal)
            self.n = self.n + 1
        else:
            idx = np.random.randint(self.N)
            self.data[idx] = transition
            self.terminal[idx] = terminal
    
    def mini_batch(self,size):
        '''
        Outputs minibatch of size 'size'. If size is above size of data, it will only output a batch of total size
        '''
        idxes = np.random.randint(self.N,size = size)
        try:
            batch = [self.data[i] for i in idxes]
            batch_state = [self.terminal[i] for i in idxes]
        except IndexError:
            batch = self.data
            batch_state = self.terminal
        return batch, batch_state
