import random 
from collections import deque, namedtuple
import itertools
import torch 
torch.manual_seed(42)
random.seed(a=42)

# Tuple for storing experience steps in the replay memory
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'prev_action'))

class ReplayMemory(object):
    """
    ReplayMemory is a cyclic buffer of bounded size that holds the transitions observed during interaction with the environment.
    It allows to sample random batches of transitions for training the neural network.

    Args:
        memory_size (int): The maximum number of transitions to store in the memory.
        batch_size (int): The number of transitions to sample in each training batch.
        reccurent (bool, optional): If True, sample sequential batches for recurrent networks. Defaults to False.
        env_copy (bool, optional): If True, returns batches with non-batched actions. Defaults to False.
    """    
    def __init__(self, memory_size, batch_size, reccurent = False, env_copy = False):

        self.memory_size = memory_size
        self.memory = deque([], maxlen=self.memory_size)
        self.batch_size = batch_size 
        self.reccurent = reccurent
        self.env_copy = env_copy

    def push(self, *args):

        """
        Save a transition into the replay memory.

        Args:
            *args: Transition components (state, action, reward, prev_action).
        """
        
        self.memory.append(Transition(*args))

    def sample(self):

        """
        Sample a random batch of transitions from the memory.

        Returns:
            list: A list of randomly sampled transitions.
        """

        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)

    def get_batch(self):

        """
        Retrieve a batch of transitions for training.

        Returns:
            tuple: A tuple containing batches of states, actions, rewards, and previous actions.
        """

        if len(self.memory) < self.batch_size:
            return 
        if self.batch_size < 2:
            raise ValueError('Argumnet batch_size must be >= 2')
        if self.reccurent:
            batch = self.get_sequential_batch()
        else:
            batch = self.sample()
        batch = Transition(*zip(*batch))

        if self.env_copy:
            return torch.cat(batch.state), batch.action, torch.cat(batch.reward), torch.cat(batch.prev_action)
        else:
            return torch.cat(batch.state), torch.cat(batch.action) , torch.cat(batch.reward), torch.cat(batch.prev_action)
    
    def get_sequential_batch(self):

        """
        Retrieve a sequential batch of transitions for recurrent networks.

        Returns:
            list: A list of sequential transitions.
        """

        latest_start = len(self.memory) - self.batch_size
        random_start = random.randint(0, latest_start)
        batch = list(itertools.islice(self.memory, random_start, (random_start + self.batch_size)))
        none_index = [i for i,v in enumerate(batch) if v[0] == None]
        if none_index != []:
            if none_index[0] > 1:
                batch = batch[:none_index[0]]
            else:
                batch = batch[(none_index[0]+1):]
                
        return batch 

