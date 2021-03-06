import random
import numpy as np

from collections import deque
from .sumtree import SumTree

class MemoryBuffer(object):
    """ Memory Buffer Helper class for Experience Replay
    using a double-ended queue or a Sum Tree (for PER)
    """
    def __init__(self, buffer_size, with_per = False):
        """ Initialization
        """
        if(with_per):
            # Prioritized Experience Replay
            self.alpha = 0.5
            self.epsilon = 0.01
            self.buffer = SumTree(buffer_size)
        else:
            # Standard Buffer
            self.buffer = deque()
        self.count = 0
        self.with_per = with_per
        self.buffer_size = buffer_size

    def memorize(self, *data):
        """ Save an experience to memory, optionally with its TD-Error
        """

        # experience = (state, state2, action, reward, done, new_state1, new_state2)

        if self.count < self.buffer_size:
            self.buffer.append(data)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(data)

    def priority(self, error):
        """ Compute an experience priority, as per Schaul et al.
        """
        return (error + self.epsilon) ** self.alpha

    def size(self):
        """ Current Buffer Occupation
        """
        return self.count

    def sample_batch(self, batch_size):
        """ Sample a batch, optionally with (PER)
        """
        batch = []

        # Sample randomly from Buffer
        if self.count < batch_size:
            idx = None
            batch = random.sample(self.buffer, self.count)
        else:
            idx = None
            batch = random.sample(self.buffer, batch_size)

        # Return a batch of experience
        s_batch = np.array([i[0] for i in batch])
        s1_batch = np.array([i[1] for i in batch])
        a_batch = np.array([i[2] for i in batch])
        r_batch = np.array([i[3] for i in batch])
        d_batch = np.array([i[4] for i in batch])
        new_s_batch = np.array([i[5] for i in batch])
        new_s1_batch = np.array([i[6] for i in batch])
        return s_batch, s1_batch, a_batch, r_batch, d_batch, new_s_batch, new_s1_batch, idx

    def update(self, idx, new_error):
        """ Update priority for idx (PER)
        """
        self.buffer.update(idx, self.priority(new_error))

    def clear(self):
        """ Clear buffer / Sum Tree
        """
        if(self.with_per): self.buffer = SumTree(buffer_size)
        else: self.buffer = deque()
        self.count = 0
