import random
import numpy as np

from collections import deque
from .sumtree import SumTree


class MemoryBuffer(object):
    """ Memory Buffer Helper class for Experience Replay
    using a double-ended queue or a Sum Tree (for PER)
    """
    def __init__(self, buffer_size, with_per=False):
        """ Initialization
        """
        if with_per:
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

    def memorize(self, state1, state2, action, reward, done, new_state1, new_state2, error=None):
        """ Save an experience to memory, optionally with its TD-Error
        """

        experience = (state1, state2, action, reward, done, new_state1, new_state2)
        if self.with_per:
            priority = self.priority(error[0])
            self.buffer.add(priority, experience)
            self.count += 1
            # print("Count {}".format(self.count))
        else:
            # Check if buffer is already full
            if self.count < self.buffer_size:
                self.buffer.append(experience)
                self.count += 1
            else:
                self.buffer.popleft()
                self.buffer.append(experience)

    def clear_deqeue(self):
        self.buffer.clear()

    def priority(self, error):
        """ Compute an experience priority, as per Schaul et al.
        """
        return (error + self.epsilon) ** self.alpha

    def size(self):
        """ Current Buffer Occupation
        """
        return self.buffer.total()

    def sample_batch(self, batch_size):
        """ Sample a batch, optionally with (PER)
        """
        batch = []
        # Sample using prorities
        if self.with_per:
            T = self.buffer.total() // batch_size
            for i in range(batch_size):
                a, b = T * i, T * (i + 1)
                s = random.uniform(a, b)
                idx, error, data = self.buffer.get(s)
                batch.append((*data, idx))
            idx = np.array([i[7] for i in batch])
        # Sample randomly from Buffer
        elif self.count < batch_size:
            idx = None
            batch = random.sample(self.buffer, self.count)
        else:
            idx = None
            batch = random.sample(self.buffer, batch_size)

        # batch = self.buffer
        # Return a batch of experience
        s1_batch = np.array([i[0] for i in batch])
        s2_batch = np.array([i[1] for i in batch])
        a_batch = np.array([i[2] for i in batch])
        r_batch = np.array([i[3] for i in batch])
        d_batch = np.array([i[4] for i in batch])
        new_s1_batch = np.array([i[5] for i in batch])
        new_s2_batch = np.array([i[6] for i in batch])
        return s1_batch, s2_batch, a_batch, r_batch, d_batch, new_s1_batch, new_s2_batch, idx

    def update(self, idx, new_error):
        """ Update priority for idx (PER)
        """
        self.buffer.update(idx, self.priority(new_error))

    def clear(self):
        """ Clear buffer / Sum Tree
        """
        if self.with_per:
            self.buffer = SumTree(self.buffer_size)
        else:
            self.buffer = deque()
        self.count = 0
