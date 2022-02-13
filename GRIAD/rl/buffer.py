
from collections import deque, namedtuple
import jax
from functools import partial
from jax import jit
import random
import numpy as np

transition_namedtuple = namedtuple("TransitionBatch", 
                                    ["S", "A", "R", "Done", "S_next", "logP"])
                                    
class TransitionBatch():

    def __init__(self, S, A, R, Done, S_next, logP=0):

        self.S = S
        self.A = A
        self.R = R
        self.Done = Done
        self.S_next = S_next
        self.logP = logP
    
    def _to_named_tuple(self):
        return transition_namedtuple(self.S, self.A, self.R, self.Done, self.S_next, self.logP)
    
    @classmethod
    def _from_singles(cls, s, a, r, done, s_next, logp=0):
        return cls(s, a, r, done, s_next, logp)

    
class Buffer():

    def __init__(self, capacity):
        self.capacity = capacity
        self.clear()
    
    def clear(self):
        self.storage = deque(maxlen=self.capacity)
    
    def add(self, transition: TransitionBatch):
        self.storage.extend([transition._to_named_tuple()])
    
    def sample(self, batch_size):
        transitions = random.sample(self.storage, batch_size)
        return jax.tree_multimap(lambda *leaves: np.stack(leaves), *transitions)
    
    def __str__(self):
        return str(self.storage)

    def __len__(self):
        return len(self.storage)