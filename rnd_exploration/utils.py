import numpy as np
from collections import deque
import dill

def compare(a, b):

    els = np.sum(a != b)
    total = a.size
    per = 100 * els / total
    return els



class RunningAverage:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)

    def update(self, value):
        self.values.append(value)

    @property
    def average(self):
        if len(self.values) > 0 :
            return sum(self.values) / len(self.values)
        return 0.0

    def reset(self):
        self.values.clear()
        
size = 19
with open('configs/train.pl', 'rb') as file:
    train_config = dill.load(file)

with open('configs/test_reachable.pl', 'rb') as file:
    test_config = dill.load(file)

with open('configs/validation_unreachable.pl', 'rb') as file:
    val_config = dill.load(file)
