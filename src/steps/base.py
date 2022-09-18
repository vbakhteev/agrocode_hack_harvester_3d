import time
from abc import ABC, abstractmethod

import numpy as np


class BaseStep(ABC):
    def __init__(self):
        self.time = []

    def __call__(self, sample: dict) -> dict:
        start = time.time()
        sample = self.call(sample)
        end = time.time()
        self.time.append(end - start)
        return sample

    @abstractmethod
    def call(self, sample: dict) -> dict:
        pass

    def get_time_spent(self):
        mean = np.mean(self.time)
        std = np.mean(self.time)
        return mean, std, np.max(self.time)