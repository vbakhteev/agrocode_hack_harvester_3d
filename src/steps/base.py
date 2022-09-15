from abc import ABC, abstractmethod


class BaseStep(ABC):
    def __call__(self, sample: dict) -> dict:
        return self.call(sample)

    @abstractmethod
    def call(self, sample: dict) -> dict:
        pass
