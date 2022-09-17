from dataclasses import dataclass, field
import math

import numpy as np
import typing as tp


@dataclass
class HistoryBuffer:
    end_ratio: float = 0.1
    max_memory: int = 3
    _data: tp.List[float] = field(default_factory=list)

    def is_initialized(self) -> bool:
        return len(self._data) > 0

    def add_value(self, value: float) -> None:
        self._data.append(value)
        if len(self._data) > self.max_memory:
            self._data = self._data[1:]

    @property
    def value(self) -> float:
        assert self.is_initialized()
        return np.mean(self._filter_extra_values(self._data, self.end_ratio))

    @staticmethod
    def _filter_extra_values(values: tp.List[float], end_ratio: float) -> tp.List[float]:
        total_elements_from_end = math.floor(len(values) * end_ratio)
        return sorted(values)[total_elements_from_end: len(values)-total_elements_from_end]
