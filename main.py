from typing import List

from src.data_steam import DataStream
from src.steps import BaseStep, PointsDetection2d


pipeline: List[BaseStep] = [
    PointsDetection2d(),
]


def main(data_dir: str):
    stream = DataStream(data_dir)
    
    for sample in stream:
        for step in pipeline:
            sample = step(sample)


if __name__=='__main__':
    data_dir = 'data_sample'
    main(data_dir)