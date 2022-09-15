import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data_sample/')
    args = parser.parse_args()

    main(data_dir=args.data)