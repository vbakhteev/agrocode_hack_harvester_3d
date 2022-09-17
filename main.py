import argparse
from typing import List

from tqdm import tqdm

from src.data_steam import DataStream, OutputStream
from src.steps import BaseStep, PointsDetection2d, PointsProjection2D
import matplotlib.pyplot as plt

pipeline: List[BaseStep] = [
    # PointsProjection2D(),
    PointsDetection2d(),

]


def main(data_dir: str, output_dir: str):
    istream = DataStream(data_dir)
    ostream = OutputStream(output_dir)

    for sample in tqdm(istream, total=len(istream)):
        for step in pipeline:
            sample = step(sample)
        ostream(sample)

    ostream.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data_sample/')
    parser.add_argument('--out', type=str, default='out/')
    args = parser.parse_args()

    main(data_dir=args.data, output_dir=args.out)
