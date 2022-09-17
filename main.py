import argparse
from pathlib import Path
from typing import List

from tqdm import tqdm

from src.data_steam import DataStream, OutputStream
from src.steps import (
    BaseStep,
    PointsProjection2D,
    SegmentationStep,
    DetectPointsOnMask,
    PointsProjection3D,

)
from src.steps import BaseStep, PointsDetection2d, PointsProjection2D, PointsProjection3D, BodyInfoExtractionStep

pipeline: List[BaseStep] = [
    SegmentationStep('model.onnx'),
    DetectPointsOnMask(),
    PointsProjection2D(),
    PointsProjection3D(),
    BodyInfoExtractionStep()
]


def main(data_dir: str, output_dir: str):
    istream = DataStream(data_dir)
    ostream = OutputStream(Path(output_dir) / Path(data_dir).name)

    for i, sample in tqdm(enumerate(istream), total=len(istream)):
        for step in pipeline:
            sample = step(sample)
        ostream(sample)

    ostream.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/20220719_183951/')
    parser.add_argument('--out', type=str, default='out/')
    args = parser.parse_args()

    main(data_dir=args.data, output_dir=args.out)
