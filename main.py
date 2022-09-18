import argparse
from pathlib import Path
from typing import List

from tqdm import tqdm

from src.data_steam import DataStream, OutputStream
from src.steps import (
    SegmentationStep,
    DetectPointsOnMask,

)
from src.steps import BaseStep, PointsProjection2D, PointsProjection3D, BodyInfoExtractionStep, BodyKeypointsTracking

pipeline: List[BaseStep] = [
    SegmentationStep('model.onnx'),
    DetectPointsOnMask(),
    PointsProjection2D(),
    PointsProjection3D(),
    BodyInfoExtractionStep(),
    # BodyKeypointsTracking()
]


def main(data_dir: str, output_dir: str):
    istream = DataStream(data_dir)
    ostream = OutputStream(Path(output_dir) / Path(data_dir).name)

    for i, sample in tqdm(enumerate(istream), total=len(istream)):
        for step in pipeline:
            sample = step(sample)
        ostream(sample)

    ostream.close()

    print("#" * 50)
    total = 0
    for step in pipeline:
        mean, std, max_ = step.get_time_spent()
        total += mean
        name = type(step).__name__
        print(f"{name}: mean={mean:.4f}s, std={std:.4f}s, max={max_:.4f}s")
    print(f"Total: {total:.4f}s. FPS: {(1 / total):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/20220719_183951/')
    parser.add_argument('--out', type=str, default='out/')
    args = parser.parse_args()

    main(data_dir=args.data, output_dir=args.out)
