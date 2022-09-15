from pathlib import Path

from src.io import read_image, read_json


class DataStream:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        pass

    def __iter__(self):
        paths = list(self.data_dir.glob('package_*'))
        paths = sorted(
            paths, key=lambda p: int(p.stem.split('_')[1])
        )

        for sample_dir in paths:
            color_frame = read_image(sample_dir / 'color_frame.png')
            meta = read_json(sample_dir / 'meta.txt')
            point_cloud = None

            sample = {
                "color_frame": color_frame,
                "meta": meta,
                "point_cloud": point_cloud,
            }
            yield sample
