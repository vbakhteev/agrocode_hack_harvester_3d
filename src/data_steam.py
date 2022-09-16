from pathlib import Path

from src.io import read_image, read_json, read_ply, WriteVideoStreamImageio


class DataStream:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.paths = list(self.data_dir.glob('package_*'))
        self.paths = sorted(
            self.paths, key=lambda p: int(p.stem.split('_')[1])
        )

    def __len__(self):
        return len(self.paths)

    def __iter__(self):
        for sample_dir in self.paths:
            color_frame = read_image(sample_dir / 'color_frame.png')
            meta = read_json(sample_dir / 'meta.txt')
            point_cloud = read_ply(sample_dir / 'point_cloud.ply')

            sample = {
                "color_frame": color_frame,
                "meta": meta,
                "point_cloud": point_cloud,
            }
            yield sample


class OutputStream:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.video_stream = WriteVideoStreamImageio(
            path=self.data_dir / 'color_video.mp4',
            fps=10,
        )

    def __call__(self, sample: dict) -> None:
        self.video_stream(sample['color_frame'])

    def close(self) -> None:
        self.video_stream.close()
