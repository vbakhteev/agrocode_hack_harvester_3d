from pathlib import Path

import cv2
import numpy as np
import pandas as pd

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

    def __getitem__(self, idx):
        path = self.paths[idx]
        return self._process_path(path)

    def __iter__(self):
        for sample_dir in self.paths:
            sample = self._process_path(sample_dir)
            yield sample

    def _process_path(self, sample_dir):
        color_frame = read_image(sample_dir / 'color_frame.png')
        meta = read_json(sample_dir / 'meta.txt')
        point_cloud = read_ply(sample_dir / 'point_cloud.ply')
        sample = {
            "video_id": self.data_dir.name,
            "package_id": sample_dir.stem,
            "color_frame": color_frame,
            "meta": meta,
            "point_cloud": point_cloud,
        }
        return sample


class OutputStream:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.video_stream = WriteVideoStreamImageio(
            path=self.data_dir / 'color_video.mp4',
            fps=10,
        )

        self.results = []

    def __call__(self, sample: dict) -> None:
        # image_draw = sample['color_frame'].copy()
        # center = (
        #     np.floor(sample['center'][1] * sample['meta']['intrinsics']["height"]).astype(int),
        #     np.floor(sample['center'][0] * sample['meta']['intrinsics']["width"]).astype(int)
        # )
        # cv2.circle(image_draw, center, 10, (255, 0, 0), thickness=-1)
        #
        # for p in sample["keypoints_2d_reconstructed"]:
        #     center = (
        #         np.floor(p[1] * sample['meta']['intrinsics']["height"]).astype(int),
        #         np.floor(p[0] * sample['meta']['intrinsics']["width"]).astype(int)
        #     )
        #     cv2.circle(image_draw, center, 10, (255, 0, 0), thickness=-1)
        #
        # cv2.putText(
        #     image_draw, f"l={round(sample['length'], 4)}, w={round(sample['width'], 4)}", (50, 50),
        #     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
        # )
        #
        # if sample.get("tracked_coords") is not None:
        #     cv2.putText(
        #         image_draw,
        #         f"+,w={round(sample.get('full_body_width', 0), 4)},l={round(sample.get('full_body_length', 0), 4)}",
        #         (50, 100),
        #         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
        #     )
        # else:
        #     cv2.putText(
        #         image_draw,
        #         f"-,w={round(sample.get('full_body_width', 0), 4)},l={round(sample.get('full_body_length', 0), 4)}",
        #         (50, 100),
        #         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
        #     )
        #
        #
        # self.video_stream(image_draw)

        center_to_log = sample["center"] / np.array([sample['meta']['intrinsics']["height"], sample['meta']['intrinsics']["width"]]) * np.array([sample['meta']['intrinsics']["width"], sample['meta']['intrinsics']["height"]])

        sample_info = {
            "package_id": sample["package_id"],
            "board_point_x": center_to_log[1],
            "board_point_y": center_to_log[0],
            "width": sample['width'],
            "height": sample['length']
        }

        self.results.append(sample_info)

    def close(self) -> None:
        self.video_stream.close()
        df_results = pd.DataFrame(self.results)
        df_results.to_csv(self.data_dir / 'submission.csv', index=False)
