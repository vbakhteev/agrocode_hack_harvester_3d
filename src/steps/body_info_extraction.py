from collections import defaultdict
from dataclasses import dataclass
import typing as tp

import numpy as np

from .base import BaseStep
from ..geometry import final_point_rectangle, check_point_is_outside, project_to_2d
from ..tracking_utils import HistoryBuffer


@dataclass
class CenterHistoryBuffer(HistoryBuffer):
    max_memory = 10

    @property
    def value(self) -> float:
        assert self.is_initialized()
        return np.mean(self._data, axis=0)


class BodyInfoExtractionStep(BaseStep):
    def __init__(self) -> None:
        self.center_history = defaultdict(CenterHistoryBuffer)
        self.width_history = defaultdict(lambda: HistoryBuffer(max_memory=10))
        self.length_history = defaultdict(lambda: HistoryBuffer(max_memory=10))

    def call(self, sample: dict) -> dict:
        # Reconstruct points
        image_height, image_width = sample["meta"]["intrinsics"]['height'], sample["meta"]["intrinsics"]['width']

        # TODO: Kalman filter continuation
        if not sample["2d_points"].any():
            sample['center'] = [0, 0]
            sample["length"] = 0
            sample["width"] = 0
            return sample

        points_on_the_border_status = [
            check_point_is_outside(point, max_y=image_height, max_x=image_width)
            for point in sample["2d_points"][[0, 3, 2, 1]]
        ]
        keypoints_3d_reconstructed = sample["keypoints_3d"]
        keypoints_3d_reconstructed_filtered = keypoints_3d_reconstructed


        # # TODO: мб несколько раз запускать для большей робастности
        # for _ in range(0):
        #     keypoints_3d_reconstructed = self._reconstruct_external_3d_points(
        #         keypoints_3d_reconstructed, points_on_the_border_status
        #     )
        #     keypoints_3d_reconstructed_filtered = [
        #         point
        #         if not check_point_is_outside(project_to_2d(rec_point, sample['meta']), max_y=image_height,
        #                                       max_x=image_width)
        #         else
        #         rec_point
        #         for point, rec_point in zip(sample["keypoints_3d"], keypoints_3d_reconstructed)
        #     ]

        # Measure parameters
        center_1 = (keypoints_3d_reconstructed_filtered[0] + keypoints_3d_reconstructed_filtered[2]) / 2
        center_2 = (keypoints_3d_reconstructed_filtered[1] + keypoints_3d_reconstructed_filtered[3]) / 2
        center_3d = (center_1 + center_2) / 2

        keypoints_2d_reconstructed_filtered = [
            project_to_2d(kp, sample['meta']) / np.array([image_height, image_width]) for kp in keypoints_3d_reconstructed_filtered
        ]

        width_1 = np.linalg.norm(keypoints_2d_reconstructed_filtered[0] - keypoints_2d_reconstructed_filtered[1])
        width_2 = np.linalg.norm(keypoints_2d_reconstructed_filtered[2] - keypoints_2d_reconstructed_filtered[3])

        length_1 = np.linalg.norm(keypoints_2d_reconstructed_filtered[1] - keypoints_2d_reconstructed_filtered[2])
        length_2 = np.linalg.norm(keypoints_2d_reconstructed_filtered[0] - keypoints_2d_reconstructed_filtered[3])

        sample["keypoints_on_the_border_status"] = np.array(points_on_the_border_status)
        sample["keypoints_3d_reconstructed"] = np.stack(keypoints_3d_reconstructed_filtered)
        sample["keypoints_2d_reconstructed"] = np.stack(keypoints_2d_reconstructed_filtered) * np.array([image_height, image_width]) / np.array([image_width, image_height])

        cur_length = max(length_1, length_2)
        cur_length_buffer = self.length_history[sample["video_id"]]
        cur_length_buffer.add_value(cur_length)
        sample["length"] = 0.7 * cur_length_buffer.value + 0.3 * cur_length

        cur_width = max(width_1, width_2)
        cur_width_buffer = self.width_history[sample["video_id"]]
        cur_width_buffer.add_value(cur_width)
        sample["width"] = 0.7 * cur_width_buffer.value + 0.3 * cur_width

        cur_center_buffer = self.center_history[sample["video_id"]]
        cur_center_buffer.add_value(center_3d)

        adjusted_center = 0.7 * cur_center_buffer.value + 0.3 * center_3d

        sample["center"] = project_to_2d(adjusted_center, sample['meta']) / np.array([image_width, image_height])
        return sample

    @staticmethod
    def _reconstruct_external_3d_points(
            keypoints_3d: tp.List[np.ndarray], points_on_the_border_status: tp.List[bool]
    ) -> tp.List[np.ndarray]:
        reconstructed_points = []
        # стоим напротив грузовика, смотрим на лобовое стекло. Перед - первая стенка, зад - вторая.
        # front_left, front_right, bottom_right, bottom_left - порядок точек
        for i, (cur_point, cur_out_status) in enumerate(zip(keypoints_3d, points_on_the_border_status)):
            if not cur_out_status:
                reconstructed_points.append(cur_point)
                continue

            prev_point = keypoints_3d[i - 1]
            next_point = keypoints_3d[(i + 1) % len(keypoints_3d)]
            next_next_point = keypoints_3d[(i + 2) % len(keypoints_3d)]

            reconstructed_point = final_point_rectangle(prev_point, next_point, next_next_point)
            reconstructed_points.append(reconstructed_point)
        return reconstructed_points
