import math
from collections import defaultdict
from dataclasses import dataclass, field
import typing as tp

import numpy as np

from .base import BaseStep
from ..geometry import final_point_rectangle, reconstruct_rectangle_by_neighbour_points, reconstruct_points


@dataclass
class DimensionsInfoTracker:
    end_ratio: float = 0.1
    _max_memory: int = 100
    _lengths_history: tp.List[float] = field(default_factory=list)
    _widths_history: tp.List[float] = field(default_factory=list)

    def is_initialized(self) -> bool:
        return len(self._lengths_history) > 0 and len(self._widths_history) > 0

    def add_length(self, length: float) -> None:
        self._lengths_history.append(length)
        if len(self._lengths_history) > self._max_memory:
            self._lengths_history = self._lengths_history[1:]

    def add_width(self, width: float) -> None:
        self._widths_history.append(width)
        if len(self._widths_history) > self._max_memory:
            self._widths_history = self._widths_history[1:]

    @property
    def length(self) -> float:
        return np.mean(self._filter_extra_values(self._lengths_history, self.end_ratio))

    @property
    def width(self) -> float:
        return np.mean(self._filter_extra_values(self._widths_history, self.end_ratio))

    @staticmethod
    def _filter_extra_values(values: tp.List[float], end_ratio: float) -> tp.List[float]:
        total_elements_from_end = math.floor(len(values) * end_ratio)
        return sorted(values)[total_elements_from_end: -total_elements_from_end]


class BodyKeypointsTracking(BaseStep):
    def __init__(self):
        self.body_info = defaultdict(DimensionsInfoTracker)

    # TODO: implement 3d coords
    # TODO: может надо как-то восстановленное объединять с предыдущим восстановленным
    # TODO: можно инициализироваться по 2 точкам
    def call(self, sample: dict) -> dict:
        cur_session_info = self.body_info[sample["video_id"]]

        points_on_the_border_status = sample["keypoints_on_the_border_status"]
        keypoints_3d_reconstructed_filtered = np.stack([
            coord[:2] for coord in sample["keypoints_3d_reconstructed"]
        ])
        sample["tracked_coords"] = None
        if sum(points_on_the_border_status) == 0:
            sample["tracked_coords"] = keypoints_3d_reconstructed_filtered
        elif sum(points_on_the_border_status) == 1:
            idx = np.where(points_on_the_border_status)[0][0]

            prev_point = keypoints_3d_reconstructed_filtered[idx - 1]
            next_point = keypoints_3d_reconstructed_filtered[(idx + 1) % len(keypoints_3d_reconstructed_filtered)]
            next_next_point = keypoints_3d_reconstructed_filtered[(idx + 2) % len(keypoints_3d_reconstructed_filtered)]

            final_point = final_point_rectangle(prev_point, next_point, next_next_point)
            tracked_coords = keypoints_3d_reconstructed_filtered
            tracked_coords[idx] = final_point

            sample["tracked_coords"] = tracked_coords

        elif sum(points_on_the_border_status) == 2 and cur_session_info.is_initialized():
            full_body_length, full_body_width = cur_session_info.length, cur_session_info.width

            if points_on_the_border_status[0] and points_on_the_border_status[1]:
                ref_point3, ref_point4 = keypoints_3d_reconstructed_filtered[2], keypoints_3d_reconstructed_filtered[3]
                sample["tracked_coords"] = reconstruct_rectangle_by_neighbour_points(ref_point3, ref_point4, full_body_length, 2)
            elif points_on_the_border_status[0] and points_on_the_border_status[2]:
                ratio = full_body_width / full_body_length
                ref_point2, ref_point4 = keypoints_3d_reconstructed_filtered[1], keypoints_3d_reconstructed_filtered[3]
                sample["tracked_coords"] = reconstruct_points(ref_point2, ref_point4, ratio, offset=1)
            elif points_on_the_border_status[0] and points_on_the_border_status[3]:
                ref_point2, ref_point3 = keypoints_3d_reconstructed_filtered[1], keypoints_3d_reconstructed_filtered[2]
                sample["tracked_coords"] = reconstruct_rectangle_by_neighbour_points(ref_point2, ref_point3, full_body_width, 1)
            elif points_on_the_border_status[1] and points_on_the_border_status[2]:
                ref_point4, ref_point1 = keypoints_3d_reconstructed_filtered[3], keypoints_3d_reconstructed_filtered[0]
                sample["tracked_coords"] = reconstruct_rectangle_by_neighbour_points(ref_point4, ref_point1, full_body_width, 3)
            elif points_on_the_border_status[1] and points_on_the_border_status[3]:
                ratio = full_body_length / full_body_width
                ref_point1, ref_point3 = keypoints_3d_reconstructed_filtered[0], keypoints_3d_reconstructed_filtered[2]
                sample["tracked_coords"] = reconstruct_points(ref_point1, ref_point3, ratio, offset=0)
            elif points_on_the_border_status[2] and points_on_the_border_status[3]:
                ref_point1, ref_point2 = keypoints_3d_reconstructed_filtered[0], keypoints_3d_reconstructed_filtered[1]
                sample["tracked_coords"] = reconstruct_rectangle_by_neighbour_points(ref_point1, ref_point2, full_body_length, 0)
        elif "tracked_coords" not in sample:
            sample["tracked_coords"] = None

        # TODO: добавить z координаты к выходу

        cur_widths = []
        cur_lengths = []
        if not points_on_the_border_status[0] and not points_on_the_border_status[1]:
            cur_widths.append(
                np.linalg.norm(np.keypoints_3d_reconstructed_filtered[0] - keypoints_3d_reconstructed_filtered[1])
            )
        if not points_on_the_border_status[2] and not points_on_the_border_status[3]:
            cur_widths.append(
                np.linalg.norm(keypoints_3d_reconstructed_filtered[2] - keypoints_3d_reconstructed_filtered[3])
            )
        if not points_on_the_border_status[1] and not points_on_the_border_status[2]:
            cur_lengths.append(
                np.linalg.norm(keypoints_3d_reconstructed_filtered[1] - keypoints_3d_reconstructed_filtered[2])
            )
        if not points_on_the_border_status[0] and not points_on_the_border_status[3]:
            cur_lengths.append(
                np.linalg.norm(keypoints_3d_reconstructed_filtered[0] - keypoints_3d_reconstructed_filtered[3])
            )

        if len(cur_widths) > 0:
            cur_width = np.mean(cur_widths)
            cur_session_info.add_width(cur_width)

        if len(cur_lengths) > 0:
            cur_length = np.mean(cur_lengths)
            cur_session_info.add_length(cur_length)

        full_body_length, full_body_width = None, None
        if cur_session_info.is_initialized():
            full_body_length = cur_session_info.length
            full_body_width = cur_session_info.width

        sample["full_body_length"] = full_body_length
        sample["full_body_width"] = full_body_width

        return sample
