from collections import defaultdict
from dataclasses import dataclass
import typing as tp

import numpy as np

from .base import BaseStep
from ..geometry import final_point_rectangle, reconstruct_rectangle_by_neighbour_points, reconstruct_points
from ..tracking_utils import HistoryBuffer


@dataclass
class DimensionsInfoTracker:
    end_ratio: float = 0.1
    _max_memory: int = 100
    _lengths_buffer: tp.Optional[HistoryBuffer] = None
    _widths_buffer: tp.Optional[HistoryBuffer] = None

    def __post_init__(self) -> None:
        self._lengths_buffer = HistoryBuffer(end_ratio=self.end_ratio, max_memory=self._max_memory)
        self._widths_buffer = HistoryBuffer(end_ratio=self.end_ratio, max_memory=self._max_memory)

    def is_initialized(self) -> bool:
        return self._lengths_buffer.is_initialized() and self._widths_buffer.is_initialized()

    def add_length(self, length: float) -> None:
        self._lengths_buffer.add_value(length)

    def add_width(self, width: float) -> None:
        self._widths_buffer.add_value(width)

    @property
    def length(self) -> float:
        return self._lengths_buffer.value

    @property
    def width(self) -> float:
        return self._widths_buffer.value


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

        # Трекинг и восстановление точек
        sample["tracked_coords"] = None
        if sum(points_on_the_border_status) == 0:
            print("No reconstruction needed")
            sample["tracked_coords"] = list(keypoints_3d_reconstructed_filtered)
        elif sum(points_on_the_border_status) == 1:
            print("Three points reconstruction")
            idx = np.where(points_on_the_border_status)[0][0]

            prev_point = keypoints_3d_reconstructed_filtered[idx - 1]
            next_point = keypoints_3d_reconstructed_filtered[(idx + 1) % len(keypoints_3d_reconstructed_filtered)]
            next_next_point = keypoints_3d_reconstructed_filtered[(idx + 2) % len(keypoints_3d_reconstructed_filtered)]

            final_point = final_point_rectangle(prev_point, next_point, next_next_point)
            tracked_coords = keypoints_3d_reconstructed_filtered
            tracked_coords[idx] = final_point

            sample["tracked_coords"] = list(tracked_coords)

        elif sum(points_on_the_border_status) == 2 and cur_session_info.is_initialized():
            print("Two points reconstruction")
            full_body_length, full_body_width = cur_session_info.length, cur_session_info.width

            if points_on_the_border_status[0] and points_on_the_border_status[1]:
                print("Rec 1")
                ref_point3, ref_point4 = keypoints_3d_reconstructed_filtered[2], keypoints_3d_reconstructed_filtered[3]
                sample["tracked_coords"] = reconstruct_rectangle_by_neighbour_points(ref_point3, ref_point4, full_body_length, 2)
            elif points_on_the_border_status[0] and points_on_the_border_status[2]:
                print("Rec 2")
                ratio = full_body_width / full_body_length
                ref_point2, ref_point4 = keypoints_3d_reconstructed_filtered[1], keypoints_3d_reconstructed_filtered[3]
                sample["tracked_coords"] = reconstruct_points(ref_point2, ref_point4, ratio, offset=1)
            elif points_on_the_border_status[0] and points_on_the_border_status[3]:
                print("Rec 3")
                ref_point2, ref_point3 = keypoints_3d_reconstructed_filtered[1], keypoints_3d_reconstructed_filtered[2]
                sample["tracked_coords"] = reconstruct_rectangle_by_neighbour_points(ref_point2, ref_point3, full_body_width, 1)
            elif points_on_the_border_status[1] and points_on_the_border_status[2]:
                print("Rec 4")
                ref_point4, ref_point1 = keypoints_3d_reconstructed_filtered[3], keypoints_3d_reconstructed_filtered[0]
                sample["tracked_coords"] = reconstruct_rectangle_by_neighbour_points(ref_point4, ref_point1, full_body_width, 3)
            elif points_on_the_border_status[1] and points_on_the_border_status[3]:
                print("Rec 5")
                ratio = full_body_length / full_body_width
                ref_point1, ref_point3 = keypoints_3d_reconstructed_filtered[0], keypoints_3d_reconstructed_filtered[2]
                sample["tracked_coords"] = reconstruct_points(ref_point1, ref_point3, ratio, offset=0)
            elif points_on_the_border_status[2] and points_on_the_border_status[3]:
                print("Rec 6")
                ref_point1, ref_point2 = keypoints_3d_reconstructed_filtered[0], keypoints_3d_reconstructed_filtered[1]
                sample["tracked_coords"] = reconstruct_rectangle_by_neighbour_points(ref_point1, ref_point2, full_body_length, 0)
        else:
            print("No reconstruction available")
        # Добавление z координат к точкам

        if sample["tracked_coords"] is not None:
            default_z_value = sample["keypoints_3d_reconstructed"][~points_on_the_border_status][:, 2].mean()

            for i, tracked_point in enumerate(sample["tracked_coords"]):
                found_pair = False

                for j, (given_point, given_point_outside_status) in enumerate(zip(sample["keypoints_3d_reconstructed"], points_on_the_border_status)):
                    if np.linalg.norm(tracked_point - given_point[:2]) < 1e-4 and not given_point_outside_status:
                        sample["tracked_coords"][i] = given_point
                        found_pair = True
                        break
                if not found_pair:
                    sample["tracked_coords"][i] = np.array([tracked_point[0], tracked_point[1], default_z_value])
            sample["tracked_coords"] = np.stack(sample["tracked_coords"])

        # Сохранение инфы о ширине и высоте

        cur_widths = []
        cur_lengths = []
        if not points_on_the_border_status[0] and not points_on_the_border_status[1]:
            cur_widths.append(
                np.linalg.norm(keypoints_3d_reconstructed_filtered[0] - keypoints_3d_reconstructed_filtered[1])
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

        full_body_length, full_body_width = 0, 0
        if cur_session_info.is_initialized():
            full_body_length = cur_session_info.length
            full_body_width = cur_session_info.width

        sample["full_body_length"] = full_body_length
        sample["full_body_width"] = full_body_width

        return sample
