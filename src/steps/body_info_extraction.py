import typing as tp

import numpy as np

from .base import BaseStep
from ..geometry import final_point_rectangle, check_point_is_outside


class BodyInfoExtractionStep(BaseStep):
    def call(self, sample: dict) -> dict:
        # Reconstruct points
        image_height, image_width = sample["image_height"], sample["image_width"]

        height_width_np = np.array([image_height, image_width])

        points_on_the_border_status = [
            check_point_is_outside(point, max_y=image_height, max_x=image_width)
            for point in sample["keypoints_2d"]
        ]
        keypoints_3d_reconstructed = sample["keypoints_3d"]
        keypoints_3d_reconstructed_filtered = None
        # TODO: мб несколько раз запускать для большей робастности
        for _ in range(1):
            keypoints_3d_reconstructed = self._reconstruct_external_3d_points(
                keypoints_3d_reconstructed, points_on_the_border_status
            )
            keypoints_3d_reconstructed_filtered = [
                point
                if not check_point_is_outside(rec_point, max_y=image_height, max_x=image_width)
                else
                rec_point
                for point, rec_point in zip(sample["keypoints_3d"], keypoints_3d_reconstructed)
            ]

        assert keypoints_3d_reconstructed_filtered is not None

        # Measure parameters
        center_1 = (keypoints_3d_reconstructed_filtered[0] + keypoints_3d_reconstructed_filtered[2]) / 2
        center_2 = (keypoints_3d_reconstructed_filtered[1] + keypoints_3d_reconstructed_filtered[3]) / 2
        center_3d = (center_1 + center_2) / 2
        center_2d = project_to_2d(center_3d) / height_width_np

        keypoints_2d_reconstructed_filtered = [
            project_to_2d(kp) / height_width_np for kp in keypoints_3d_reconstructed_filtered
        ]

        width_1 = np.linalg.norm(keypoints_2d_reconstructed_filtered[0] - keypoints_2d_reconstructed_filtered[1])
        width_2 = np.linalg.norm(keypoints_2d_reconstructed_filtered[2] - keypoints_2d_reconstructed_filtered[3])

        length_1 = np.linalg.norm(keypoints_2d_reconstructed_filtered[1] - keypoints_2d_reconstructed_filtered[2])
        length_2 = np.linalg.norm(keypoints_2d_reconstructed_filtered[0] - keypoints_2d_reconstructed_filtered[3])

        sample["keypoints_on_the_border_status"] = np.array(points_on_the_border_status)
        sample["keypoints_3d_reconstructed"] = np.stack(keypoints_3d_reconstructed_filtered)
        sample["keypoints_2d_reconstructed"] = np.stack(keypoints_2d_reconstructed_filtered)

        # TODO: мб усреднять
        # TODO: мб назвать length
        sample["height"] = max(length_1, length_2)
        sample["width"] = max(width_1, width_2)

        sample["center"] = center_2d

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