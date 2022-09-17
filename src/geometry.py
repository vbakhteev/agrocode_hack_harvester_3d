import math

import numpy as np


def check_point_is_outside(point: np.ndarray, max_y: int, max_x: int, eps: float = 5) -> bool:
    x, y = point
    if x < eps or x > max_x - eps:
        return True
    if y < eps or y > max_y - eps:
        return True
    return False


def clip_point(point: np.ndarray, max_x: int, max_y: int) -> np.array:
    x, y = point
    x = min(max_x, max(0, x))
    y = min(max_y, max(0, y))
    return np.array([x, y])


def final_point_rectangle(prev_point: np.ndarray, next_point: np.ndarray, next_next_point: np.ndarray) -> np.ndarray:
    directional_vector = next_point - next_next_point
    reconstructed_point = prev_point + directional_vector
    return reconstructed_point


def reconstruct_rectangle_by_neighbour_points(ref_point1: np.ndarray, ref_point2: np.ndarray, length: float,
                                              offset: int) -> np.ndarray:
    direction_vector_r1_r2 = ref_point1 - ref_point2
    direction_vector_r1_r2 = direction_vector_r1_r2 / np.linalg.norm(direction_vector_r1_r2)
    theta = -math.pi / 2
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    direction_vector_r3_r2 = rotation_matrix @ direction_vector_r1_r2
    direction_vector_r3_r2 = direction_vector_r3_r2 * length
    ref_point3 = ref_point2 + direction_vector_r3_r2
    ref_point4 = final_point_rectangle(ref_point3, ref_point1, ref_point2)
    result = np.stack([ref_point1, ref_point2, ref_point3, ref_point4])
    offset_result = np.concatenate([result[-offset:], result[:-offset]])
    return offset_result


def reconstruct_points(p1: np.ndarray, p3: np.ndarray, ratio: float, offset: int) -> np.ndarray:
    # Если обходить точки против часовой стрелки, то ratio - отношение длины
    # более раннего отрезка к более позднему
    p1_p3_dist = np.linalg.norm(p1 - p3)
    p1_p2_dist = p1_p3_dist / (ratio ** 2 + 1) ** 0.5
    p3_p2_dist = ratio * p1_p2_dist

    theta = -np.arcsin(p3_p2_dist / p1_p3_dist)

    direction_vector_p1_p3 = p3 - p1
    direction_vector_p1_p3 = direction_vector_p1_p3 / np.linalg.norm(direction_vector_p1_p3)
    p2_p1_rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    p2_p1_vector = p2_p1_rotation_matrix @ direction_vector_p1_p3
    p2 = p1 + p2_p1_vector * p1_p2_dist
    p4 = p3 - p2_p1_vector * p1_p2_dist
    result = np.stack([p1, p2, p3, p4])
    offset_result = np.concatenate([result[-offset:], result[:-offset]])
    return offset_result


def project_to_2d(point_3d, parameters):
    Z = point_3d[0]
    X = point_3d[1]
    Y = point_3d[2]
    u = (X / Z) * parameters['intrinsics']['fx'] + parameters['intrinsics']['ppx']
    v = (Y / Z) * parameters['intrinsics']['fy'] + parameters['intrinsics']['ppy']
    return u, v
