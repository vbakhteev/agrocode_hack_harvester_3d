import cv2
import numpy as np

from .base import BaseStep


class DetectPointsOnMask(BaseStep):
    def call(self, sample: dict) -> dict:
        mask = sample['mask']

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            sample['top_points'] = None
            return sample

        biggest_contour = max(contours, key=cv2.contourArea)
        biggest_contour = cv2.convexHull(biggest_contour)

        points = get_rectangle_coords(
            biggest_contour, h=mask.shape[0], w=mask.shape[1],
        )
        sample['top_points'] = np.array(points)

        return sample


def get_rectangle_coords(contour, h, w):
    right_bot, left_bot = get_bot_points(contour, h=h)
    left_top, right_top = get_top_points_with_perspective(
        contour, h=h, w=w
    )

    return left_top, right_top, right_bot, left_bot


def get_bot_points(contour, h):
    contour = contour[:, 0].copy()

    coords_sum = contour.sum(1)
    right_bot = contour[np.argmax(coords_sum)].copy()

    contour[:, 1] = h - contour[:, 1]
    coords_sum = contour.sum(1)
    left_bot = contour[np.argmin(coords_sum)].copy()
    left_bot[1] = h - left_bot[1]

    return right_bot, left_bot


def get_top_points(contour, img_h):
    contour = contour[:, 0].copy()

    coords_sum = contour.sum(1)
    left_top = contour[np.argmin(coords_sum)].copy()

    contour[:, 1] = img_h - contour[:, 1]
    coords_sum = contour.sum(1)
    right_top = contour[np.argmax(coords_sum)].copy()
    right_top[1] = img_h - right_top[1]
    
    return left_top, right_top


def get_top_points_with_perspective(contour, h, w):
    src = np.float32([[0, h], [w, h], [0, 0], [w, 0]])
    dst = np.float32([[175, h], [w - 175, h], [0, 0], [w, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    flat_contour = cv2.perspectiveTransform(
        contour.astype(np.float32), M
    )

    left_top, right_top = get_top_points(flat_contour, img_h=h)

    left_top, right_top = cv2.perspectiveTransform(
        np.array([[left_top, right_top]], dtype=np.float32), Minv
    )[0].astype(int)

    return left_top, right_top
