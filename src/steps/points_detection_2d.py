import cv2
import numpy as np

from .base import BaseStep


class PointsDetection2d(BaseStep):
    """Детектирует границы кузова на 2д картинке.
    Возвращает массив точек, описывающих линии
    Кол-во точек > 4 и всегда разное
    """
    def call(self, sample):
        image = sample['color_frame']
        bot_points, bot_contour = close_side_coords(image)
        top_points, top_contour = top_side_coords(image, bot_contour)

        img_draw = image.copy()
        if top_points is not None:
            for p in top_points:
                cv2.circle(img_draw, p, 5, (255, 0, 0), -1)
            cv2.drawContours(img_draw, [top_contour], -1, (255, 0, 0))
        for p in bot_points:
            cv2.circle(img_draw, p, 5, (0, 0, 255), -1)
        cv2.drawContours(img_draw, [bot_contour], -1, (0, 0, 255))
        
        sample['detection_image'] = img_draw
        return sample


def get_orange_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (5, 100, 125), (13, 200, 255))
    return mask


def get_rectangle_coords(contour, img_h):
    contour = contour[:, 0].copy()
    coords_sum = contour.sum(1)
    left_top = contour[np.argmin(coords_sum)].copy()
    right_bot = contour[np.argmax(coords_sum)].copy()

    contour[:, 1] = img_h - contour[:, 1]
    coords_sum = contour.sum(1)
    left_bot = contour[np.argmin(coords_sum)].copy()
    left_bot[1] = img_h - left_bot[1]
    right_top = contour[np.argmax(coords_sum)].copy()
    right_top[1] = img_h - right_top[1]
    
    return left_top, right_top, right_bot, left_bot


def close_side_coords(img):
    mask = get_orange_mask(img)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    biggest_contours = sorted(contours, key=cv2.contourArea)[-2:]
    if cv2.contourArea(biggest_contours[0]) / cv2.contourArea(biggest_contours[1]) < 0.2:
        biggest_contours = [biggest_contours[1]]
    biggest_contour = np.concatenate(biggest_contours, axis=0)
    biggest_contour = cv2.convexHull(biggest_contour)

    bot_points = get_rectangle_coords(biggest_contour, mask.shape[0])
    
    return bot_points, biggest_contour[:, 0]


def top_side_coords(img, bot_contour):
    edges = cv2.Canny(img,100,200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.arcLength(c, True) > 100]
    edges = np.zeros_like(edges)
    edges = cv2.drawContours(edges, contours, -1, (255,))

    edges[:, :bot_contour[:, 0].min()] = 0
    edges[:, bot_contour[:, 0].max():] = 0
    edges[bot_contour[:, 1].min():] = 0

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        one_big_contour = np.concatenate(contours, axis=0)
        one_big_contour = cv2.convexHull(one_big_contour)

        top_points = get_rectangle_coords(one_big_contour, img.shape[0])
    else:
        top_points = None
        one_big_contour = None

    return top_points, one_big_contour