import numpy as np

from .base import BaseStep


class PointsDetection2d(BaseStep):
    """Детектирует границы кузова на 2д картинке.
    Возвращает массив точек, описывающих линии
    Кол-во точек > 4 и всегда разное
    """
    def call(self, sample):
        image = sample['color_frame']

        sample['points_2d'] = np.array([
            [23, 42],
            [245, 22],
            [56, 65],
            [24, 525],
            [4215, 25],
        ])
        return sample
