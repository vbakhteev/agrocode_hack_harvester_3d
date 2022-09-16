import numpy as np

from .base import BaseStep
import open3d as o3d


class PointsProjection2D(BaseStep):
    """Детектирует границы кузова на 2д картинке.
    Возвращает массив точек, описывающих линии
    Кол-во точек > 4 и всегда разное
    """

    def call(self, sample):
        vis = o3d.visualization.Visualizer()
        meta = sample['meta']
        vis.create_window(width=meta["intrinsics"]['width'], height=meta["intrinsics"]['height'])
        vis.add_geometry(sample['point_cloud'])
        view_ctl = vis.get_view_control()
        cam_params, intrinsic, extrinsic = self._get_extrinsic_and_intrinsic_camera_parameters(meta)
        view_ctl.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
        depth = vis.capture_depth_float_buffer(do_render=True)
        return np.asarray(depth)

    def _get_extrinsic_and_intrinsic_camera_parameters(self, parameters, order='xyz'):
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=parameters['intrinsics']['width'],
            height=parameters['intrinsics']['height'],
            fx=parameters['intrinsics']['fx'],
            fy=parameters['intrinsics']['fy'],
            cx=parameters['intrinsics']['ppx'],
            cy=parameters['intrinsics']['ppy'],
        )
        extrinsic = self._extrinsic_matrix(
            theta1=parameters['yaw_rad'],
            theta2=parameters['pitch_rad'],
            theta3=parameters['roll_rad'],
            order=order
        )
        cam_params = o3d.camera.PinholeCameraParameters()
        cam_params.extrinsic = extrinsic
        cam_params.intrinsic = intrinsic
        return cam_params, intrinsic, extrinsic

    def _extrinsic_matrix(self, theta1, theta2, theta3, order='xyz'):
        """
        input
            theta1, theta2, theta3 = rotation angles in rotation order (degrees)
            oreder = rotation order of x,y,z　e.g. XZY rotation -- 'xzy'
        output
            3x3 rotation matrix (numpy array)
        """
        c1 = np.cos(theta1 * np.pi / 180)
        s1 = np.sin(theta1 * np.pi / 180)
        c2 = np.cos(theta2 * np.pi / 180)
        s2 = np.sin(theta2 * np.pi / 180)
        c3 = np.cos(theta3 * np.pi / 180)
        s3 = np.sin(theta3 * np.pi / 180)

        if order == 'xzx':
            matrix = np.array([[c2, -c3 * s2, s2 * s3],
                               [c1 * s2, c1 * c2 * c3 - s1 * s3, -c3 * s1 - c1 * c2 * s3],
                               [s1 * s2, c1 * s3 + c2 * c3 * s1, c1 * c3 - c2 * s1 * s3]])
        elif order == 'xyx':
            matrix = np.array([[c2, s2 * s3, c3 * s2],
                               [s1 * s2, c1 * c3 - c2 * s1 * s3, -c1 * s3 - c2 * c3 * s1],
                               [-c1 * s2, c3 * s1 + c1 * c2 * s3, c1 * c2 * c3 - s1 * s3]])
        elif order == 'yxy':
            matrix = np.array([[c1 * c3 - c2 * s1 * s3, s1 * s2, c1 * s3 + c2 * c3 * s1],
                               [s2 * s3, c2, -c3 * s2],
                               [-c3 * s1 - c1 * c2 * s3, c1 * s2, c1 * c2 * c3 - s1 * s3]])
        elif order == 'yzy':
            matrix = np.array([[c1 * c2 * c3 - s1 * s3, -c1 * s2, c3 * s1 + c1 * c2 * s3],
                               [c3 * s2, c2, s2 * s3],
                               [-c1 * s3 - c2 * c3 * s1, s1 * s2, c1 * c3 - c2 * s1 * s3]])
        elif order == 'zyz':
            matrix = np.array([[c1 * c2 * c3 - s1 * s3, -c3 * s1 - c1 * c2 * s3, c1 * s2],
                               [c1 * s3 + c2 * c3 * s1, c1 * c3 - c2 * s1 * s3, s1 * s2],
                               [-c3 * s2, s2 * s3, c2]])
        elif order == 'zxz':
            matrix = np.array([[c1 * c3 - c2 * s1 * s3, -c1 * s3 - c2 * c3 * s1, s1 * s2],
                               [c3 * s1 + c1 * c2 * s3, c1 * c2 * c3 - s1 * s3, -c1 * s2],
                               [s2 * s3, c3 * s2, c2]])
        elif order == 'xyz':
            matrix = np.array([[c2 * c3, -c2 * s3, s2],
                               [c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3, -c2 * s1],
                               [s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3, c1 * c2]])
        elif order == 'xzy':
            matrix = np.array([[c2 * c3, -s2, c2 * s3],
                               [s1 * s3 + c1 * c3 * s2, c1 * c2, c1 * s2 * s3 - c3 * s1],
                               [c3 * s1 * s2 - c1 * s3, c2 * s1, c1 * c3 + s1 * s2 * s3]])
        elif order == 'yxz':
            matrix = np.array([[c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3, c2 * s1],
                               [c2 * s3, c2 * c3, -s2],
                               [c1 * s2 * s3 - c3 * s1, c1 * c3 * s2 + s1 * s3, c1 * c2]])
        elif order == 'yzx':
            matrix = np.array([[c1 * c2, s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3],
                               [s2, c2 * c3, -c2 * s3],
                               [-c2 * s1, c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3]])
        elif order == 'zyx':
            matrix = np.array([[c1 * c2, c1 * s2 * s3 - c3 * s1, s1 * s3 + c1 * c3 * s2],
                               [c2 * s1, c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3],
                               [-s2, c2 * s3, c2 * c3]])
        elif order == 'zxy':
            matrix = np.array([[c1 * c3 - s1 * s2 * s3, -c2 * s1, c1 * s3 + c3 * s1 * s2],
                               [c3 * s1 + c1 * s2 * s3, c1 * c2, s1 * s3 - c1 * c3 * s2],
                               [-c2 * s3, s2, c2 * c3]])

        matrix = np.concatenate([matrix, np.zeros(3).reshape((3, 1))], axis=1)
        matrix = np.concatenate([matrix, [[0, 0, 0, 1]]])

        return matrix