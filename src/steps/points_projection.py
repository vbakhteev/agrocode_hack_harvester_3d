import numpy as np

from .base import BaseStep
import open3d as o3d
import pyrealsense2 as rs


def get_extrinsic_and_intrinsic_camera_parameters(parameters, order='zxy'):
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=parameters['intrinsics']['width'],
        height=parameters['intrinsics']['height'],
        fx=parameters['intrinsics']['fx'],
        fy=parameters['intrinsics']['fy'],
        cx=parameters['intrinsics']['ppx'],
        cy=parameters['intrinsics']['ppy'],
    )
    extrinsic = extrinsic_matrix(
        theta1=parameters['yaw_rad'],
        theta2=parameters['pitch_rad'],
        theta3=parameters['roll_rad'],
        order=order
    )
    cam_params = o3d.camera.PinholeCameraParameters()
    cam_params.extrinsic = extrinsic
    cam_params.intrinsic = intrinsic
    return cam_params, intrinsic, extrinsic


def extrinsic_matrix(theta1, theta2, theta3, order='zxy'):
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
        cam_params, intrinsic, extrinsic = get_extrinsic_and_intrinsic_camera_parameters(meta)
        view_ctl.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
        depth = vis.capture_depth_float_buffer(do_render=True)
        sample['depth_image'] = np.asarray(depth)
        return sample


class PointsProjection3D(BaseStep):
    def __init__(self):
        pass

    def call(self, sample):
        top_points = sample['top_points']
        bot_points = sample['bot_points']
        depth_image = sample['depth_image']
        top_phys_coords = []
        bot_phys_coords = []
        if top_points.any():
            for point in top_points:
                y = point[0]
                x = point[1]
                depth = depth_image[x][y]
                if depth == 0:
                    depth = self.find_nearest_3d_point_depth(x, y, depth_image, parameters=sample['meta'], expansion=1)
                top_phys_coords.append(
                    self.convert_depth_to_phys_coord_using_realsense(x, y, depth, parameters=sample['meta']))
            sample['top_phys_coords'] = np.asarray(top_phys_coords)
            sample['keypoints_3d'] = np.asarray(top_phys_coords)[[0, 3, 2, 1]]
        if bot_points.any():
            for point in bot_points:
                y = point[0]
                x = point[1]
                depth = depth_image[x][y]
                if depth == 0:
                    depth = self.find_nearest_3d_point_depth(x, y, depth_image, parameters=sample['meta'], expansion=1)
                bot_phys_coords.append(
                    self.convert_depth_to_phys_coord_using_realsense(x, y, depth, parameters=sample['meta']))
            sample['bot_phys_coords'] = np.asarray(bot_phys_coords)

        return sample

    def find_nearest_3d_point_depth(self, x, y, depth_image, parameters, expansion=1):
        for i in range(expansion):
            Ys = np.linspace(y - expansion, y + expansion, 2 * expansion + 1)
            Xs = np.linspace(x - expansion, x + expansion, 2 * expansion + 1)
        y_s = np.concatenate([np.repeat([Ys[-1]], 2 * expansion + 1), np.repeat([Ys[0]], 2 * expansion + 1)])
        x_s = np.concatenate([np.repeat([Xs[-1]], 2 * expansion + 1), np.repeat([Xs[0]], 2 * expansion + 1)])
        horizontal_points = list(zip(np.repeat([Xs], 2, axis=0).reshape(-1), y_s))
        vertical_points = list(zip(x_s, np.repeat([Ys], 2, axis=0).reshape(-1)))
        candidate_points = np.concatenate([horizontal_points, vertical_points])
        for candidate_point in candidate_points:
            x_hat = int(candidate_point[0])
            y_hat = int(candidate_point[1])
            if x_hat >= 0:
                x_hat = min(x_hat, parameters['intrinsics']['width'])
            else:
                x_hat = 0
            if y_hat >= 0:
                y_hat = min(y_hat, parameters['intrinsics']['height'])
            else:
                y_hat = 0
            depth = depth_image[x_hat][y_hat]
            if depth != 0:
                return depth
        return self.find_nearest_3d_point_depth(x, y, depth_image, parameters, expansion + 1)

    def _get_camera_params(self, parameters):
        _intrinsics = rs.intrinsics()
        _extrinsics = rs.extrinsics()
        _, _, extrinsics = get_extrinsic_and_intrinsic_camera_parameters(parameters)
        _extrinsics.rotation = list(extrinsics[:3, :3].reshape(-1))
        _extrinsics.translation = list(extrinsics[:3, 3])

        _intrinsics.width = parameters['intrinsics']['width']
        _intrinsics.height = parameters['intrinsics']['height']
        _intrinsics.ppx = parameters['intrinsics']['ppx']
        _intrinsics.ppy = parameters['intrinsics']['ppy']
        _intrinsics.fx = parameters['intrinsics']['fx']
        _intrinsics.fy = parameters['intrinsics']['fy']
        _intrinsics.model = rs.distortion.inverse_brown_conrady
        _intrinsics.coeffs = [i for i in parameters['intrinsics']['coeffs']]
        return _intrinsics, _extrinsics

    def convert_depth_to_phys_coord_using_realsense(self, x, y, depth, parameters):
        _intrinsics, _extrinsics = self._get_camera_params(parameters)
        result = rs.rs2_deproject_pixel_to_point(_intrinsics, [x, y], depth)
        return [result[2], result[0], result[1]]
