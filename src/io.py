import json

import cv2
import open3d as o3d


def read_image(p):
    return cv2.imread(str(p))[:, :, ::-1].copy()


def read_json(p):
    with open(p) as f:
        data = json.load(f)
    
    return data


def read_ply(p):
    ply = o3d.io.read_point_cloud(str(p))
    return ply


class WriteVideoStreamImageio:
    def __init__(self, path, fps):
        self.path = path
        self.fps = fps
        self.writer = None

    def __call__(self, frame):
        if self.writer is None:
            h, w, _ = frame.shape
            self.writer = cv2.VideoWriter(
                str(self.path),
                cv2.VideoWriter_fourcc(*'MP4V'),
                self.fps,
                (w, h),
            )

        self.writer.write(frame[:, :, ::-1].copy())

    def close(self):
        self.writer.release()
