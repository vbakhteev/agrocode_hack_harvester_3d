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
