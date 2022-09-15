import json

import cv2


def read_image(p):
    return cv2.imread(str(p))[:, :, ::-1].copy()

def read_json(p):
    with open(p) as f:
        data = json.load(f)
    
    return data
