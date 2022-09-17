import cv2
import numpy as np
import onnxruntime

from .base import BaseStep


class SegmentationStep(BaseStep):
    def __init__(self, weights_path, img_h=160, img_w=288, threshold=0.5):
        self.img_h = img_h
        self.img_w = img_w
        self.threshold = threshold
        self.model = OnnxRuntimeModel(weights_path)
    
    def call(self, sample):
        orig_img = sample['color_frame']

        mask = segment(orig_img, self.model, self.img_w, self.img_h, self.threshold)

        sample['mask'] = mask

        return sample


def segment(orig_img, model, w, h, threshold):
    img = cv2.resize(orig_img, dsize=(w, h))
    img = (img / 255).astype(np.float32)
    img = np.moveaxis(img, -1, 0)[None]

    proba = model.forward(img)[0, 0]
    mask = (proba > threshold).astype(np.uint8)
    upsampled_mask = cv2.resize(mask, dsize=(orig_img.shape[1], orig_img.shape[0]))

    return upsampled_mask

class OnnxRuntimeModel:
    def __init__(self, weights_path):
        self.model = onnxruntime.InferenceSession(str(weights_path))

    def forward(self, x):
        ort_inputs = {self.model.get_inputs()[0].name: x}
        pred = self.model.run(None, ort_inputs)[0]
        return pred
