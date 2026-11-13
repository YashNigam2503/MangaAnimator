"""Face detector used to check diffusion fills: a fill that invents a face is rejected.

The model is a YOLOv8 anime face detector, run from its ONNX file with
onnxruntime. It works the same on CPU and GPU machines.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

PAD_VALUE = 114  # grey border YOLO models are trained with


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, iou_limit: float) -> list[int]:
    """Indices of the boxes to keep: the best-scoring one of each overlapping cluster."""
    order = scores.argsort()[::-1]
    keep = []
    while len(order):
        best, rest = order[0], order[1:]
        keep.append(int(best))
        x0 = np.maximum(boxes[best, 0], boxes[rest, 0])
        y0 = np.maximum(boxes[best, 1], boxes[rest, 1])
        x1 = np.minimum(boxes[best, 2], boxes[rest, 2])
        y1 = np.minimum(boxes[best, 3], boxes[rest, 3])
        shared = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = rest[shared / (area[best] + area[rest] - shared) <= iou_limit]
    return keep


class AnimeFaceDetector:
    def __init__(self, cfg: dict):
        import onnxruntime
        from huggingface_hub import hf_hub_download
        self.size = cfg["imgsz"]
        self.conf = cfg["conf"]
        self.iou = cfg["iou"]
        self.session = onnxruntime.InferenceSession(
            hf_hub_download(cfg["model_repo"], cfg["model_file"]), providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, image: Image.Image) -> list[tuple[float, float, float, float, float]]:
        """Faces in the image as (x0, y0, x1, y1, confidence), in image pixels."""
        scale = self.size / max(image.size)
        resized = image.convert("RGB").resize(
            (max(round(image.width * scale), 1), max(round(image.height * scale), 1)), Image.BILINEAR)
        canvas = np.full((self.size, self.size, 3), PAD_VALUE, dtype=np.uint8)
        canvas[:resized.height, :resized.width] = np.asarray(resized)
        blob = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        out = self.session.run(None, {self.input_name: blob})[0][0]
        if out.shape[0] < out.shape[1]:
            out = out.T                                   # rows: centre x, centre y, width, height, class scores
        scores = out[:, 4:].max(axis=1)
        found = scores >= self.conf
        centre, scores = out[found, :4], scores[found]
        boxes = np.column_stack([centre[:, 0] - centre[:, 2] / 2, centre[:, 1] - centre[:, 3] / 2,
                                 centre[:, 0] + centre[:, 2] / 2, centre[:, 1] + centre[:, 3] / 2]) / scale
        return [(*(float(v) for v in boxes[i]), float(scores[i]))
                for i in non_max_suppression(boxes, scores, self.iou)]


def face_inside_mask(faces: list[tuple], mask: Image.Image) -> tuple | None:
    """The first face whose centre lies on a white pixel of the mask, or None."""
    pixels = np.asarray(mask.convert("L"))
    for face in faces:
        cx, cy = round((face[0] + face[2]) / 2), round((face[1] + face[3]) / 2)
        if 0 <= cy < pixels.shape[0] and 0 <= cx < pixels.shape[1] and pixels[cy, cx] > 0:
            return face
    return None
