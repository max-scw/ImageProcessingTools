import numpy as np


def xywh2xyxy(xywh: np.ndarray) -> np.ndarray:
    """
    transforms bounding-box coordinates from center-point representation to corner-point representation
    Args:
        xywh:

    Returns:

    """
    # xywh is assumed to be an array of shape (n, 4)
    x, y, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def xyxy2xywh(xyxy: np.ndarray) -> np.ndarray:
    """
    transforms bounding-box coordinates from corner-point representation to center-point representation
    """
    # xyxy is assumed to be an array of shape (n, 4)
    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return np.stack([x, y, w, h], axis=1)
