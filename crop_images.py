from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import logging

from PIL import Image
import numpy as np

from typing import Tuple, Union

from utils.bbox import xyxy2xywh, xywh2xyxy


def crop_bounding_boxes(boxes, limits):
    # Extract limits
    left, upper, right, bottom = limits

    # Shift coordinates by left and upper limits
    boxes[:, [0, 2]] -= left   # Shift x_min and x_max by left
    boxes[:, [1, 3]] -= upper  # Shift y_min and y_max by upper

    # Calculate new width and height limits
    width = right - left
    height = bottom - upper

    # Clip coordinates to the new limits
    boxes[:, 0] = np.clip(boxes[:, 0], 0, width)   # x_min
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height)  # y_min
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width)   # x_max
    boxes[:, 3] = np.clip(boxes[:, 3], 0, height)  # y_max

    # TODO: delete boxes that now have no area anymore
    return boxes


def crop_image(
        img: Image,
        dimensions: Tuple[Union[float, int], Union[float, int], Union[float, int], Union[float, int]],
        label: np.ndarray = None,
        label_format: str = None
) -> Tuple[Image, Union[np.ndarray, None]]:
    sz = img.size * 2

    # to absolute dimensions
    dimensions_abs = [
        int(0 if d < 0 else d * s if d <= 1 else d)
        for d, s in zip(dimensions, sz)
    ]
    logging.debug(f"Cropping image to {dimensions_abs}.")
    im = img.crop(dimensions_abs)

    # crop labels
    if label is not None:
        cls = label[:, 0]
        bbox = label[:, 1:5]
        # use corner-point coordinates
        if label_format == "xywh":
            bbox = xywh2xyxy(bbox)
        # to absolute coordinates
        bbox_abs = bbox * sz
        # crop
        bbox_abs = crop_bounding_boxes(bbox_abs, dimensions_abs)
        # to relative coordinates
        bbox_rel = bbox_abs / (im.size * 2)  # use new image size in case it was padded

        # back transformation
        if label_format == "xywh":
            bbox_rel = xyxy2xywh(bbox_rel)

        label = np.hstack((cls.reshape(-1, 1), bbox_rel))

    return im, label


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--source", type=str, help="Directory where the images are stored.")
    parser.add_argument("--destination", type=str, default="",
                        help="Directory to copy the selected images to")
    parser.add_argument("--file-extension", type=str, default="", help="File type")

    parser.add_argument("--roi", type=float, nargs="+",
                        help="Tuple to specify the region of interest (roi): [left, upper, right, lower].")
    parser.add_argument("--bb-format", type=str, default="xywh",
                        help="Format of the bounding boxes in the label file. Can be xyxy or xywh.")

    opt = parser.parse_args()

    destination = Path(opt.destination)

    files = list(Path(opt.source).glob("*." + opt.file_extension.strip(".") if opt.file_extension else "*"))
    for fl in tqdm(files):
        # load file
        img = Image.open(fl)
        # check if a label file exists as well
        fl_txt = fl.with_suffix(".txt")
        if fl_txt.is_file():
            labels = np.loadtxt(fl_txt.as_posix())
        else:
            labels = None

        # crop image and label
        img_crp, label_crp = crop_image(
            img,
            opt.roi,
            label=labels,
            label_format=opt.bb_format
        )
        # save image
        img_crp.save(destination / fl.name)
        # save label
        if label_crp is not None:
            with open(destination / f"{fl.stem}.txt", "w") as fid:
                fid.writelines("\n".join([" ".join([f"{e:.3g}" for e in el]) for el in label_crp]))
