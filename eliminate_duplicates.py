from pathlib import Path
import shutil
from argparse import ArgumentParser
import sys

from tqdm import tqdm
import logging

from utils.casting import cast_logging_level
from utils.describe_images import describe_images

from typing import Union, List, Dict

import subprocess


def get_file_paths_with_dir(root_dir: str, file_extension: str):
    command = f'dir /S /B "{root_dir}\\*.{file_extension.strip(".")}"'
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    paths = result.stdout.splitlines()
    return paths


def get_exceptional_images(
        features: dict,
        mse_threshold: float = 0.1,
        existing_features: dict = None
) -> List[Union[str, Path]]:
    # initialize list of (exceptional) features

    exceptional_images = []
    exceptional_images_new = []
    if isinstance(existing_features, dict):
        exceptional_images = list(existing_features.keys())

    # combine both dictionaries of feature vectors
    features_all = {**features, **existing_features}

    for ky, val in features.items():
        # calculate differences to already stored images
        diff = [((features_all[nm] - val) ** 2).mean().sqrt() for nm in exceptional_images_new + exceptional_images]
        if diff == [] or min(diff) > mse_threshold:
            exceptional_images_new.append(ky)
    return exceptional_images_new


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source", type=str, help="Directory where the images are stored.")
    parser.add_argument("--destination", type=str, default="",
                        help="Directory to copy the selected images to")
    parser.add_argument("--file-extension", type=str, default="", help="File type")
    parser.add_argument("--extra", type=str, default="", nargs="+", help="Example files that should be ex/included.")

    parser.add_argument("--threshold", type=float, default=0.3, help="Threshold for differentiation")

    parser.add_argument("--find-distinct", action="store_true",
                        help="Select distinct files (selects similar files per default)")

    parser.add_argument("--logging-level", type=str, default="INFO", help="Logging level")

    opt = parser.parse_args()

    # set up logging
    level = cast_logging_level(opt.logging_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logging.debug(f"Input arguments: {opt}")

    # process input
    source = Path(opt.source)
    existing_images = Path(opt.extra)
    image_extension = opt.file_extension.strip(".")

    # export folder
    destination = Path(opt.destination)

    logging.debug(f"get files in {source.as_posix()}(f'*.{image_extension}') ...")
    # p2images_new = list(folder.rglob(f"*.{image_extension}"))
    p2images_new = get_file_paths_with_dir(source.as_posix(), image_extension)
    logging.debug(f"Describe {len(p2images_new)} images by MobileNetV3 ...")
    features = describe_images(p2images_new)

    # p2images_old = list(existing_images.rglob(f"*.{image_extension}"))
    p2images_old = get_file_paths_with_dir(existing_images.as_posix(), image_extension)
    logging.debug(f"Describe {len(p2images_old)} existing images by MobileNetV3 ...")
    features_existing_images = describe_images(p2images_old)

    logging.info("Get exceptional images ...")
    images = get_exceptional_images(
        features,
        mse_threshold=opt.threshold,
        # directory or list of extra files
        existing_features=features_existing_images
    )

    logging.info("Copying images ...")
    # move and rename images
    if not destination.exists():
        destination.mkdir()

    for img in tqdm(images, desc="Copy images"):
        # fldr = p2img.parent.name
        p2export = destination / "_".join(img.parts[-2:])
        # copy file (+ rename)
        shutil.copy(img, p2export)
