from argparse import ArgumentParser
from pathlib import Path
import shutil
from tqdm import tqdm
import logging
import sys
from math import floor

from eliminate_duplicates import build_descriptor_model, describe_image
from casting import cast_logging_level

from typing import List, Union


def calculate_feature_diffs(features1, features_all) -> List[float]:
    # calculate differences to already stored images
    return [float(((features1 - val) ** 2).mean().sqrt()) for val in features_all]


def select_images(
        source: Union[str, Path],  # source directory with images
        destination: Union[str, Path],  # export directory to compy images to
        threshold: float,
        find_distinct: bool = False,  # find similar or distinct images
        delete_files: bool = False,  # delete not used files afterwards?
        file_extension: str = "jpg",  # image extension
        extra_files: List[Union[str, Path]] = None,  # directory or list of extra files
        th_scheduler_window: int = 200,  # linear de/increasing scheduler
        th_scheduler_factor: float = 1,
        max_n_files: int = -1
):
    # build suffix for pathlib object
    suffix = f".{file_extension.strip('.')}"
    source = Path(source)

    logging.debug("Build descriptor model")
    model, preprocess = build_descriptor_model()

    logging.info(f"Describe extra images")
    if len(extra_files) == 1 and Path(extra_files[0]).is_dir():
        extra_ = list(Path(extra_files[0]).rglob("*" + suffix))
    else:
        extra_ = extra_files

    features = []
    for img in tqdm(extra_):
        features.append(describe_image(img, model, preprocess))

    # update threshold
    update_th = lambda x: threshold * (th_scheduler_factor ** floor(x / th_scheduler_window))
    th = update_th(len(features))

    logging.info(f"Walk through source directory ...")
    unused_files = []
    for i, img in tqdm(enumerate(source.rglob("*" + suffix))):
        img_features = describe_image(img, model, preprocess)

        differences = calculate_feature_diffs(img_features, features)

        diff_value = min(differences) if isinstance(differences, list) or differences == [] else 0 # FIXME
        if find_distinct:
            copy_file = True if (diff_value is None) or (diff_value > th) else False
        else:
            copy_file = True if (diff_value is None) or (diff_value < th) else False

        if copy_file:
            # new file name
            name_parts = img.parts[len(source.parts):]
            # delete points from directory parts
            name_parts = [el.replace(".", "") for el in name_parts[:-1]] + [name_parts[-1]]
            # joint parts to new filename
            filename = "_".join(name_parts)
            p2export = destination / filename

            if p2export.exists():
                logging.debug(f"File {p2export.as_posix()} already exists. File {img.as_posix()} was not copied")
            else:
                logging.debug(f"Copy {img.as_posix()} -> {p2export.as_posix()} | {diff_value:.3g}")
                # copy file (+ rename)
                shutil.copy(img, p2export)
                # keep features
                if find_distinct:
                    features.append(img_features)
                # update threshold
                th = update_th(len(features))
        else:
            unused_files.append(img)

        if 0 < max_n_files <= i:
            break
    logging.info(f"Done. {len(features) - len(extra_)} files copied.")

    # delete files
    if delete_files:
        logging.info(f"Deleting {len(unused_files)} files")
        for p2fl in tqdm(unused_files):
            p2fl.unlink()
            shutil.rmtree(p2fl.parent)
        logging.info(f"Done deleting unused files.")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--source", type=str, help="Directory where the images are stored.")
    parser.add_argument("--destination", type=str, default="",
                        help="Directory to copy the selected images to")
    parser.add_argument("--file-extension", type=str, default="", help="File type")
    parser.add_argument("--extra", type=str, default="", nargs="+", help="Example files that should be ex/included.")

    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for differentiation")
    parser.add_argument("--th-scheduler-window", type=int, default=200, help="Threshold for differentiation")
    parser.add_argument("--th-scheduler-factor", type=float, default=1, help="Threshold for differentiation")

    parser.add_argument("--find-distinct", action="store_true",
                        help="Select distinct files (selects similar files per default)")
    parser.add_argument("--delete-files", action="store_true", help="Delete files from source afterwards")
    parser.add_argument("--max-n-files", type=int, default=-1, help="Number of files to scan through.")

    parser.add_argument("--logging-level", type=str, default="DEBUG", help="Logging level")

    opt = parser.parse_args()

    # set up logging
    level = cast_logging_level(opt.logging_level, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logging.debug(f"Input arguments: {opt}")

    # call function
    select_images(
        # source directory with images
        source=opt.source,
        # image extension
        file_extension=opt.file_extension,
        # export directory to compy images to
        destination=opt.destination,
        # directory or list of extra files
        extra_files=opt.extra,
        # threshold
        threshold=opt.threshold,
        # linear de/increasing scheduler
        th_scheduler_window=opt.th_scheduler_window,
        th_scheduler_factor=opt.th_scheduler_factor,
        # find similar or distinct images
        find_distinct=opt.find_distinct,
        # delete not used files afterwards?
        delete_files=opt.delete_files,
        # how many files to scan?
        max_n_files=opt.max_n_files
    )
