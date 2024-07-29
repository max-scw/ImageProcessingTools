from argparse import ArgumentParser
import shutil
from pathlib import Path
from tqdm import tqdm
import logging
import sys

from utils.casting import cast_logging_level


from typing import List, Union, Generator


def delete_files(files2delete: Union[List[Path], Generator], n: int = None):

    if n is None or (isinstance(n, int) and n < 1):
        if not isinstance(files2delete, list):
            files2delete = list(files2delete)
        n = len(files2delete)

    logging.info(f"Deleting {n} files")
    for i, p2fl in enumerate(tqdm(files2delete, total=n)):
        if i >= n:
            break
        p2fl.unlink()
        shutil.rmtree(p2fl.parent)
    logging.info(f"Done deleting unused files.")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--source", type=str, help="Directory where the files are stored.")
    parser.add_argument("--file-extension", type=str, default="", help="File type")

    parser.add_argument("--max-n-files", type=int, default=None, help="Number of files to scan through.")

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

    delete_files(
        files2delete=Path(opt.source).glob(f"**/*.{opt.file_extension.strip('.')}"),
        n=opt.max_n_files
    )
