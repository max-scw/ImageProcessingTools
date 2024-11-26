from .casting import (
    cast_logging_level,
    cast
)

from .bbox import (
    xywh2xyxy,
    xyxy2xywh
)

from .describe_images import (
    build_descriptor_model,
    describe_image,
    describe_images
)

import logging
import sys

from typing import Union


def setup_logging(
        name: str,
        level: Union[str, int] = logging.INFO,
) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(name)

__all__ = [
    # casting
    cast_logging_level,
    cast,
    # bbox
    xywh2xyxy,
    xyxy2xywh,
    # describe images
    build_descriptor_model,
    describe_image,
    describe_images,
    # local
    setup_logging
]

