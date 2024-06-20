from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

from PIL import Image

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--source", type=str, help="Directory where the images are stored.")
    parser.add_argument("--destination", type=str, default="",
                        help="Directory to copy the selected images to")
    parser.add_argument("--file-extension", type=str, default="", help="File type")

    parser.add_argument("--resize-factor", type=float, default=1.0, help="Factor by which the images should be resized.")
    parser.add_argument("--target-file-type", type=str, default="", help="File type")

    opt = parser.parse_args()

    opt.source = r"C:\Users\schwmax\Proj\Coding\YOLOv7_scw\dataset\Damper300\data1"
    opt.file_extension = ".bmp"
    opt.destination = "selected_images"
    opt.target_file_type = ".jpg"

    destination = Path(opt.destination)

    files = list(Path(opt.source).glob("*." + opt.file_extension.strip(".") if opt.file_extension else "*"))
    for fl in tqdm(files):
        # load file
        img = Image.open(fl)
        img_rzd = img
        # resized = img.resize((opt.resize_factor, opt.resize_factor))  # TODO integrate resizing
        filename = fl.stem + "." + (opt.target_file_type.strip(".") if opt.target_file_type else fl.suffix)
        img_rzd.save(destination / filename)

