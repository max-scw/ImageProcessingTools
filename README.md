# ImageProcessingTools

This project provides Python-based command line functions to sort and deduplicate images.
Measuring the similarity or difference between images is done using the feature of a small MobileNetV3 which was pretrained on ImageNet data. (The python libraries *torch* and *torchvision* are used for this.)

It further provides some handy functions to process folders with images (`--source`) in general, e.g. convert the image format (`convert_image.py`) or crop them to a defined size (`crop_images.py`).


## Quick start
### Functionality
| file                      | description                                                                                                      | arguments                                                                                                                                                                                            |
|---------------------------|------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `convert_images.py`       | convert all images in a folder to a specified image format                                                       | `--source`, `--destination`, `--file-extension`, `--target-file-type`                                                                                                                                |
| `crop_images.py`          | crops all images and their corresponding bounding boxes by corordinates or factors in [left, top, right, bottom] | `--source`, `--destination`, `--file-extension`, `--roi`, `--bb-format`                                                                                                                              |
| `delete_files.py`         | deletes `--max-n-files` in a folder                                                                              | `--source`, `-max-n-files`, `--logging-level`                                                                                                                                                        |
| `eliminate_duplicates.py` | Copies distinct (similar) images from `--source` folder to `--destination` folder ** depreciated**               | `--source`, `--destination`, `--file-extension`, `--extra`, `--find-distinct`, `--logging-level`                                                                                                     |
| `select_images.py`        | Copies distinct (similar) images from `--source` folder to `--destination` folder                                | `--source`, `--destination`, `--file-extension`, `--extra`, `--threshold`, `--th-scheduler-window` , `--th-scheduler-factor`, `-find-distinct`, `-delete-files`, `--max-n-files` , `--logging-level` |

### Installation
Set up a virtual environment installing all [requirements.txt](requirements.txt).
````shell
python venv .venv  # create virtual environment
source ./.venv/bin/activate  # activate virtual environment (linux)
pip install -r requirements.txt  # install requirements
````
### Usage
#### Selecting distinct / similar images
The file [select_images.py](select_images.py) provides the utility to compare images in a source folder (`--source`) to the files in an extra folder (`--extra`) and copies the images according to a [root mean square (rmse)](https://en.wikipedia.org/wiki/Root_mean_square) threshold (`--threshold`) to another folder (`--destination`).
One switches between finding similar images, which the default case, to finding distinct images with the flag `--find-distinct`.


Finding similar images:
````shell
python select_images.py --source ./data  --file-extension .jpg --destination ./reference --extra ./output --threshold 0.30
````

Finding distinct images:
````shell
python select_images.py --source ./data  --file-extension .jpg --destination ./output --extra ./output --find-distinct --threshold 0.30
````
![ImageProcessingTools.gif](docs%2FImageProcessingTools_select_images_find-distinct.gif)

There are further options, such as specifying the logging level, the maximum number of files that should be searched, or that the not-copied files should be deleted. Both comes in handy when working with *very* large data.

#### Deleting files
Deletes all or `--max-n-files` in a `--source` directory (optionally with `--file-extension`).
````shell
python delete_files.py --source ./data
````

#### Eliminating duplicates
Uses Mean-Squared-Error to find duplicates in a `--source` directory and copies the unique images to a `--destination` directory.
Essentially, the function works in the same way as [select_images.py](select_images.py) and may be eliminated for redundancy in the future.

#### Convert image format
````shell
python contert_images.py --source ./data --destination ./data_new --file-extension .bmp --target-file-type .jpg
````

#### Crop images (and corresponding bounding-boxes)
The region of interest (ROI, `--roi`) is specified in left, top, bottom right coordinates (it uses the [Image.crop()](https://pillow.readthedocs.io/en/stable/reference/Image.html) function from the python package PIL). The coordinates may be given as absolute pixel values or as relative values. If there exists a textfile with the dame name as the image file, it is assumed that this file contains the corresponding bounding boxes and crops them as well.
````shell
python crop_images.py --source ./data --destination ./data_crop --file-extension .jpg --roi 0.2 0 0.8 1 --bb-format xywh
````

### Project structure
````
ImageProcessingTools
+-- docs  # additional documents and media files for this README-file
+-- utils  # helper functions
    |-- bbox.py  # transformations of bounding box representations
    |-- casting.py  # handles casting commandline imput to python data types
    |-- describe_images.py  # builds the descriptor model
|-- convert_images.py  # command line function to concert images to a different image format
|-- crop_images.py  # command line function to crop images and bounding boxes
|-- delete_files.py  # command line function to walk through a folder deleting files
|-- eliminate_duplicates.py  # command line function to copy unique images to a folder
|-- LICENSE
|-- README.md
|-- requirements.txt
|-- select_images.py  # provides a command line function to find distinct or similar images
````


## Notes
Functions are tested on Python 3.11.

## Authors and acknowledgment
- max-scw

## Project status
active (more or less)