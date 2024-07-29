# ImageProcessingTools

This project provides Python-based command line functions to sort and deduplicate images.
Measuring the similarity or difference between images is done using the feature of a small MobileNetV3 which was pretrained on ImageNet data. (The python libraries *torch* and *torchvision* are used for this.)

## Quick start
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
Essentially, the function works in the same way as [select_images.py](select_images.py) and may be eliminated for redundancy in future.

### Project structure
````
ImageProcessingTools
+-- docs  # additional documents and media files for this README-file
+-- utils  # helper functions
    |-- describe_images.py  # builds the descriptor model
    |-- casting.py  # handles casting commandline imput to python data types
|-- delete_files.py  # provides a command line function to walk through a folder deleting files
|-- eliminate_duplicates.py  # provides a command line function to copy unique images to a folder
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