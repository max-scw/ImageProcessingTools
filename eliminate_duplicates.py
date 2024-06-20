from pathlib import Path
import shutil
from PIL import Image

from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


from typing import Union, List, Dict

import subprocess


def get_file_paths_with_dir(root_dir: str, file_extension: str):
    command = f'dir /S /B "{root_dir}\\*.{file_extension.strip(".")}"'
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    paths = result.stdout.splitlines()
    return paths


def build_descriptor_model():
    # Load a pre-trained model
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    # Remove the classification layer (we only want the feature extractor)
    model = nn.Sequential(*list(model.children())[:-1])
    # Set the model to evaluation mode
    model.eval()

    # set up preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return model, preprocess


def describe_image(p2img: Union[Path, str], model, preprocess) -> torch.tensor:
    # load image as color image (preprocessing expects an image with 3 channels)
    img = Image.open(p2img).convert("RGB")
    # preprocess image to apply the NN
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension
    # Encode the image using the pre-trained model
    with torch.no_grad():
        encoded_features = model(input_batch)
    return encoded_features.squeeze()


def describe_images(files: List[Union[Path, str]]) -> Dict[Path, torch.tensor]:
    model, preprocess = build_descriptor_model()

    image_features = dict()
    for p2img in tqdm(files, desc="Describe images"):
        image_features[p2img] = describe_image(p2img, model, preprocess)
    return image_features


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


