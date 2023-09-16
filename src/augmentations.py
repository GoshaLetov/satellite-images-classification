import albumentations

from albumentations.pytorch import ToTensorV2
from config import AugmentationConfig
from config import DataConfig
from src import utilities
from typing import List
from typing import Union


TRANSFORM_TYPE = Union[albumentations.BasicTransform, albumentations.BaseCompose]


def get_augmentations(augmentations: List[AugmentationConfig]) -> List:
    return [
        utilities.load_object(object_path=f'albumentations.{augmentation.name}')(**augmentation.kwargs)
        for augmentation in augmentations
    ]


def get_transforms(
    config: DataConfig,
    preprocessing: bool = True,
    augmentations: bool = True,
    postprocessing: bool = True,
) -> TRANSFORM_TYPE:

    transforms = []

    if preprocessing:
        transforms.append(albumentations.Resize(height=config.height, width=config.width))

    if augmentations:
        transforms.extend(get_augmentations(augmentations=config.augmentations))

    if postprocessing:
        transforms.extend([albumentations.Normalize(), ToTensorV2()])

    return albumentations.Compose(transforms)
