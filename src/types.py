import albumentations
import torchtyping

from typing import Union

IMAGE_TENSOR = torchtyping.TensorType['HEIGHT', 'WEIGHT', 'CHANNELS', float]
LABEL_TENSOR = torchtyping.TensorType['NUM_LABELS', 'LABEL_FLAG', float]

BATCH_IMAGE_TENSOR = torchtyping.TensorType['BATCH', 'HEIGHT', 'WEIGHT', 'CHANNELS', float]
BATCH_LABEL_TENSOR = torchtyping.TensorType['BATCH', 'NUM_LABELS', 'LABEL_FLAG', float]

TRANSFORM_TYPE = Union[albumentations.BasicTransform, albumentations.BaseCompose]
