import cv2
import os
import pandas as pd

from src.types import IMAGE_TENSOR
from src.types import LABEL_TENSOR
from src.types import TRANSFORM_TYPE
from torch.utils.data import Dataset
from typing import Optional
from typing import Tuple


class PlanetDataset(Dataset):
    def __init__(self, annotations: pd.DataFrame, image_path: str, transform: Optional[TRANSFORM_TYPE] = None) -> None:
        super().__init__()
        self.annotations = annotations
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item: int) -> Tuple[IMAGE_TENSOR, LABEL_TENSOR]:
        sample = self.annotations.iloc[[item]]
        image_path = f'{os.path.join(self.image_path, sample.index.values[0])}.jpg'
        sample = {
            'image': cv2.cvtColor(src=cv2.imread(filename=image_path), code=cv2.COLOR_BGR2RGB),
            'label': sample.values.ravel(),
        }

        if self.transform:
            sample = self.transform(**sample)

        return sample['image'], sample['label']
