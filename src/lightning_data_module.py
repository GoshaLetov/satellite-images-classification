import logging
import os
import pandas as pd

from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import MultiLabelBinarizer
from src import constants
from src.augmentations import get_transforms
from src.config import DataConfig
from src.dataset import PlanetDataset
from src.dataset_splitter import stratify_shuffle_split_subsets
from torch.utils.data import DataLoader
from typing import Optional


def split_and_save_annotations(data_path: str, train_fraction: float = 0.8) -> None:
    annotations = pd.read_csv(os.path.join(data_path, constants.INPUT_ANNOTATIONS_NAME))

    encoder = MultiLabelBinarizer(classes=constants.CLASSES, sparse_output=False)

    logging.info(f'Original dataset: {len(annotations)}')
    annotations = pd.DataFrame(
        data=encoder.fit_transform(annotations.tags.str.split(' ')).astype(int),
        columns=list(constants.CLASSES),
        index=annotations.image_name,
    )
    logging.info(f'Final dataset: {len(annotations)}')

    stratified_annotations = stratify_shuffle_split_subsets(annotations=annotations, train_fraction=train_fraction)
    train_annotations, valid_annotations, test_annotations = stratified_annotations

    logging.info(f'Train dataset: {len(train_annotations)}')
    logging.info(f'Valid dataset: {len(valid_annotations)}')
    logging.info(f'Test dataset: {len(test_annotations)}')

    train_annotations.to_csv(os.path.join(data_path, constants.TRAIN_ANNOTATIONS_NAME), index=False)
    valid_annotations.to_csv(os.path.join(data_path, constants.VALID_ANNOTATIONS_NAME), index=False)
    test_annotations.to_csv(os.path.join(data_path, constants.TEST_ANNOTATIONS_NAME), index=False)

    logging.info('Datasets successfully saved!')


def read_annotations(data_path: str, file: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_path, file)).set_index(keys=['image_name'])


class PlanetDataModule(LightningDataModule):

    def __init__(self, data_config: DataConfig) -> None:
        super().__init__()
        self._config = data_config

        self.train_dataset: Optional[PlanetDataset] = None
        self.valid_dataset: Optional[PlanetDataset] = None
        self.test_dataset: Optional[PlanetDataset] = None

    def prepare_data(self) -> None:
        is_files_exists = all([
            os.path.exists(os.path.join(self._config.data_path, constants.TRAIN_ANNOTATIONS_NAME)),
            os.path.exists(os.path.join(self._config.data_path, constants.VALID_ANNOTATIONS_NAME)),
            os.path.exists(os.path.join(self._config.data_path, constants.TEST_ANNOTATIONS_NAME)),
        ])
        if not is_files_exists:
            split_and_save_annotations(data_path=self._config.data_path, train_fraction=self._config.train_fraction)

    def setup(self, stage: str) -> None:

        if stage == 'fit':
            train_annotations = read_annotations(
                data_path=self._config.data_path,
                file=constants.TRAIN_ANNOTATIONS_NAME,
            )
            valid_annotations = read_annotations(
                data_path=self._config.data_path,
                file=constants.VALID_ANNOTATIONS_NAME,
            )

            self.train_dataset = PlanetDataset(
                annotations=train_annotations,
                image_path=os.path.join(self._config.data_path, 'train'),
                transform=get_transforms(config=self._config, augmentations=True),
            )
            self.valid_dataset = PlanetDataset(
                annotations=valid_annotations,
                image_path=os.path.join(self._config.data_path, 'train'),
                transform=get_transforms(config=self._config, augmentations=False),
            )

        if stage == 'test':
            test_annotations = read_annotations(
                data_path=self._config.data_path,
                file=constants.TEST_ANNOTATIONS_NAME,
            )
            self.test_dataset = PlanetDataset(
                annotations=test_annotations,
                image_path=os.path.join(self._config.data_path, 'train'),
                transform=get_transforms(config=self._config, augmentations=False),
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self._config.train_batch_size,
            num_workers=self._config.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self._config.valid_batch_size,
            num_workers=self._config.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self._config.valid_batch_size,
            num_workers=self._config.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
