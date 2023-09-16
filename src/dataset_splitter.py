import logging
import numpy as np
import pandas as pd

from skmultilearn.model_selection.iterative_stratification import IterativeStratification
from typing import Tuple


def _iterative_split(xs: np.array, ys: np.array, distribution: list) -> Tuple[np.array, np.array]:
    stratifier = IterativeStratification(n_splits=2, sample_distribution_per_fold=distribution)

    first_split, second_split = next(stratifier.split(X=xs, y=ys))

    return first_split, second_split


def stratify_shuffle_split_subsets(
    annotations: pd.DataFrame, train_fraction: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Разбиение датасета на train/valid/test."""

    x_columns = ['image_name']
    y_columns = list(annotations.select_dtypes('int').columns)

    all_x = annotations.reset_index()[x_columns].to_numpy()
    all_y = annotations.reset_index()[y_columns].to_numpy()

    train_indexes, else_indexes = _iterative_split(all_x, all_y, distribution=[1 - train_fraction, train_fraction])
    x_train, x_else = all_x[train_indexes], all_x[else_indexes]
    y_train, y_else = all_y[train_indexes], all_y[else_indexes]

    test_indexes, valid_indexes = _iterative_split(x_else, y_else, distribution=[0.5, 0.5])
    x_test, x_valid = x_else[test_indexes], x_else[valid_indexes]
    y_test, y_valid = y_else[test_indexes], y_else[valid_indexes]

    train_subset = pd.DataFrame(data=np.concatenate([x_train, y_train], axis=1), columns=x_columns + y_columns)
    valid_subset = pd.DataFrame(data=np.concatenate([x_valid, y_valid], axis=1), columns=x_columns + y_columns)
    test_subset = pd.DataFrame(data=np.concatenate([x_test, y_test], axis=1), columns=x_columns + y_columns)

    logging.info('Stratifying dataset is completed.')

    return train_subset, valid_subset, test_subset
