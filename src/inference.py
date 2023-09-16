import argparse
import cv2
import os
import torch

from lightning_module import PlanetClassificationModel
from src import constants
from src.augmentations import get_transforms
from src.config import Config


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='image path')
    return parser.parse_args()


def infer_best_model(image_path: str):
    config = Config.from_yaml(path=os.path.join(constants.PROJECT_PATH, 'configs/config-best.yaml'))
    weight = torch.load(f=os.path.join(constants.WEIGHTS_PATH, 'best.pt'))

    model = PlanetClassificationModel(config=config)
    model.load_state_dict(state_dict=weight)

    transform = get_transforms(config=config.data, augmentations=False)

    batch = {'image': cv2.cvtColor(src=cv2.imread(filename=image_path), code=cv2.COLOR_BGR2RGB)}

    model.eval()
    with torch.no_grad():
        probas = model.forward(transform(**batch)['image'].unsqueeze(0)).squeeze(0)

    detected_labels_on_image = []
    for label, proba in zip(constants.CLASSES, probas):
        if proba > config.metric.threshold:
            detected_labels_on_image.append(label)

    if not detected_labels_on_image:
        return 'No labels detected'
    return ', '.join(detected_labels_on_image)


if __name__ == '__main__':
    args = arg_parse()
    detected_labels = infer_best_model(image_path=args.image_path)
