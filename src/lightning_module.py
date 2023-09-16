import timm
import torch

from pytorch_lightning import LightningModule
from src.config import Config
from src.losses import get_criterion
from src.metrics import get_metric_collection
from src.types import BATCH_IMAGE_TENSOR
from src.types import BATCH_LABEL_TENSOR
from typing import Any
from typing import Tuple


class PlanetClassificationModel(LightningModule):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self._config = config
        self._model = timm.create_model(
            model_name=config.model.name,
            pretrained=config.model.pretrained,
            num_classes=config.model.num_labels,
        )
        self._activation = torch.nn.Sigmoid()
        metric = get_metric_collection(metrics=self._config.metric, num_labels=self._config.model.num_labels)
        self._valid_metrics = metric.clone('valid.')
        self._test_metrics = metric.clone('test.')
        self._criterion = get_criterion(criterion_config=self._config.criterion)

        self.save_hyperparameters()

    def forward(self, tensor: BATCH_IMAGE_TENSOR) -> BATCH_LABEL_TENSOR:
        return self._activation(self._model(tensor))

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.Adam(params=self._model.parameters(), **self._config.optimizer.kwargs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, **self._config.scheduler.kwargs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self._config.metric.monitor_metric,
                'interval': 'epoch',
                'frequency': 1,
            },
        }

    def training_step(self, batch: Tuple[BATCH_IMAGE_TENSOR, BATCH_LABEL_TENSOR], batch_idx: int) -> torch.Tensor:
        images, labels = batch
        loss, logs = self._criterion(probas=self.forward(images), labels=labels, prefix='train')
        self._log_criterion(logs=logs)
        return loss

    def validation_step(self, batch: Tuple[BATCH_IMAGE_TENSOR, BATCH_LABEL_TENSOR], batch_idx: int) -> None:
        images, labels = batch
        probas = self.forward(images)
        loss, logs = self._criterion(probas=probas, labels=labels, prefix='valid')
        self._log_criterion(logs=logs)
        self._valid_metrics(preds=probas, target=labels)

    def test_step(self, batch: Tuple[BATCH_IMAGE_TENSOR, BATCH_LABEL_TENSOR], batch_idx: int) -> None:
        images, labels = batch
        probas = self.forward(images)
        self._test_metrics(preds=probas, target=labels)

    def on_validation_epoch_start(self) -> None:
        self._valid_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self._valid_metrics.compute(), on_epoch=True)

    def on_test_epoch_end(self) -> None:
        self.log_dict(self._test_metrics.compute(), on_epoch=True)

    def _log_criterion(self, logs: list[dict[str, float]]) -> None:
        for log in logs:
            self.log(**log)
