from omegaconf import OmegaConf
from pydantic import BaseModel
from typing import List


class ProjectConfig(BaseModel):
    name: str
    seed: int
    experiment: str
    log_every_n_steps: int


class ModelConfig(BaseModel):
    name: str
    num_labels: int
    pretrained: bool
    n_epochs: int
    accelerator: str
    device: int


class AugmentationConfig(BaseModel):
    name: str
    kwargs: dict


class DataConfig(BaseModel):
    data_path: str
    num_workers: int
    train_fraction: float
    train_batch_size: int
    valid_batch_size: int
    width: int
    height: int
    augmentations: List[AugmentationConfig]


class MetricConfig(BaseModel):
    name: str
    kwargs: dict


class MetricsConfig(BaseModel):
    task: str
    threshold: float
    monitor_metric: str
    monitor_mode: str
    metrics: List[MetricConfig]


class CriterionConfig(BaseModel):
    name: str
    weight: float
    kwargs: dict


class OptimizerConfig(BaseModel):
    name: str
    kwargs: dict


class SchedulerConfig(BaseModel):
    name: str
    kwargs: dict


class Config(BaseModel):
    project: ProjectConfig
    model: ModelConfig
    data: DataConfig
    metric: MetricsConfig
    criterion: List[CriterionConfig]
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)
