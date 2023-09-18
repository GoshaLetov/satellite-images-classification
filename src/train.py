import argparse
import logging
import os
import shutil
import torch

from clearml import Task
from pytorch_lightning import (
    Trainer,
    seed_everything,
)
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from src import constants
from src.config import Config
from src.lightning_data_module import PlanetDataModule
from src.lightning_module import PlanetClassificationModel


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    return parser.parse_args()


def train(run_config: Config):
    datamodule = PlanetDataModule(data_config=run_config.data)
    model = PlanetClassificationModel(config=run_config)

    task = Task.init(
        project_name=run_config.project.name,
        task_name=run_config.project.experiment,
        auto_connect_frameworks=True,
    )
    task.connect(run_config.model_dump())

    experiment_save_path = os.path.join(constants.CLEARML_PATH, run_config.project.experiment)
    os.makedirs(experiment_save_path, exist_ok=True)
    os.makedirs(constants.WEIGHTS_PATH, exist_ok=True)
    os.makedirs(constants.ONNX_PATH, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        experiment_save_path,
        monitor=run_config.metric.monitor_metric,
        mode=run_config.metric.monitor_mode,
        save_top_k=1,
        filename=f'epoch_{{epoch:02d}}-{{{run_config.metric.monitor_metric}:.3f}}',
    )

    trainer = Trainer(
        max_epochs=run_config.model.n_epochs,
        accelerator=run_config.model.accelerator,
        devices=[run_config.model.device],
        log_every_n_steps=run_config.project.log_every_n_steps,
        callbacks=[
            checkpoint_callback,
            EarlyStopping(
                monitor=run_config.metric.monitor_metric,
                patience=4,
                mode=run_config.metric.monitor_mode,
            ),
            LearningRateMonitor(logging_interval='epoch'),
        ],
        logger=TensorBoardLogger(
            save_dir=constants.PL_LOGS_PATH,
            name=run_config.project.experiment,
        ),
    )

    torch.set_float32_matmul_precision(precision='high')

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(ckpt_path=checkpoint_callback.best_model_path, datamodule=datamodule)

    model = model.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path)

    torch.save(
        obj=model.state_dict(),
        f=os.path.join(constants.WEIGHTS_PATH, f'{run_config.project.experiment}.pt'),
    )
    model.to_onnx(
        file_path=os.path.join(constants.ONNX_PATH, f'{run_config.project.experiment}.onnx'),
        input_sample=torch.randn(1, 3, config.data.height, config.data.width),
        input_names=['input'],
        output_names=['output'],
    )

    shutil.rmtree(path=constants.PL_LOGS_PATH)
    shutil.rmtree(path=constants.CLEARML_PATH)


if __name__ == '__main__':
    args = arg_parse()
    config = Config.from_yaml(path=args.config_file)
    logging.basicConfig(level=logging.INFO)
    seed_everything(seed=config.project.seed, workers=True)
    train(run_config=config)
