import config
import torchmetrics
import typing

from src import utilities


def get_metric_collection(
    metrics: typing.Union[config.MetricsConfig, typing.List[config.MetricsConfig]],
    num_labels: int,
) -> torchmetrics.MetricCollection:

    kwargs = {'task': metrics.task, 'num_classes': num_labels, 'num_labels': num_labels}

    return torchmetrics.MetricCollection(metrics={
        metric.name.split('.')[-1]: utilities.load_object(object_path=metric.name)(**kwargs, **metric.kwargs)
        for metric in metrics.metrics
    })
