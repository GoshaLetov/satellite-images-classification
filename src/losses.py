import config
import torch
import typing

from src import utilities


class CriterionWithWeight(torch.nn.Module):
    def __init__(self, criterion: torch.nn.Module, name: str, weight: typing.Optional[float] = 1.0) -> None:
        super().__init__()
        self.criterion = criterion
        self.weight = weight
        self.name = name.split('.')[-1]

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits.to(torch.float32), labels.to(torch.float32)) * self.weight


class Criterion(torch.nn.Module):
    def __init__(self, criterion_list: typing.List[CriterionWithWeight]) -> None:
        super().__init__()
        self.criterion_list = criterion_list

    def forward(
        self,
        probas: torch.Tensor,
        labels: torch.Tensor,
        prefix: str,
    ) -> typing.Tuple[torch.Tensor, dict[str: float]]:

        loss = torch.tensor(0).to(dtype=torch.float32, device=labels.device)
        logs = []
        for criterion in self.criterion_list:
            batch_loss = criterion(probas, labels)
            loss += batch_loss
            logs.append({'name': f'{prefix}.{criterion.name}', 'value': batch_loss.item()})

        if len(self.criterion_list) > 1:
            logs.append({'name': f'{prefix}.TotalLoss', 'value': loss.item()})

        return loss, logs

    def to(self, *args, **kwargs) -> 'Criterion':
        return self


def get_criterion(
    criterion_config: typing.Union[config.CriterionConfig, typing.List[config.CriterionConfig]],
) -> Criterion:

    if not isinstance(criterion_config, list):
        criterion_config = [criterion_config]

    criterion_list = [
        CriterionWithWeight(
            criterion=utilities.load_object(object_path=criterion.name)(**criterion.kwargs),
            name=criterion.name,
            weight=criterion.weight,
        ) for criterion in criterion_config
    ]

    return Criterion(criterion_list=criterion_list)
