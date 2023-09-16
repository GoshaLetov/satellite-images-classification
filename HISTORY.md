## История экспериментов

### 10.09.2023. Эксперименты:
- [Learning rate](https://app.clear.ml/projects/f505bad049b04cd19b3661d25df2ed62/experiments/f3808a955ba241ccb92ccd280ba8240d/output/execution)

Выводы:
- Целевая метрика F1-Score
- Выше Learning rate

### 10.09.2023. Эксперименты:
- [Аугментации](https://app.clear.ml/projects/f505bad049b04cd19b3661d25df2ed62/experiments/d6f934f7e55a4c10b0a0463973b6589b/output/execution)
- [Модель](https://app.clear.ml/projects/f505bad049b04cd19b3661d25df2ed62/experiments/d6f934f7e55a4c10b0a0463973b6589b/output/execution)

Выводы:
- EfficientNetB1 дает скор выше EfficientNetB0 (F1, но Accuracy меньше)
- Сильные аугментации помогают

### [Baseline](https://app.clear.ml/projects/f505bad049b04cd19b3661d25df2ed62/experiments/06bd792ccdb840ad9d7ea13c58e7c80c/output/execution):
Параметры:
- Accuracy: 0.9534
- Модель: EfficientNetB0
- AdamW: lr = 1e-3, weight_decay = 1e-5
- Аугментации: Сильные
