### Данные

Включает 40479 снимков и 24 лейбла.
Скачать данные и их разметку можно по ссылке с [Kaggle](https://www.kaggle.com/datasets/nikitarom/planets-dataset).

Либо при помощи команды:
   ```
   make download_train_data
   make preprocess_train_data
   ```

Структура папки с данными должна быть следующей:
   ```
   data:
      test - Неразмеченные снимки:
         ...
      train - Размеченные снимки:
         image1.jpg
         image2.jpg
         ...
      train_classes.csv - Файл с разметкой снимков        
   ```

### Подготовка проекта 
1. Создание и активация окружения
    ```
    python3.10 -m venv /path/to/new/virtual/environment
    ```
    ```
    source /path/to/new/virtual/environment/bin/activate
    ```

2. Установка пакетов

    В активированном окружении:
    ```
   pip install -U pip 
   pip install -r requirements.txt
    ```

3. Настройка ClearML

Вводим команду, далее следуем инструкциям:
   ```
   clearml-init
   ```

4. Конфигурация

Для изменения конфигурации обучения необходимо поменять параметры файла [config_file_name.yaml](configs/config-baseline.yaml)
Важно указать полный путь до папки с данными в поле `data.data_dir`.

5. Запуск обучения
Для запуска обучения запускаем скрипт (необходимо изменить названия файла конфигурации)
   ```
   PYTHONPATH=. python src/train.py configs/config_file_name.yaml
   ```
6. Сохранение весов модели

Изменить название файла с лучшими весами модели
   ```
   make update_base_model FILE=FILENAME
   ```

### Использование готовой модели

1. Загрузка весов

   ```
   dvc pull
   ```

2. Использование (будет загружать конфигурацию из файла config-best.py и веса из best.pt)

   ```
   PYTHONPATH=. python src/inference.py path/to/image
   ```
