import os

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
WEIGHTS_PATH = os.path.join(os.path.join(PROJECT_PATH, 'output'), 'weights')
ONNX_PATH = os.path.join(os.path.join(PROJECT_PATH, 'output'), 'onnx')
PL_LOGS_PATH = os.path.join(PROJECT_PATH, 'lightning_logs')
CLEARML_PATH = os.path.join(PROJECT_PATH, 'experiments')

INPUT_ANNOTATIONS_NAME = 'train_classes.csv'
TRAIN_ANNOTATIONS_NAME = 'train_annotations.csv'
VALID_ANNOTATIONS_NAME = 'valid_annotations.csv'
TEST_ANNOTATIONS_NAME = 'test_annotations.csv'

CLASSES: tuple = (
    'agriculture',
    'artisinal_mine',
    'bare_ground',
    'blooming',
    'blow_down',
    'clear',
    'cloudy',
    'conventional_mine',
    'cultivation',
    'habitation',
    'haze',
    'partly_cloudy',
    'primary',
    'road',
    'selective_logging',
    'slash_burn',
    'water',
)
