import os
import numpy as np

##################  VARIABLES  ##################
DATA_SIZE = os.environ.get("DATA_SIZE")
MODEL_TARGET = os.environ.get("MODEL_TARGET")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
INSTANCE = os.environ.get("INSTANCE")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")
# GAR_IMAGE = os.environ.get("GAR_IMAGE")
# GAR_MEMORY = os.environ.get("GAR_MEMORY")

##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('.'), "data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('.'), "training_outputs")

BATCH_SIZE = 128
IMAGE_SIZE = (256, 256)
IMAGE_HEIGHT = IMAGE_SIZE[0]
IMAGE_WIDTH = IMAGE_SIZE[1]
NUM_CHANNELS = 3
SEED = 42
LEARNING_RATE = 0.001
PATIENCE = 5
EPOCHS = 1




################## VALIDATIONS #################

env_valid_options = dict(
    DATA_SIZE=["lightweight", "all"],
    MODEL_TARGET=["local", "gcs"],
)

def validate_env_value(env, valid_options):
    env_value = os.environ[env]
    if env_value not in valid_options:
        raise NameError(f"Invalid value for {env} in `.env` file: {env_value} must be in {valid_options}")


for env, valid_options in env_valid_options.items():
    validate_env_value(env, valid_options)
