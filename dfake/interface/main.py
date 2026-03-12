# import numpy as np
# import pandas as pd

from pathlib import Path
# from colorama import Fore, Style

from keras.utils import image_dataset_from_directory

#Pretrained model for transfer learning
from keras.applications.efficientnet import EfficientNetB3, preprocess_input

from dfake.params import *
from dfake.dl_logic.model import initialize_model, compile_model, train_model, evaluate_model
from dfake.dl_logic.registry import load_model, save_model, save_results

from google.cloud import storage


def train(learning_rate=LEARNING_RATE,
          batch_size=BATCH_SIZE,
          patience=PATIENCE
          ):
    """
    - Get data from GCP bucket or local folder
    - Train model
    - Store training results and model weights
    """

    print("\n⭐️ Use case: train")
    print("\nLoading preprocessed validation data...")


    # client = storage.Client()
    # bucket = client.bucket(BUCKET_NAME)
    # blob = bucket.blob(f"models/{model_filename}")
    #Lightweight dataset
    train_data_dir = Path(LOCAL_DATA_PATH).joinpath(f"{DATA_SIZE}", "train")
    val_data_dir = Path(LOCAL_DATA_PATH).joinpath(f"{DATA_SIZE}", "valid")


    #Load data
    train_ds = image_dataset_from_directory(
    train_data_dir,
    labels="inferred",
    label_mode="binary",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE)


    val_ds = image_dataset_from_directory(
    val_data_dir,
    labels="inferred",
    label_mode="binary",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE)




    # Train model using `model.py`
    input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS)

    base_model = EfficientNetB3(weights="imagenet",
                                include_top=False,
                                input_shape=input_shape)

    model = initialize_model(input_shape, base_model, preprocess_input)

    model = compile_model(model, learning_rate=learning_rate)

    model, history = train_model(model,
                                 train_ds,
                                 batch_size=batch_size,
                                 patience=patience,
                                 validation_data=val_ds
                                 )


    val_accuracy = history.history['accuracy']
    val_recall = history.history['recall']
    val_precision = history.history['precision']

    params = dict(
        context="train",
        training_set_size=DATA_SIZE
    )

    # Save results on the hard drive using dfake.ml_logic.registry
    save_results(params=params, metrics=dict(accuracy=val_accuracy,
                                             recall=val_recall,
                                             precision=val_precision))

    # Save model weight on the hard drive and on GCS
    save_model(model=model)

    print("✅ train() done \n")

    return val_accuracy, val_recall, val_precision


def evaluate():
    """
    Evaluate the performance of the model on test data
    Return accuracy, recall and precision as floats
    """
    print("\n⭐️ Use case: evaluate")

    model = load_model()
    assert model is not None

    test_data_dir = Path(LOCAL_DATA_PATH).joinpath(f"{DATA_SIZE}", "test")

    test_ds = image_dataset_from_directory(
    test_data_dir,
    labels="inferred",
    label_mode="binary",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE)

    metrics_dict = evaluate_model(model, test_ds)
    accuracy = metrics_dict["accuracy"]
    recall = metrics_dict["recall"]
    precision = metrics_dict["precision"]

    params = dict(
        context="evaluate", # Package behavior
        training_set_size=DATA_SIZE
    )

    save_results(params=params, metrics=metrics_dict)

    print("✅ evaluate() done \n")

    return accuracy, recall, precision


def pred(img = None):
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    # model = load_model()
    # assert model is not None

    pass

    # return


if __name__ == '__main__':
    train()
    evaluate()
    pred()
