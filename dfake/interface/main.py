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
    test_data_dir = Path(LOCAL_DATA_PATH).joinpath(f"{DATA_SIZE}", "test")


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

    test_ds = image_dataset_from_directory(
    test_data_dir,
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

    return val_mae


def evaluate(
        min_date:str = '2014-01-01',
        max_date:str = '2015-01-01',
        stage: str = "Production"
    ) -> float:
    """
    Evaluate the performance of the model on test data
    Return accuracy, recall and precision as floats
    """
    print("\n⭐️ Use case: evaluate")

    model = load_model(stage=stage)
    assert model is not None

    # Retrieve `query` data from BigQuery or from `data_query_cache_path` if the file already exists!
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"query_{min_date}_{max_date}_{DATA_SIZE}.csv")
    data_processed = get_data_with_cache(GCP_PROJECT, query, data_query_cache_path)

    if data_processed.shape[0] == 0:
        print("❌ No data to evaluate on")
        return None

    # data_processed = data_processed.to_numpy()

    X_new = data_processed.iloc[:, 1:-1]
    y_new = data_processed.iloc[:, -1]
    print(X_new.shape)
    print(y_new.shape)

    metrics_dict = evaluate_model(model=model, X=X_new, y=y_new)
    mae = metrics_dict["mae"]

    params = dict(
        context="evaluate", # Package behavior
        training_set_size=DATA_SIZE,
        row_count=len(X_new)
    )

    save_results(params=params, metrics=metrics_dict)

    print("✅ evaluate() done \n")

    return mae


def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    if X_pred is None:
        X_pred = pd.DataFrame(dict(
        pickup_datetime=[pd.Timestamp("2013-07-06 17:18:00", tz='UTC')],
        pickup_longitude=[-73.950655],
        pickup_latitude=[40.783282],
        dropoff_longitude=[-73.984365],
        dropoff_latitude=[40.769802],
        passenger_count=[1],
    ))

    model = load_model()
    assert model is not None

    X_processed = preprocess_features(X_pred)
    y_pred = model.predict(X_processed)

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    return y_pred


if __name__ == '__main__':
    train()
    evaluate()
    pred()
