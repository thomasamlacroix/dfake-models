# import numpy as np
# import pandas as pd

from pathlib import Path
# from colorama import Fore, Style


from dfake.params import *
from dfake.dl_logic.model import initialize_model, compile_model, train_model, evaluate_model
from dfake.dl_logic.registry import load_model, save_model, save_results


def train(learning_rate=0.001,
          batch_size = 256,
          patience = 2
          ) -> float:
    """

    """

    print("\n⭐️ Use case: train")
    print("\nLoading preprocessed validation data...")


    #Load data



    # Train model using `model.py`
    model = initialize_model(input_shape=X_train_processed.shape[1])

    model = compile_model(model, learning_rate=learning_rate)

    model, history = train_model(model,
                                 X_train_processed,
                                 y_train,
                                 batch_size=batch_size,
                                 patience=patience,
                                 validation_data=(X_val_processed, y_val)
                                 )

    val_mae = np.min(history.history['val_mae'])

    params = dict(
        context="train",
        training_set_size=DATA_SIZE,
        row_count=len(X_train_processed),
    )

    # Save results on the hard drive using taxifare.ml_logic.registry
    save_results(params=params, metrics=dict(mae=val_mae))

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model)

    print("✅ train() done \n")

    return val_mae


def evaluate(
        min_date:str = '2014-01-01',
        max_date:str = '2015-01-01',
        stage: str = "Production"
    ) -> float:
    """
    Evaluate the performance of the latest production model on processed data
    Return MAE as a float
    """
    print(Fore.MAGENTA + "\n⭐️ Use case: evaluate" + Style.RESET_ALL)

    model = load_model(stage=stage)
    assert model is not None

    min_date = parse(min_date).strftime('%Y-%m-%d') # e.g '2009-01-01'
    max_date = parse(max_date).strftime('%Y-%m-%d') # e.g '2009-01-01'

    # Query your BigQuery processed table and get data_processed using `get_data_with_cache`
    query = f"""
        SELECT *
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_{DATA_SIZE}
        WHERE pickup_datetime BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY pickup_datetime
        """

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
