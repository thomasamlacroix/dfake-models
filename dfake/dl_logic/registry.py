import glob
import os
import time
import pickle

import joblib

from google.cloud import storage

import mlflow
# from mlflow.tracking import MlflowClient

from dfake.params import *


def save_results(params, metrics) -> None:
    """
    Persist params & metrics on MLflow
    """
    #Saving to MLflow
    if params is not None:
        mlflow.log_params(params)
    if metrics is not None:
        mlflow.log_metrics(metrics)

    print("✅ Results saved on MLflow")



def save_model(model=None):
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.joblib"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.joblib"
    - Also persist it on MLflow
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.joblib")
    # model.save(model_path)

    # print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":
        model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.joblib" for instance
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")
        return None

    #Saving model to MLflow
    mlflow.tensorflow.log_model(model=model,
                                artifact_path="model",
                                registered_model_name=MLFLOW_MODEL_NAME
                                )

    print("✅ Model saved to mlflow")

    return None


def load_model():
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'

    Return None (but do not Raise) if no model is found
    """

    # if MODEL_TARGET == "local":
    #     print(f"\nLoad latest model from local registry...")

    #     # Get the latest model version name by the timestamp on disk
    #     local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
    #     local_model_paths = glob.glob(f"{local_model_directory}/*")

    #     if not local_model_paths:
    #         return None

    #     most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

    #     print(f"\nLoad latest model from disk...")

    #     latest_model = joblib.load(most_recent_model_path_on_disk)

    #     print("✅ Model loaded from local disk")

    #     return latest_model

    if MODEL_TARGET == "gcs":
        print(f"\nLoad latest model from GCS...")

        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))

        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
            latest_blob.download_to_filename(latest_model_path_to_save)

            latest_model = joblib.load(latest_model_path_to_save)

            print("✅ Latest model downloaded from cloud storage")

            return latest_model
        except:
            print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

            return None

    else:
        return None



def mlflow_run(func):
    """
    Generic function to log params and results to MLflow along with TensorFlow auto-logging

    Args:
        - func (function): Function you want to run within the MLflow run
        - params (dict, optional): Params to add to the run in MLflow. Defaults to None.
        - context (str, optional): Param describing the context of the run. Defaults to "Train".
    """
    def wrapper(*args, **kwargs):
        mlflow.end_run()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT)

        with mlflow.start_run():
            mlflow.tensorflow.autolog()
            results = func(*args, **kwargs)

        print("✅ mlflow_run auto-log done")

        return results
    return wrapper
