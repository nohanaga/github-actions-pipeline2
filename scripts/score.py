import os
import json
import logging
from mlflow.pyfunc import load_model

# Called when the service is loaded
def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    logging.info("AZUREML_MODEL_DIR: " + os.environ["AZUREML_MODEL_DIR"])

    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "models")
    model = load_model(model_path)  
    logging.info("Init complete")

def run(mini_batch):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info(f"run method start: {__file__}, run({mini_batch})")

    input = json.loads(mini_batch)["data"]
    logging.info(f"input: {input}")

    predictions = model.predict(input)
    logging.info('Predictions:' + str(predictions))
    logging.info("Request processed")

    return predictions.tolist() # return a dataframe or a list
