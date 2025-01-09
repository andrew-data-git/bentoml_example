'''Define a BentoML service'''
import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

model_name = 'iris-svc'
runner = bentoml.mlflow.get(f'{model_name}:latest').to_runner()
iris_service = bentoml.Service('iris_classifier', runners=[runner])

@iris_service.api(
    input= NumpyNdarray(),
    output= NumpyNdarray()
)

def classify(input_data):
    '''Define the endpoint'''
    return runner.predict.run(input_data)