'''For testing purposes, define a script to make a call to the BentoML service'''

from typing import Tuple
import json
import numpy as np
import pandas as pd
import requests
import logging 

SERVICE_URL = 'http://localhost:3000/classify'

def make_request_to_bento_service(service_url, input_array):
    serialised_data = json.dumps(input_array.values.tolist())
    response = requests.post(
        service_url,
        data=serialised_data,
        headers={'content-type':'application/json'}
    )
    return response.text

def main(x):
    logging.info('START. Beginning trial of BentoML Service.')
    prediction = make_request_to_bento_service(SERVICE_URL, x)
    flowers = {0:'setosa', 1: 'virginica', 2: 'versicolor'}
    print(f'Prediction: {flowers[prediction]}')

if __name__ == '__main__':
    fake_data = pd.DataFrame([
            [1,2,3,4],
            [4,3,2,1],
            [10,20,30,40],
            [4.5,3.5,2.5,1.5]
            ])
    logging.basicConfig(level=logging.DEBUG)

    main(fake_data)