"""
    Training and deploying models at scale
"""
import os
import keras 
import numpy as np 
import tensorflow as tf
from some_model import MODEL_PATH, X_NEW
import sys 
import json 
import requests 

CURR_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__))) 
LOAD_MODEL = False 
SERVER_URL = 'http://localhost:8501/v1/models/my_model:predict'

headers = {"content-type": "application/json"}
json_i = json.dumps({"signature_name": "serving_default", "instances": X_NEW[0:5].tolist() })
print("response")
# write json to file  TBD 
json_response = requests.post(SERVER_URL, data=json_i, headers=headers)
pred = json.loads( json_response.text )
print(pred)