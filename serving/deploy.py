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
import grpc 
from tensorflow_serving.apis.predict_pb2 import PredictRequest
from tensorflow_serving.apis import prediction_service_pb2_grpc


tf.__version__ # TensorFlow version: 2.16.1
keras.__version__ # 3.1.1

CURR_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__))) 
LOAD_MODEL = False 
SERVER_URL = "http://localhost:8501/v1/models/my_model:predict" 
VERSION_A = "0001"
VERSION_B = "0002"
MODEL_NAME = "my_model"

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["MODEL_DIR"] = f"{MODEL_PATH}"

tf.saved_model.load(MODEL_PATH)
x = X_NEW[0:3]

def gprc():
    request = PredictRequest()
    request.model_spec.name = MODEL_NAME
    request.model_spec.signature_name = "serving_default"
    x = x.astype('float32')
    request.inputs['inputs'].CopyFrom(tf.make_tensor_proto(x))

    channel = grpc.insecure_channel('localhost:8500')
    service = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    resp = service.Predict(request)
    sys.exit() 
    proto = resp.outputs["output_0"]
    print(proto)

def rest():
    headers = {"content-type": "application/json"}
    x = (X_NEW[0:1])
   
    json_i = json.dumps(
        {
            "signature_name": "serving_default" ,
            "instances": x.tolist(), 
        }
    )
    print("response")
    # write json to file  TBD 
    json_response = requests.post(SERVER_URL, data=json_i, headers=headers)
    pred = json.loads( json_response.text )
    print(pred)
    sys.exit()

rest()