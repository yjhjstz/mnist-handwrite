import sys, os
import numpy as np
import requests
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

from tensorflow_serving.apis import predict_pb2
sys.path.append("./gen_protos")
import caffe2_service_pb2
import utils


def inference_pytorch(input, host, token=''):
    spec = caffe2_service_pb2.ModelSpec()
    spec.name = 'mnist'

    request = caffe2_service_pb2.PredictRequest()
    response = caffe2_service_pb2.PredictResponse()
    request.model_spec.CopyFrom(spec)

    array = input.reshape(1,1,28,28).astype(np.float32)
    request.inputs['input_1'].CopyFrom(utils.NumpyArrayToCaffe2Tensor(array))
    data = request.SerializeToString()

    data_type = "application/proto"
    headers = {
        # !!! set content type 
        'content-type': data_type,
        # !!! replace your token
        'Authorization': "Bearer " + token
    }

    url = "http://" + host + "/v1/model/pytorch/predict"
    res = requests.post(url=url,
                        data=data,
                        headers=headers)
    if (res.status_code == 200 and res.headers['Content-Type'] == data_type):
        response.ParseFromString(res.content)
        v = response.outputs['0'].float_data
        return np.array(v).flatten().tolist()
    else:
        print(res.content)


def inference_tf(input, host, token=''):
    request = predict_pb2.PredictRequest()
    response = predict_pb2.PredictResponse()

    request.model_spec.name = 'mnist2'
    request.model_spec.signature_name = 'predict_images'

    request.inputs['images'].CopyFrom(
        tf.make_tensor_proto(input, shape=[1, 784]))

    data = request.SerializeToString()

    data_type = "application/proto"
    headers = {
        # !!! set content type 
        'content-type': data_type,
        # !!! replace your token
        'Authorization': "Bearer " + token
    }
    url = "http://" + host + "/v1/model/tensorflow/predict"

    res = requests.post(url, data, headers=headers, verify=False)
    if (res.status_code == 200 and res.headers['Content-Type'] == data_type):
        # print res.content
        response.ParseFromString(res.content)
        v = response.outputs['scores'].float_val
        return np.array(v).flatten().tolist()
    else:
        # handle error msg
        print(res.content)

# webapp
app = Flask(__name__)
app.config['HOST'] = os.getenv("host")

@app.route('/api/mnist', methods=['POST'])
def mnist():
    host = app.config['HOST']
    #input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784).astype(np.float32)
    input = (np.array(request.json, dtype=np.uint8) / 255.0).reshape(1, 784).astype(np.float32)
    output1 = inference_pytorch(input, host)
    output2 = inference_tf(input, host)
    return jsonify(results=[output1, output2])


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
