from flask import Flask, request
import os 
import sys
import numpy as np

#To find the utils.utils package
testdir = os.getcwd()
sys.path.insert(0, "/".join(testdir.split("/")[:-1] + ["mnist"]))

from utils import utils


app = Flask(__name__)
clf = utils.load('/Users/shankarjha/Documents/Personal/MTech/Semester 4/MLOPS/mlops-mnist/mnist/models/svm_gamma_0.001.pkl')

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/predict", methods=['POST', 'GET'])
def predict():
    input_json = request.json
    image = input_json['image']
    image = np.array(image).reshape(1, -1)
    predicted = clf.predict(image)
    return str(predicted[0])

app.run('0.0.0.0', debug = True, port = '5000')