from flask import Flask, request
import os 
import sys
import numpy as np

#To find the utils.utils package
testdir = os.getcwd()
sys.path.insert(0, "/".join(testdir.split("/")[:-1] + ["mnist"]))

from utils import utils


app = Flask(__name__)
clf_svm = utils.load('../mnist/models/svm_gamma_0.001.pkl')

clf_dt = utils.load('../mnist/models/dtree_maxd_10.pkl')


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/svm_predict", methods=['POST', 'GET'])
def predict_svm():
    input_json = request.json
    image = input_json['image']
    image = np.array(image).reshape(1, -1)
    predicted = clf_svm.predict(image)
    return str(predicted[0])

@app.route("/decision_tree_predict", methods=['POST', 'GET'])
def predict_dt():
    input_json = request.json
    image = input_json['image']
    image = np.array(image).reshape(1, -1)
    predicted = clf_dt.predict(image)
    return str(predicted[0])

app.run('0.0.0.0', debug = True, port = '5000')