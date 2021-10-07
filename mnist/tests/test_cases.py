from sklearn import datasets

import sys, os, math
import random

#To find the utils.utils package
testdir = os.getcwd()
sys.path.insert(0, "/".join(testdir.split("/")[:-1]))

print(os.getcwd())
path = '../models'
from utils.utils import run_classification_experiment, create_splits, get_performance

def test_equality():
    assert 1 == 1
    
def test_model_writing():
    digits    = datasets.load_digits()
    n_samples = digits.data.shape[0]
    X         = digits.images.reshape(n_samples, -1)
    y         = digits.target

    sample_indices = random.choices(range(n_samples), k = 100)

    X_sample, y_sample =  X[sample_indices], y[sample_indices]
    gamma = 0.01

    model = run_classification_experiment(path, gamma=gamma, 
    X_train = X_sample, 
    y_train=y_sample,
    X_val=X_sample, 
    y_val=y_sample)

    assert os.path.isfile(f'{path}/{model}')

def test_small_data_overfit_checking():
    digits    = datasets.load_digits()
    n_samples = digits.data.shape[0]
    X         = digits.images.reshape(n_samples, -1)


    X_train, X_test, X_val, y_train, y_test, y_val = create_splits(X, digits, test_size=0.3, validation_size_from_test_size=0.5)

    sample_indices = random.choices(range(X_test.shape[0]), k = 100)

    X_sample, y_sample =  X_test[sample_indices], y_test[sample_indices]

    model = run_classification_experiment(path, gamma=0.01, 
    X_train = X_train, 
    y_train=y_train,
    X_val=X_train, 
    y_val=y_train)

    train_metrics = get_performance(path, model, X_train, y_train)
    small_data_metrics = get_performance(path, model, X_sample, y_sample)

    assert train_metrics['acc'] > small_data_metrics['acc'] + 0.1
    assert train_metrics['f1'] > small_data_metrics['f1'] + 0.1

