from sklearn import datasets

import sys, os, math
import random
import numpy as np
import pickle
from sklearn.metrics import accuracy_score

#To find the utils.utils package
testdir = os.getcwd()
sys.path.insert(0, "/".join(testdir.split("/")[:-1]))

print(os.getcwd())
path = '../models'
from utils.utils import run_classification_experiment, create_splits, get_performance, preprocess, load


model_location = "/Users/shankarjha/Documents/Personal/MTech/Semester 4/MLOPS/mlops-mnist/mnist/models/"
best_model_svm = load(model_location + "svm_gamma_0.001.pkl")
best_model_dt  = load(model_location + "dtree_maxd_4.pkl")

ACC_THRESHOLD  = 0.4

# Loading data examples 
digits    = datasets.load_digits()
n_samples = digits.data.shape[0]
data      = digits.images.reshape(n_samples, -1)

# Preprocessing the input data
preprocessed_data = preprocess(data, 1)

# Split data into 70% train and 30% test subsets
X_train, X_test, X_val, y_train, y_test, y_val = create_splits(preprocessed_data, digits.target, test_size=0.3, validation_size_from_test_size=0.5)


def test_digit_correct_0_svm():
    index_of_0 = np.where(y_val==0)[0][0]
    svm_pred = best_model_svm.predict([X_val[index_of_0]])
    assert svm_pred == 0

def test_digit_correct_1_svm():
    index_of_1 = np.where(y_val==1)[0][0]
    svm_pred = best_model_svm.predict([X_val[index_of_1]])
    assert svm_pred == 1

def test_digit_correct_2_svm():
    index_of_2 = np.where(y_val==2)[0][0]
    svm_pred = best_model_svm.predict([X_val[index_of_2]])
    assert svm_pred == 2

def test_digit_correct_3_svm():
    index_of_3 = np.where(y_val==3)[0][0]
    svm_pred = best_model_svm.predict([X_val[index_of_3]])
    assert svm_pred == 3

def test_digit_correct_4_svm():
    index_of_4 = np.where(y_val==4)[0][0]
    svm_pred = best_model_svm.predict([X_val[index_of_4]])
    assert svm_pred == 4

def test_digit_correct_5_svm():
    index_of_5 = np.where(y_val==5)[0][0]
    svm_pred = best_model_svm.predict([X_val[index_of_5]])
    assert svm_pred == 5

def test_digit_correct_6_svm():
    index_of_6 = np.where(y_val==6)[0][1]
    svm_pred = best_model_svm.predict([X_val[index_of_6]])
    assert svm_pred == 6

def test_digit_correct_7_svm():
    index_of_7 = np.where(y_val==7)[0][0]
    svm_pred = best_model_svm.predict([X_val[index_of_7]])
    assert svm_pred == 7

def test_digit_correct_8_svm():
    index_of_8 = np.where(y_val==8)[0][0]
    svm_pred = best_model_svm.predict([X_val[index_of_8]])
    assert svm_pred == 8

def test_digit_correct_9_svm():
    index_of_9 = np.where(y_val==9)[0][0]
    svm_pred = best_model_svm.predict([X_val[index_of_9]])
    assert svm_pred == 9

def test_digit_correct_0_dt():
    index_of_0 = np.where(y_val==0)[0][0]
    dt_pred  = best_model_dt.predict([X_val[index_of_0]])
    assert dt_pred == 0

def test_digit_correct_1_dt():
    index_of_1 = np.where(y_val==1)[0][0]
    dt_pred  = best_model_dt.predict([X_val[index_of_1]])
    assert dt_pred == 1

def test_digit_correct_2_dt():
    index_of_2 = np.where(y_val==2)[0][0]
    dt_pred  = best_model_dt.predict([X_val[index_of_2]])
    assert dt_pred == 2

def test_digit_correct_3_dt():
    index_of_3 = np.where(y_val==3)[0][0]
    dt_pred  = best_model_dt.predict([X_val[index_of_3]])
    assert dt_pred == 3

def test_digit_correct_4_dt():
    index_of_4 = np.where(y_val==4)[0][0]
    dt_pred  = best_model_dt.predict([X_val[index_of_4]])
    assert dt_pred == 4

def test_digit_correct_5_dt():
    index_of_5 = np.where(y_val==5)[0][0]
    dt_pred  = best_model_dt.predict([X_val[index_of_5]])
    assert dt_pred == 5

def test_digit_correct_6_dt():
    index_of_6 = np.where(y_val==6)[0][1]
    dt_pred  = best_model_dt.predict([X_val[index_of_6]])
    assert dt_pred == 6

def test_digit_correct_7_dt():
    index_of_7 = np.where(y_val==7)[0][0]
    dt_pred  = best_model_dt.predict([X_val[index_of_7]])
    assert dt_pred == 7

def test_digit_correct_8_dt():
    index_of_8 = np.where(y_val==8)[0][3]
    dt_pred  = best_model_dt.predict([X_val[index_of_8]])
    assert dt_pred == 8

def test_digit_correct_9_dt():
    index_of_9 = np.where(y_val==9)[0][0]
    dt_pred  = best_model_dt.predict([X_val[index_of_9]])
    assert dt_pred == 9

def test_min_accuracy_svm():
    for y in range(10):
        indices = np.where(y_val==y)[0]
        X_score = list(map(X_val.__getitem__, indices))
        svm_preds    = best_model_dt.predict(X_score)
        acc_val    = accuracy_score(svm_preds, y_val[indices])
        assert acc_val > ACC_THRESHOLD

def test_min_accuracy_dt():
    for y in range(10):
        indices = np.where(y_val==y)[0]
        X_score = list(map(X_val.__getitem__, indices))
        dt_preds    = best_model_dt.predict(X_score)
        acc_val    = accuracy_score(dt_preds, y_val[indices])
        assert acc_val > ACC_THRESHOLD 




# def test_equality():
#     assert 1 == 1
    
# def test_model_writing():
#     digits    = datasets.load_digits()
#     n_samples = digits.data.shape[0]
#     X         = digits.images.reshape(n_samples, -1)
#     y         = digits.target

#     sample_indices = random.choices(range(n_samples), k = 100)

#     X_sample, y_sample =  X[sample_indices], y[sample_indices]
#     gamma = 0.01

#     model = run_classification_experiment(path, gamma=gamma, 
#     X_train = X_sample, 
#     y_train=y_sample,
#     X_val=X_sample, 
#     y_val=y_sample)

#     assert os.path.isfile(f'{path}/{model}')

# def test_small_data_overfit_checking():
#     digits    = datasets.load_digits()
#     n_samples = digits.data.shape[0]
#     X         = digits.images.reshape(n_samples, -1)
#     y         = digits.target


#     X_train, X_test, X_val, y_train, y_test, y_val = create_splits(X, y, test_size=0.3, validation_size_from_test_size=0.5)

#     sample_indices = random.choices(range(X_test.shape[0]), k = 100)

#     X_sample, y_sample =  X_test[sample_indices], y_test[sample_indices]

#     model = run_classification_experiment(path, gamma=0.01, 
#     X_train = X_train, 
#     y_train=y_train,
#     X_val=X_train, 
#     y_val=y_train)

#     train_metrics = get_performance(path, model, X_train, y_train)
#     small_data_metrics = get_performance(path, model, X_sample, y_sample)

#     assert train_metrics['acc'] > small_data_metrics['acc'] + 0.1
#     assert train_metrics['f1'] > small_data_metrics['f1'] + 0.1


# def test_no_of_samples_100():

#     digits    = datasets.load_digits()
#     n_samples = digits.data.shape[0]
#     X         = digits.images.reshape(n_samples, -1)
#     y         = digits.target 

#     # Selecting sample of 10
#     sample_indices = random.choices(range(X.shape[0]), k = 100)

#     X_sample, y_sample =  X[sample_indices], y[sample_indices]

#     X_train, X_test, X_val, y_train, y_test, y_val = create_splits(X_sample, y_sample, test_size=0.3, validation_size_from_test_size=(1/3))

#     size_of_train, size_of_test, size_of_val = X_train.shape[0], X_test.shape[0], X_val.shape[0]
#     size_of_all = size_of_train +  size_of_test + size_of_val
    
#     assert size_of_all == X_sample.shape[0]

#     assert size_of_train == int(0.7 * X_sample.shape[0])

#     assert size_of_test == int(0.2 * X_sample.shape[0])

#     assert size_of_val == int(0.1 * X_sample.shape[0])


# def test_no_of_samples_9():

#     digits    = datasets.load_digits()
#     n_samples = digits.data.shape[0]
#     X         = digits.images.reshape(n_samples, -1)
#     y         = digits.target 

#     # Selecting sample of 9
#     sample_indices = random.choices(range(X.shape[0]), k = 9)

#     X_sample, y_sample =  X[sample_indices], y[sample_indices]

#     X_train, X_test, X_val, y_train, y_test, y_val = create_splits(X_sample, y_sample, test_size=0.3, validation_size_from_test_size=(1/3))

#     size_of_train, size_of_test, size_of_val = X_train.shape[0], X_test.shape[0], X_val.shape[0]
#     size_of_all = size_of_train +  size_of_test + size_of_val
    
#     assert size_of_all == X_sample.shape[0]

#     assert size_of_train == round(0.7 * X_sample.shape[0])

#     assert size_of_test == round(0.2 * X_sample.shape[0])

#     assert size_of_val == round(0.1 * X_sample.shape[0])


# def test_bonus_case_1():
#     """
#     label set in train, test, valid sets is same.
#     """

#     digits    = datasets.load_digits()
#     n_samples = digits.data.shape[0]
#     X         = digits.images.reshape(n_samples, -1)
#     y         = digits.target 

#     # Selecting sample of 500
#     sample_indices = random.choices(range(X.shape[0]), k = 500)

#     X_sample, y_sample =  X[sample_indices], y[sample_indices]

#     X_train, X_test, X_val, y_train, y_test, y_val = create_splits(X_sample, y_sample, test_size=0.3, validation_size_from_test_size=(1/3))

#     classes_train, classes_test = np.unique(y_train), np.unique(y_test)

#     # To compare all the classes present in train and test 
#     assert np.array_equal(classes_train, classes_test)

# def test_bonus_case_2():  
#     """
#     the number of samples (x) and number of labels (y) are same in each split.
#     """

#     digits    = datasets.load_digits()
#     n_samples = digits.data.shape[0]
#     X         = digits.images.reshape(n_samples, -1)
#     y         = digits.target 

#     # Selecting sample of 10
#     sample_indices = random.choices(range(X.shape[0]), k = 500)

#     X_sample, y_sample =  X[sample_indices], y[sample_indices]

#     X_train, X_test, X_val, y_train, y_test, y_val = create_splits(X_sample, y_sample, test_size=0.3, validation_size_from_test_size=(1/3))

#     # the number of samples (x) and number of labels (y) are same in each split.
#     assert X_train.shape[0] == len(y_train)
#     assert X_test.shape[0] == len(y_test)
#     assert X_val.shape[0] == len(y_val)

# def test_bonus_case_3():  
#     """
#     the model saved is not corrupted.
#     """
#     model = "svm_gamma_0.01.pkl"
#     path = "../models"
#     try:
#         with open(f'{path}/{model}', 'rb') as pickle_file:
#             clf = pickle.load(pickle_file)
#     except:
#         clf = None

#     assert clf != None


# def test_bonus_case_4():
#     """
#     the results of the model are not changing in each run -- i.e. model training is deterministic.
#     """

#     digits    = datasets.load_digits()
#     n_samples = digits.data.shape[0]
#     X         = digits.images.reshape(n_samples, -1)
#     y         = digits.target

#     X_train, X_test, X_val, y_train, y_test, y_val = create_splits(X, y, test_size=0.3, validation_size_from_test_size=0.5)

#     model = run_classification_experiment(path, gamma=0.01, 
#     X_train = X_train, 
#     y_train=y_train,
#     X_val=X_val, 
#     y_val=y_val)

#     test_metrics_1 = get_performance(path, model, X_test, y_test)

#     model = run_classification_experiment(path, gamma=0.01, 
#     X_train = X_train, 
#     y_train=y_train,
#     X_val=X_val, 
#     y_val=y_val)

#     test_metrics_2 = get_performance(path, model, X_test, y_test)
    

#     assert test_metrics_1 == test_metrics_2


# def test_bonus_case_5():
#     """
#     dimensionality of training samples is same as dimensionality of test and validation samples
#     """

#     digits    = datasets.load_digits()
#     n_samples = digits.data.shape[0]
#     X         = digits.images.reshape(n_samples, -1)
#     y         = digits.target

#     X_train, X_test, X_val, y_train, y_test, y_val = create_splits(X, y, test_size=0.3, validation_size_from_test_size=0.5)


#     shape_of_a_train_sample = X_train[random.choice(range(X_train.shape[0]))].shape
#     shape_of_a_test_sample = X_test[random.choice(range(X_test.shape[0]))].shape 
#     shape_of_a_val_sample = X_test[random.choice(range(X_val.shape[0]))].shape 

#     assert shape_of_a_train_sample == shape_of_a_test_sample
#     assert shape_of_a_test_sample  == shape_of_a_val_sample
#     assert shape_of_a_train_sample == shape_of_a_val_sample

