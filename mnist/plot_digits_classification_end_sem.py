"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

Note: This is specially implmented for End Semester exam 
"""

# Import datasets
from utils.utils import preprocess, create_splits, run_classification_experiment, get_performance
from sklearn import datasets
import pandas as pd
import numpy as np
import sys

import itertools

digits    = datasets.load_digits()
n_samples = digits.data.shape[0]
data      = digits.images.reshape(n_samples, -1)

# selected_algorithm = sys.argv[1]

# Preprocessing the input data
preprocessed_data = preprocess(data, 1)

splits = {}

used_param, models, acc_vals, f1_vals  = [], [], [], []
path = "./models"

# Split data into 70% train and 30% test subsets; This has to happen 3 times 
X_train, X_test, X_val, y_train, y_test, y_val = create_splits(preprocessed_data, digits.target, test_size=0.3, validation_size_from_test_size=0.5, random_state = 40)
splits['split_1'] = [X_train, X_test, X_val, y_train, y_test, y_val]

X_train, X_test, X_val, y_train, y_test, y_val = create_splits(preprocessed_data, digits.target, test_size=0.3, validation_size_from_test_size=0.5, random_state = 41)
splits['split_2'] = [X_train, X_test, X_val, y_train, y_test, y_val]

X_train, X_test, X_val, y_train, y_test, y_val = create_splits(preprocessed_data, digits.target, test_size=0.3, validation_size_from_test_size=0.5, random_state = 42)
splits['split_3'] = [X_train, X_test, X_val, y_train, y_test, y_val]

param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}

param_sets = list(itertools.product(*[param_grid['C'], param_grid['gamma'], param_grid['kernel']]))

gammas  = []
Cs      = []
kernels = []

run1_train = []
run1_dev   = []
run1_test  = []

run2_train = []
run2_dev   = []
run2_test  = []

run3_train = []
run3_dev   = []
run3_test  = []

mean_train = []
mean_dev   = []
mean_test  = []


for param_set in param_sets:
    
    Cs.append(param_set[0])
    gammas.append(param_set[1])
    kernels.append(param_set[2])

    for i, split in enumerate(splits.keys()):
        
        X_train = splits[split][0]
        X_test  = splits[split][1]
        X_val   = splits[split][2]

        y_train = splits[split][3]
        y_test  = splits[split][4]
        y_val   = splits[split][5]

        model = run_classification_experiment(path, param_set, X_train, y_train, X_val, y_val, selected_algorithm = 'svm')

        metrics_train  = get_performance(path, model, X_train, y_train)
        metrics_val    = get_performance(path, model, X_val, y_val)
        metrics_test   = get_performance(path, model, X_test, y_test)
        if i == 0:
            run1_train.append(metrics_train['acc'])
            run1_test.append(metrics_test['acc'])
            run1_dev.append(metrics_val['acc'])

        if i == 1:
            run2_train.append(metrics_train['acc'])
            run2_test.append(metrics_test['acc'])
            run2_dev.append(metrics_val['acc'])

        if i == 2:
            run3_train.append(metrics_train['acc'])
            run3_test.append(metrics_test['acc'])
            run3_dev.append(metrics_val['acc'])

    mean_train.append(np.mean([run1_train[-1],run2_train[-1],run3_train[-1]]))
    mean_test.append(np.mean([run2_test[-1],run2_test[-1],run3_test[-1]]))
    mean_dev.append(np.mean([run3_dev[-1],run2_dev[-1],run3_dev[-1]]))

df_results = pd.DataFrame({
    'C': Cs,
    'Gamma':gammas, 
    'Kernel':kernels, 
    'RUN1_train': run1_train,
    'RUN1_dev': run1_dev,
    'RUN1_test': run1_test,
    'RUN2_train': run1_train,
    'RUN2_dev': run1_dev,
    'RUN2_test': run1_test,
    'RUN3_train': run1_train,
    'RUN3_dev': run1_dev,
    'RUN3_test': run1_test,
    'MEAN_train': mean_train,
    'MEAN_dev': mean_dev,
    'MEAN_test': mean_test,

})

df_results.to_csv('results.csv', index=False)

print(df_results)