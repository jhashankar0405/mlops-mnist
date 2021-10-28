"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.
"""

# Import datasets
from utils.utils import preprocess, create_splits, run_classification_experiment, get_performance
from sklearn import datasets
import pandas as pd
import sys

digits    = datasets.load_digits()
n_samples = digits.data.shape[0]
data      = digits.images.reshape(n_samples, -1)

selected_algorithm = sys.argv[1]

gammas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
max_ds = [2, 4, 6, 8, 10]

# Preprocessing the input data
preprocessed_data = preprocess(data, 1)

# Split data into 70% train and 30% test subsets
X_train, X_test, X_val, y_train, y_test, y_val = create_splits(preprocessed_data, digits.target, test_size=0.3, validation_size_from_test_size=0.5)
used_param, models, acc_vals, f1_vals  = [], [], [], []
path = "./models"

if selected_algorithm == 'svm': 
	for gamma in gammas:
		model = run_classification_experiment(path, gamma, X_train, y_train, X_val, y_val, selected_algorithm = selected_algorithm)
		if model is not None:
			metrics = get_performance(path, model, X_val, y_val)
			used_param.append(gamma)
			models.append(model)
			acc_vals.append(metrics['acc'])
			f1_vals.append(metrics['f1'])

	df_results = pd.DataFrame({
		'param':used_param, 
		'model_name':  models, 
		'val_accuracy': acc_vals,
		'val_f1': f1_vals
		})

else:
	for max_d in max_ds:
		model = run_classification_experiment(path, max_d, X_train, y_train, X_val, y_val, selected_algorithm = selected_algorithm)
		if model is not None:
			metrics = get_performance(path, model, X_val, y_val)
			used_param.append(max_d)
			models.append(model)
			acc_vals.append(metrics['acc'])
			f1_vals.append(metrics['f1'])
	
	df_results = pd.DataFrame({
		'param':used_param, 
		'model_name':  models, 
		'val_accuracy': acc_vals,
		'val_f1': f1_vals
		})


print(df_results)
print()

print(f"Best param: {df_results.loc[df_results['val_accuracy'].idxmax(), 'param']}")

best_model = df_results.loc[df_results['val_accuracy'].idxmax(), 'model_name']

test_metrics = get_performance(path, best_model, X_test, y_test)

print(f"Accuracy of best classfier on test data is {test_metrics['acc']}")
print(f"F1-score of best classfier on test data is {test_metrics['f1']}")
