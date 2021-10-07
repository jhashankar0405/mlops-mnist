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

digits    = datasets.load_digits()
n_samples = digits.data.shape[0]
data      = digits.images.reshape(n_samples, -1)

gammas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]

# Preprocessing the input data
preprocessed_data = preprocess(data, 1)

# Split data into 70% train and 30% test subsets
X_train, X_test, X_val, y_train, y_test, y_val = create_splits(preprocessed_data, digits, test_size=0.3, validation_size_from_test_size=0.5)
used_gammas, models, acc_vals, f1_vals  = [], [], [], []
path = "./models"

for gamma in gammas:
	model = run_classification_experiment(path, gamma, X_train, y_train, X_val, y_val)
	if model is not None:
		metrics = get_performance(path, model, X_val, y_val)
		used_gammas.append(gamma)
		models.append(model)
		acc_vals.append(metrics['acc'])
		f1_vals.append(metrics['f1'])

df_results = pd.DataFrame({
	'gamma':used_gammas, 
	'model_name':  models, 
	'val_accuracy': acc_vals,
	'val_f1': f1_vals
	})

print(df_results)
print()

print(f"Best gamma: {df_results.loc[df_results['val_accuracy'].idxmax(), 'gamma']}")

best_model = df_results.loc[df_results['val_accuracy'].idxmax(), 'model_name']

test_metrics = get_performance(path, best_model, X_test, y_test)

print(f"Accuracy of best classfier on test data is {test_metrics['acc']}")
print(f"F1-score of best classfier on test data is {test_metrics['f1']}")
