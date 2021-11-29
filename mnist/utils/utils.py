from skimage.transform import rescale

from sklearn import datasets, svm
from sklearn import tree

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, roc_auc_score
import os
import pickle

def preprocess(images, rescale_factor):
    """
    Rescales the images by specified factor
    """
    resized_images = []
    for d in images:
        resized_images.append(rescale(d, rescale_factor, anti_aliasing = False))
    return resized_images


def create_splits(data, target, test_size, validation_size_from_test_size):
    """
    Splits the input data into train, test & validation set
    """

    # Split data into 70% train and 15% validation and 15% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=test_size, shuffle=False, random_state = 42)

    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=validation_size_from_test_size, shuffle=False, random_state = 42)
    
    return X_train, X_test, X_val, y_train, y_test, y_val


def run_classification_experiment(path, parameter, X_train, y_train, X_val, y_val, selected_algorithm = 'svm'):
    """
    Trains an SVM and saves the model 
    Returns: Saved model name
    """
    if selected_algorithm == 'svm':
        clf = svm.SVC(gamma=parameter, probability = True)
        name = f'svm_gamma_{parameter}.pkl'
    else:
        clf = tree.DecisionTreeClassifier()
        name = f'dtree_maxd_{parameter}.pkl'
    # Learn the digits on the train subset
    clf.fit(X_train, y_train)
    

	# Predict the value of the digit on the test subset
    val_accuracy = accuracy_score(y_val, clf.predict(X_val))
    if val_accuracy < 0.11:
        return None
    else:
        
        location = f'{path}/{name}'
        #Save the model to disk
        with open(location, 'wb') as handle:
            pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return name
    
def get_performance(path, model_name, X_test, y_test):
    with open(f'{path}/{model_name}', 'rb') as pickle_file:
        clf = pickle.load(pickle_file)

    predicted_test = clf.predict(X_test)
    predicted_test_proba = clf.predict_proba(X_test)

    result = {
        "acc": accuracy_score(y_test, predicted_test), 
        "f1": f1_score(y_test, predicted_test, average='macro'),
        'roc': roc_auc_score(y_test, predicted_test_proba, multi_class="ovr")
    }

    return result

def load(model_path):
     return pickle.load(open(model_path, 'rb'))