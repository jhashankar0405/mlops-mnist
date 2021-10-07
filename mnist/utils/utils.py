from skimage.transform import rescale

from sklearn import datasets, svm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
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


def create_splits(data, digits, test_size, validation_size_from_test_size):
    """
    Splits the input data into train, test & validation set
    """

    # Split data into 70% train and 15% validation and 15% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=test_size, shuffle=False)

    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=validation_size_from_test_size, shuffle=False)
    
    return X_train, X_test, X_val, y_train, y_test, y_val


def run_classification_experiment(path, gamma, X_train, y_train, X_val, y_val):
    """
    Trains an SVM and saves the model 
    Returns: Saved model name
    """
    clf = svm.SVC(gamma=gamma)
    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

	# Predict the value of the digit on the test subset
    val_accuracy = accuracy_score(y_val, clf.predict(X_val))
    if val_accuracy < 0.11:
        return None
    else:
        location = f'{path}/svm_gamma_{gamma}.pkl'
        #Save the model to disk
        with open(location, 'wb') as handle:
            pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return f'svm_gamma_{gamma}.pkl'
    
def get_performance(path, model_name, X_test, y_test):
    with open(f'{path}/{model_name}', 'rb') as pickle_file:
        clf = pickle.load(pickle_file)

    predicted_test = clf.predict(X_test)

    result = {
        "acc": accuracy_score(y_test, predicted_test), 
        "f1": f1_score(y_test, predicted_test, average='weighted')
    }

    return result