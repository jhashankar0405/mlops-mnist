from skimage.transform import rescale

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def preprocess(images, rescale_factor):

    resized_images = []
    for d in images:
        resized_images.append(rescale(d, rescale_factor, anti_aliasing = False))
    return resized_images


def create_splits(data, digits, test_size, validation_size_from_test_size):

    # Split data into 70% train and 15% validation and 15% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=test_size, shuffle=False)

    X_test, X_validation, y_test, y_validation = train_test_split(
        X_test, y_test, test_size=validation_size_from_test_size, shuffle=False)
    
    return X_train, X_test, X_validation, y_train, y_test, y_validation


def test(clf, X, y):
    predicted_test = clf.predict(X_test)

    print()
    print("Best Model's accuracy on Test Data: ", accuracy_score(y_test, predicted_test))
    print("Best Model's f1-score on Test Data: ", f1_score(y_test, predicted_test, average='weighted'))
    print()