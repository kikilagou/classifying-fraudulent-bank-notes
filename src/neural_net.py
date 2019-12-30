# http://archive.ics.uci.edu/ml/datasets/banknote+authentication

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = ['variance_of_wavelet_transformed_image', 'skewness_of_wavelet_transformed_image', 'curtosis_of_wavelet_transformed_image', 'entropy_of_image']
NAMES = FEATURES + ['class']
CLASS = 'class'

TRAIN_TEST_RATIO = 0.8
WEIGHTS = np.random.rand(4,1)
BIAS = np.random.rand(1)
LR = 0.005  # learning rate
EPOCHS = 20000

def load_and_prep_data(data_file):
    data = pd.read_csv(data_file, names=NAMES)
    y = data[CLASS]
    data = data.drop('class', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=1-TRAIN_TEST_RATIO)

    print(X_test.head())
    feature_set_train = np.array(X_train.values.tolist())
    feature_set_test = np.array(X_test.values.tolist())
    labels_train = np.array([y_train.values.tolist()])
    labels_test = np.array([y_test.values.tolist()])

    labels_train = labels_train.T
    labels_test = labels_test.T

    return feature_set_train, feature_set_test, labels_train, labels_test


# activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))


# derivative of activation function
def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))


# train
def train(features, labels, weights, bias, lr):
    for epoch in range(EPOCHS):
        inputs = features

        # feedforward step1
        XW = np.dot(features, weights) + bias

        #feedforward step2
        z = sigmoid(XW)

        # backpropagation step 1
        error = z - labels

        print(error.sum())

        # backpropagation step 2
        partial_der_cost_pred = error
        partial_der_pred_z = sigmoid_derivative(z)

        z_delta = partial_der_cost_pred * partial_der_pred_z
        inputs = features.T
        weights -= lr * np.dot(inputs, z_delta)

        for num in z_delta:
            bias -= lr * num

if __name__ == '__main__':
    features_train, features_test, labels_train, labels_test = load_and_prep_data("../data/banknote.data")
    train(features_train, labels_train, WEIGHTS, BIAS, LR)

    # test it out
    correct = 0
    for i in range(len(features_test)):
        single_point = features_test[i]
        result = sigmoid(np.dot(single_point, WEIGHTS) + BIAS)

        if round(result[0]) == labels_test[i][0]:
            correct += 1

    accuracy = correct / len(features_test) * 100
    print("Accuracy: {}%".format(accuracy))

