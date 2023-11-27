# -------------------------------------------------------------------------
# AUTHOR: Alexander Eckert
# FILENAME: perceptron.py
# SPECIFICATION: creating single and multi layer perceptrons
# FOR: CS 4210- Assignment #4
# TIME SPENT: 2 hours
# -----------------------------------------------------------*/
# IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to
# complete this code.
# importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier  # pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]
df = pd.read_csv('optdigits.tra', sep=',', header=None)  # reading the data by using
# Pandas library
X_training = np.array(df.values)[:, :64]  # getting the first 64 fields to form the
# feature data for training
y_training = np.array(df.values)[:, -1]  # getting the last field to form the class
# label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None)  # reading the data by using
# Pandas library
X_test = np.array(df.values)[:, :64]  # getting the first 64 fields to form the
# feature data for test
y_test = np.array(df.values)[:, -1]  # getting the last field to form the class
# label for test

highest_perceptron_accuracy = 0.0
highest_perceptron_parameters = None

highest_mlp_accuracy = 0.0
highest_mlp_parameters = None

for eta in n:  # iterates over n
    for shuffle_flag in r:  # iterates over r
        # iterates over both algorithms
        for algorithm in ['Perceptron', 'MLP']:  # iterates over the algorithms
            # Create a Neural Network classifier
            if algorithm == 'Perceptron':
                clf = Perceptron(eta0=eta, shuffle=shuffle_flag, max_iter=1000)
            else:
                clf = MLPClassifier(
                    activation='logistic',
                    learning_rate_init=eta,
                    hidden_layer_sizes=(25,),
                    shuffle=shuffle_flag,
                    max_iter=1000,
                )

            # Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            # make the classifier prediction for each test sample and start
            # computing its accuracy
            correct_predictions = 0
            total_samples = len(X_test)

            for x_testSample, y_testSample in zip(X_test, y_test):
                prediction = clf.predict([x_testSample])
                if prediction == y_testSample:
                    correct_predictions += 1

            accuracy = correct_predictions / total_samples

            # check if the calculated accuracy is higher than the previously one
            # calculated for each classifier. If so, update the highest accuracy
            # and print it together with the network hyperparameters
            if algorithm == 'Perceptron' and accuracy > highest_perceptron_accuracy:
                highest_perceptron_accuracy = accuracy
                highest_perceptron_parameters = f"learning rate={eta}, shuffle={shuffle_flag}"

            elif algorithm == 'MLP' and accuracy > highest_mlp_accuracy:
                highest_mlp_accuracy = accuracy
                highest_mlp_parameters = f"learning rate={eta}, shuffle={shuffle_flag}"

print(f"Highest Perceptron accuracy so far: {highest_perceptron_accuracy:.2f}, Parameters: {highest_perceptron_parameters}")
print(f"Highest MLP accuracy so far: {highest_mlp_accuracy:.2f}, Parameters: {highest_mlp_parameters}")
