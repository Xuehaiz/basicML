#!/usr/bin/python
#
# CIS 472/572 - Logistic Regression Template Code
#
# Author: Xuehai Zhou
# Date:   2/8/2020
#
# Please use this code as the template for your solution.
#
import argparse
import numpy as np

import re
from math import log
from math import exp
from math import sqrt


# Process arguments for LR
def handle_args():
    parser = argparse.ArgumentParser(description=
                                     'Fit logistic regression model and make predictions on test data.')

    parser.add_argument('--eta', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--l2', type=float, default=1., help='Strength of L2 regularizer')
    parser.add_argument('--maxiter', type=int, default=100, help='Maximum number of iterations')
    parser.add_argument('--model', help='File for saving model parameters')
    parser.add_argument('train', help='Training data file')
    parser.add_argument('test', help='Test data file')

    return parser.parse_args()


# Load data from a file
def read_data(filename):
    # Read names
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    f.close()

    # Read data
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)
    x = data[:, 0:-1]
    y = data[:, -1]
    return ((x, y), varnames)


# Train a logistic regression model using batch gradient descent
def train_lr(train_x, train_y, eta, l2_reg_weight, maxiter=100):
    numvars = len(train_x[0])
    w = np.array([0.0] * numvars)  # dtype=np.float128
    b = 0.0

    for i in range(maxiter):
        g_w = np.array([0.0] * numvars)
        g_b = 0
        for x, y in zip(train_x, train_y):
            a = np.dot(x.transpose(), w.transpose()) + b
            var = y / (1 + exp(y * a))
            g_b -= eta * var
            for j in range(numvars):
                g_w[j] -= eta * x[j] * var
        for k in range(numvars):
            g_w[k] += eta * (l2_reg_weight * w[k])
        b -= g_b
        for n in range(numvars):
            w[n] -= g_w[n]

        if sqrt(np.dot(g_w.transpose(), g_w.transpose()) + g_b * g_b) < 0.01:
            break

    return w, b


# Predict the probability of the positive label (y=+1) given the
# attributes, x.
def predict_lr(model, x):
    (w, b) = model
    prob = b
    for i in range(len(x)):
        prob += w[i] * x[i]

    return prob


# Load train and test data.  Learn model.  Report accuracy.
# (NOTE: You shouldn't need to change this.)
def main():
    args = handle_args()

    # Read in lists of examples.  Each example is a list of attribute values,
    # where the last element in the list is the class value.
    ((train_x, train_y), varnames) = read_data(args.train)
    ((test_x, test_y), varnames) = read_data(args.test)

    # Train model
    (w, b) = train_lr(train_x, train_y, args.eta, args.l2, maxiter=args.maxiter)

    # Write model file
    if args.model:
        f = open(args.model, "w+")
        f.write('%f\n' % b)
        for i in range(len(w)):
            f.write('%s %f\n' % (varnames[i], w[i]))

    # Make predictions, compute accuracy
    correct = 0
    for (x, y) in zip(test_x, test_y):
        prob = predict_lr((w, b), x)
        # print(prob)
        if (prob - 0.5) * y > 0:
            correct += 1
    acc = float(correct) / len(test_y)
    print("Accuracy: ", acc)


if __name__ == "__main__":
    main()
