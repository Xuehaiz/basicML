#!/usr/bin/python
#
# CIS 472/572 -- Programming Homework #1
#
# Starter code provided by Daniel Lowd, 1/25/2018
#
# Author: Xuehai Zhou
# Co-author: Zeke Petersen
import sys
import re
# Node class for the decision tree
import node
import math

train = None
varnames = None
test = None
testvarnames = None
root = None


# - partition data based on a given feature
# p: an integer index into varnames
# data: subset of the training data
def partition(p, data):
    # assert p < len(data)
    list1 = []  # 0s
    list2 = []  # 1s
    # iterate across the data points (may be a subset of the original train data)
    for i in range(len(data)):
        if data[i][p]:  # Found a 1
            list2.append(data[i])
        else:  # Found a 0
            list1.append(data[i])
    # Return two lists? One of affirmatives on the variable and one of zeroes?
    return list1, list2


# - find the best variable to split on, according to mutual information
# used: set containing the indices of the features that have already appeared in the current subtree
# data: subset of the training data
def split_on_variable(used, data):
    curr_ig = 0
    index = 0
    # Skip splitting on the class
    for i in range(len(varnames) - 1):
        if i not in used:
            ig = my_infogain(i, data)
            if ig > curr_ig:
                curr_ig = ig
                index = i
    return index, curr_ig


# Count number of times a 1 is observed in the column p of the given data
def count_class(p, data):
    total = 0  # 1s
    # iterate across the data points (may be a subset of the original train data)
    for i in range(len(data)):
        if data[i][p]:  # Found a 1
            total += 1
    return total


# Wrapper for entropy
# p should be the index of the Class feature
def my_entropy(p, data):
    length = len(data)
    if length == 0:
        return 0
    lx = count_class(p, data)
    zero_prop = lx / length
    return entropy(zero_prop)


# Helper function computes entropy of Bernoulli distribution with parameter p (index)
# p here is the proportion

# Correct according to the autograder
def entropy(p):
    if p == 0 or p == 1:
        return 0
    else:
        one_prop = 1 - p
        return (-p * math.log(p, 2)) - (one_prop * math.log(one_prop, 2))


# Compute information gain for a particular split, given the counts (y represents Class, x is the feature)
# py_pxi : number of occurences of y=1 with x_i=1 for all i=1 to n
# pxi : number of occurrences of x_i=1
# py : number of ocurrences of y=1
# total : total length of the data
def infogain(py_pxi, pxi, py, total):
    # >>>> YOUR CODE GOES HERE <<<<
    curr_ent = entropy(py / float(total))
    if pxi == 0:
        ig = curr_ent - entropy((py - py_pxi) / float(total))
    elif total == pxi:
        ig = curr_ent - entropy(py_pxi / float(pxi))
    else:
        ig = curr_ent - (pxi / float(total)) * entropy(py_pxi / float(pxi)) \
             - (1 - pxi / float(total)) * entropy((py - py_pxi) / float(total - pxi))
    return ig


# Wrapper for infogain
def my_infogain(p, data):
    # Split data in two and check average entropy

    last_ind = len(varnames) - 1
    x, y = partition(p, data)

    total = len(data)
    py = count_class(last_ind, data)
    py_pxi = count_class(last_ind, y)
    pxi = len(y)

    return infogain(py_pxi, pxi, py, total)


# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    # print(header)
    varnames = p.split(header)
    # print(varnames)
    # namehash = {}
    for l in f:
        data.append([int(x) for x in p.split(l.strip())])
    return (data, varnames)


# Saves the model to a file.  Most of the work here is done in the
# node class.  This should work as-is with no changes needed.
def print_model(root, modelfile):
    f = open(modelfile, 'w+')
    root.write(f, 0)


# used: set containing the indices of the features that have already appeared in the current subtree
# data: subset of the training data to be further split in the current subtree
# varnames: list of feature names including Class
# l: total number of non-Class feature names
def build_tree_helper(used, data, varnames, l, curr_ent, max_class):
    # Everything is the same, return the Class of the first data point as a Leaf
    if curr_ent == 0:
        return node.Leaf(varnames, data[0][l])

    # We can assume this is non-zero since handled by wrapper function
    feat, ig = split_on_variable(used, data)
    zeroes, ones = partition(feat, data)

    # There are no more valid features, - we are done (all splits were bad) - return most common
    if len(used) == l or ig == 0:
        return node.Leaf(varnames, max_class)

    # This split fully explains the Classes of the data
    if ig == curr_ent:
        if data[0][feat] == data[0][l]:
            return node.Split(varnames, feat, node.Leaf(varnames, 0), node.Leaf(varnames, 1))
        else:
            return node.Split(varnames, feat, node.Leaf(varnames, 1), node.Leaf(varnames, 0))

    z_ent = my_entropy(l, zeroes)
    o_ent = my_entropy(l, ones)

    # True copy of python list
    u = used[:]
    u.append(feat)

    # Recursive case
    return node.Split(varnames, feat, build_tree_helper(u, zeroes, varnames, l, z_ent, max_class),
                      build_tree_helper(u, ones, varnames, l, o_ent, max_class))


# Build tree in a top-down manner, selecting splits until we hit a
# pure leaf or all splits look bad.
def build_tree(data, varnames):
    # Return the d-tree root eventually
    # >>>> YOUR CODE GOES HERE <<<<
    # For now, always return a leaf predicting "1":

    l = len(varnames) - 1

    # Nothing given to train on
    if len(data) == 0:
        return

    curr_ent = my_entropy(l, data)

    max_class = count_class(l, data)
    if (len(data) - max_class) > max_class:
        max_class = 0
    else:
        max_class = 1

    used = []

    return build_tree_helper(used, data, varnames, l, curr_ent, max_class)


# "varnames" is a list of names, one for each variable
# "train" and "test" are lists of examples.
# Each example is a list of attribute values, where the last element in
# the list is the class value.
def loadAndTrain(trainS, testS, modelS):
    global train
    global varnames
    global test
    global testvarnames
    global root
    (train, varnames) = read_data(trainS)
    (test, testvarnames) = read_data(testS)
    modelfile = modelS

    # build_tree is the main function you'll have to implement, along with
    # any helper functions needed.  It should return the root node of the
    # decision tree.
    root = build_tree(train, varnames)
    print_model(root, modelfile)


def runTest():
    correct = 0
    # The position of the class label is the last element in the list.
    yi = len(test[0]) - 1
    for x in test:
        # Classification is done recursively by the node class.
        # This should work as-is.
        pred = root.classify(x)
        if pred == x[yi]:
            correct += 1
    acc = float(correct) / len(test)
    return acc


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 3):
        print('Usage: id3.py <train> <test> <model>')
        sys.exit(2)
    loadAndTrain(argv[0], argv[1], argv[2])

    acc = runTest()
    print("Accuracy: ", acc)


if __name__ == "__main__":
    main(sys.argv[1:])
