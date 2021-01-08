#!/usr/bin/python
#
# CIS 472/572 -- Programming Homework #1
#
# Starter code provided by Daniel Lowd, 1/25/2018
#
#
from __future__ import division
import sys
import re
import math

# Node class for the decision tree
import node

train = None
varnames = None
test = None
testvarnames = None
root = None


# Helper function computes entropy of Bernoulli distribution with
# parameter p
def entropy(p):
    # >>>> YOUR CODE GOES HERE <<<<
    if p == 1 or p == 0:
        ans = 1
    else:
        q = 1 - p
        ans = -(p * math.log(p, 2) + q * math.log(q, 2))
    return ans


# Compute information gain for a particular split, given the counts
# py_pxi : number of occurences of y=1 with x_i=1 for all i=1 to n
# pxi : number of occurrences of x_i=1
# py : number of ocurrences of y=1
# total : total length of the data
def infogain(py_pxi, pxi, py, total):
    # >>>> YOUR CODE GOES HERE <<<<
    if total == 0:
        entropy_py = 0
    else:
        entropy_py = entropy(py / total)
    if pxi == 0:
        entropy_px1 = 0
    else:
        entropy_px1 = entropy(py_pxi / pxi)
    if (total - pxi) == 0:
        entropy_px0 = 0
    else:
        entropy_px0 = entropy((py-py_pxi)/(total - pxi))
    return entropy_py - (pxi/total) * entropy_px1 - ((total-pxi)/total) * entropy_px0


# OTHER SUGGESTED HELPER FUNCTIONS:
# - collect counts for each variable value with each class label
def count_collect(data, varnames):
    total = len(data)
    var_len = len(varnames) - 1
    # find py  (checked)
    counter = 0
    for i in range(total):
        if data[i][var_len] == 1:
            counter += 1
    py = counter

    # find pxi and pxi_py (checked)
    pxi_list = [0] * var_len
    py_pxi_list = [0] * var_len
    for i in range(var_len):
        counter = 0
        count = 0
        for j in range(total):
            if data[j][i] == 1:
                counter += 1
            if data[j][var_len] == 1 and data[j][i] == 1:
                count += 1
        pxi_list[i] = counter
        py_pxi_list[i] = count
    return py, pxi_list, py_pxi_list


# - find the best variable to split on, according to mutual information
def split_on_variable(var_len, gain_list):
    max_gain = 0
    max_gain_pos = 0
    for i in range(var_len):
        if gain_list[i] > max_gain:
            max_gain = gain_list[i]
            max_gain_pos = i
    return max_gain_pos, ig


# - partition data based on a given variable
def partition(data, max_gain_pos):
    total = len(data)
    # sub-data for building left right sub-tree
    data_l = []
    data_r = []
    # partition the data
    for i in range(total):
        if data[i][max_gain_pos] == 0:
            data[i].pop(max_gain_pos)
            data_l.append(data[i])
        else:
            data[i].pop(max_gain_pos)
            data_r.append(data[i])
    return data_l, data_r


# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    for l in f:
        data.append([int(x) for x in p.split(l.strip())])
    return (data, varnames)


# Saves the model to a file.  Most of the work here is done in the
# node class.  This should work as-is with no changes needed.
def print_model(root, modelfile):
    f = open(modelfile, 'w+')
    root.write(f, 0)


def build_tree_helper(used, data, varnames, var_len, curr_ent):
    if curr_ent == 0:
        return node.Leaf(varnames, data[0][l])

    feat, ig = split_on_variable(used, data)

    zeros, ones = partition(feat, data)

    if len(used) == var_len or ig == 0:
        aff = count_class(var_len, data)
        neg = len(data) - aff
        if aff > neg:
            return node.Leaf(varnames, 1)
        else:
            return node.Leaf(varnames, 0)

    if ig == curr_ent:
        if len(zeros) == 0:
            return node.Leaf(varnames, ones[0][var_len])
        elif len(ones) == 0:
            return node.Leaf(varnames, zeros[0][var_len])
        else:
            if data[0][feat] == data[0][var_len]:
                return node.Split()

# Build tree in a top-down manner, selecting splits until we hit a
# pure leaf or all splits look bad.
def build_tree(data, varnames):
    # >>>> YOUR CODE GOES HERE <<<<
    total = len(data)
    var_len = len(varnames) - 1
    # base case
    if len(data) == 0:
        return
    if len(data) == 1:
        return node.Leaf(varnames, data[0][var_len])
    if var_len == 0:
        return node.Leaf(varnames, )

    # do the counting to calculate info gain
    py, pxi_list, py_pxi_list = count_collect(data, varnames)

    # calculate all info gain (checked)
    gain_list = [0] * var_len
    for i in range(var_len):
        gain_list[i] = infogain(py_pxi_list[i], pxi_list[i], py, total)
    print("gain list:", gain_list)
    # find max info gain
    max_gain_pos = split_on_variable(var_len, gain_list)
    print("max_gain_pos:", max_gain_pos)

    # partition
    data_l, data_r = partition(data, max_gain_pos)
    print(data_l)

    varnames.pop(max_gain_pos)

    rt = node.Split(varnames, max_gain_pos,
                    build_tree(data_l, varnames), build_tree(data_r, varnames))
    # recursive call
    return rt


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
    if len(argv) != 3:
        print('Usage: id3.py <train> <test> <model>')
        sys.exit(2)
    loadAndTrain(argv[0], argv[1], argv[2])

    acc = runTest()
    print("Accuracy: ", acc)

    # n = ['foo', 'bar', 'baz']
    # root = node.Split(n, 0, node.Split(n, 1, node.Leaf(n, 0), node.Leaf(n, 1)), node.Leaf(n, 0))
    # root.write(sys.stdout, 0)
    #
    # print(root.classify([0, 0, 0]))
    # print(root.classify([0, 0, 1]))
    # print(root.classify([0, 1, 0]))
    # print(root.classify([0, 1, 1]))
    # print(root.classify([1, 0, 0]))
    # print(root.classify([1, 0, 1]))
    # print(root.classify([1, 1, 0]))
    # print(root.classify([1, 1, 1]))

if __name__ == "__main__":
    main(sys.argv[1:])
