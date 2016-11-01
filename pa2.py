#!usr/bin/python
#program to implement bagging and boosting

import DecisionTreeCode     #importing the decision tree code structure from PA1 file

import numpy as np    #importing numpy and other packages
import os, sys


train_data = []
test_data = []

def load_data(datapath):
    train_file = open(datapath+"/agaricuslepiotatrain1", "r")
    test_file = open(datapath+"/agaricuslepiotatest1","r")
    train_lines = train_file.readlines()
    for line in train_lines:
        train_lines = line.strip()
        train_lines = line.split(" ")
        for l in range(len(train_lines)):
            train_lines[l] = int(train_lines[l])
            train_data.append(train_lines)
    train_file.close()
    test_lines = test_file.readlines()
    for line2 in test_lines:
        test_lines = line.strip()
        test_lines = line.split(" ")
        for k in range(len(test_lines)):
            test_lines[1] = int(test_lines)
            test_data.append(test_lines)
    test_file.close()

def learn_bagged(tdepth, numbags, datapath):
    pass;


def learn_boosted(tdepth, numtrees, datapath):
    pass;



if __name__ == "__main__":
    # The arguments to your file will be of the following form:
    # <ensemble_type> <tree_depth> <num_bags/trees> <data_set_path>
    # Ex. bag 3 10 mushrooms
    # Ex. boost 1 10 mushrooms

    # Get the ensemble type
    entype = sys.argv[1];
    # Get the depth of the trees
    tdepth = int(sys.arg[2]);
    # Get the number of bags or trees
    nummodels = int(sys.argv[3]);
    # Get the location of the data set
    datapath = sys.argv[4];

    # Check which type of ensemble is to be learned
    if entype == "bag":
        # Learned the bagged decision tree ensemble
        learn_bagged(tdepth, nummodels, datapath);
    else:
        # Learned the boosted decision tree ensemble
        learn_boosted(tdepth, nummodels, datapath);
