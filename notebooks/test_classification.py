#/usr/bin/env python3

"""
Some classification algorithms have been implemented in the C++ library;
using pybind11 this library provides Python bindings in the form of the
`info9` library.

In this Python script, this library is imported and the methods implemented
in TD6 "k-Nearest Neighbors (k-NN) Classification" may be called directly in
Python using bindings.
"""


from pathlib import Path

import numpy as np

import info9

# Path of the training and testing datasets
train_ds = Path("csv/mail_train.csv")
test_ds = Path("csv/mail_test.csv")

# Load the training dataset into memory
dataset = info9.Dataset(str(train_ds))
dataset.show(False)

# Train the classifier (i.e. build the kd-tree)
k = 3
label_col = 0
classifier = info9.KnnClassification(k, dataset, label_col)

print("kd-tree stats:")
classifier.print_kd_stats()

# Test the classifier on the testing dataset:
# Iterate over the lines in the file, predict the label
# and fill in the confusion matrix.
confusion = np.array([[0,0], [0,0]])

print()

with test_ds.open("r") as f:
    for i, line in enumerate(f):
        if (i % 50 == 0):
            print(f"Processing line {i}...")
        s = line.strip().split(",")
        y = int(s[0])
        #x = np.array(s[1:], dtype=np.float64)
        # Explicit conversion is not required: pybind will directly
        # convert the list to an Eigen::VectorXf
        x = s[1:]
        prediction = classifier.estimate(x, 0.5)
        confusion[y][prediction] += 1;

print()
print("Confusion matrix:")
print(confusion)


