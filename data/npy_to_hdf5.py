#!/usr/bin/env python3

from pathlib import Path
import sys

import h5py
import numpy as np

# Create enumerated type for labels
labels = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
label_dict = {l: i for i, l in enumerate(labels)}
label_enum = h5py.enum_dtype(label_dict, basetype='i')

label_map = np.vectorize(label_dict.get)

DATASETS = ["testa", "testb", "train"]

print(f"Will convert {DATASETS} to hdf5") 
for ds in DATASETS:
    print(f"Processing {ds}")
    representation = f"representation.{ds}.npy"
    true_labels = f"true_labels.{ds}.npy"
    out = f"{ds}.hdf5"

    with h5py.File(out, "w") as f:
        f.create_dataset("representation", data=np.load(representation))
        f.create_dataset("true_labels", data=label_map(np.load(true_labels)), dtype=label_enum)
