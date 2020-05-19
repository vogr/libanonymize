#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np

import h5py



def main(args):
    for infile in map(Path, args):
        outfile = infile.with_suffix(".hdf5")
        with h5py.File(outfile, "w") as f:
            data = np.loadtxt(infile, delimiter=",", dtype=bool)
            labels = data[:,0]
            representation = data[:,1:]
            f.create_dataset("representation", data=representation, dtype=bool)
            f.create_dataset("true_labels", data=labels, dtype=bool)

if __name__ == "__main__":
    main(sys.argv[1:])

    
