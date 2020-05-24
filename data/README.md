## Converting the datasets in a format readable by both Python and C++

1. Copy the following files in this directory:
  - representation.testa.npy
  - representation.testb.npy
  - representation.train.npy
  - true_labels.testa.npy
  - true_labels.testb.npy
  - true_labels.train.npy

2. Make sure that the Python libraries `numpy` and `h5py` are available to your Python executable :

```
$ python3 -c "import numpy ; import h5py"
# No error
```

If you don't have them installed :
- either activate the `info9_projet` Conda environment
- or install them using pip : `python3 -m pip install --user numpy h5py`

3. Run the script provided in this directory:

```
$ python3 "./npy_to_hdf5.py"
```
