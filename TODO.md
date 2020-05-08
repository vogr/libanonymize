# Todo

- find a way to cleanly include Eigen3 (installed through conda ?)
- choose best way to pass data : read once, pass-by-copy ok ? Or write to hdf5 and read here.
- Do not keep same table in memory twice
  + always write to a file : python rw to .npy for numpy arrays, c++ rw to .npy to Eigen3 MatrixXd with cnnpy.
  + use buffer protocol (pybind11) : python passes a struct containing dims, size, stride, and a pointer.
  + use pybind special treatment of Eigen
  + use pyarrow table and unwrap to arrow table in C++ (<- does it copy or not ?)

- compare our impl against TD6 and time the important fctions (loading data, projection, projection quality, estimation) + compare confusion matrix.
