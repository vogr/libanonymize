//
// Created by vogier on 30/04/2020.
//

#include "helpers.h"

int add(int i, int j) {
  return i + j;
}

RMatrixXd addm(Eigen::Ref<RMatrixXd const> const & a, Eigen::Ref<RMatrixXd const> const & b) {
  return a + b;
}

void add_in_first(Eigen::Ref<Eigen::MatrixXd> a, Eigen::Ref<Eigen::MatrixXd const> const & b) {
  a += b;
}

/*
RMatrixXd read_hdf5_dataset(std::string fname, std::string dataset_name) {
    // open the file for reading
    HDF5::File hf = HDF5::File(fname, HDF5::File::ReadOnly);

    Eigen::MatrixXd matrix;
    hf.read(dataset_name, matrix);
    
    return matrix;
}
*/
