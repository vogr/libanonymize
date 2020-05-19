//
// Created by vogier on 30/04/2020.
//

#include "helpers.h"

#include <Eigen/Dense>
// Load after Eigen
#include <highfive/H5Easy.hpp>

RMatrixXd read_hdf5_dataset(std::string const & fname, std::string const & dataset_name) {
    // open the file for reading
		H5Easy::File hf(fname, H5Easy::File::ReadOnly);
    return H5Easy::load<RMatrixXd>(hf, dataset_name);
}

