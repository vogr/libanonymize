#pragma once

//#include <pybind11/eigen.h>
#include <Eigen/Dense>

// Row-major Eigen::MatrixXd and Eigen::MatrixXi
using RMatrixXd = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
using RMatrixXi = Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;


RMatrixXd read_hdf5_dataset(std::string const & fname, std::string const & dataset_name);