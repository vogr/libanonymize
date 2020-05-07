#pragma once

//#include <pybind11/eigen.h>
#include <Eigen/Dense>

// Row-major Eigen::MatrixXd and Eigen::MatrixXi
using RMatrixXd = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
using RMatrixXi = Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;



int add(int i, int j);
RMatrixXd addm(Eigen::Ref<RMatrixXd const> const & a, Eigen::Ref<RMatrixXd const> const & b);
void add_in_first(Eigen::Ref<Eigen::MatrixXd> a, Eigen::Ref<Eigen::MatrixXd const> const & b);


RMatrixXd read_hdf5_dataset(std::string fname, std::string dataset_name);
