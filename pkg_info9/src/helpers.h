#pragma once

//#include <pybind11/eigen.h>
#include <Eigen/Dense>

// Row-major Eigen::MatrixXd and Eigen::MatrixXi
using RMatrixXd = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
using RMatrixXi = Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;



int add(int i, int j);
Eigen::MatrixXd addm(const Eigen::Ref<Eigen::MatrixXd> & a, const Eigen::Ref<Eigen::MatrixXd> & b);
void add_in_first(Eigen::Ref<Eigen::MatrixXd> a, Eigen::Ref<Eigen::MatrixXd const> b);


Eigen::MatrixXd read_hdf5_dataset(std::string fname, std::string dataset_name);
