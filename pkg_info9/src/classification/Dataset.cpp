
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include "Dataset.hpp"




/*
 * Our Eigen matrix should be stored row-major, as we will
 * use rows as vectors !
 * We load the matrix into memory as row-major ; is this sufficient ?
 */
 

void Dataset::Show(bool verbose) const {
	int n = getNbrSamples();
	int d = getDim();
	std::cout<<"Dataset with " << n <<" samples, and "<< d <<" dimensions."<<std::endl;
	if (verbose) {
		for (int i=0; i< n; i++) {
			for (int j=0; j < d; j++) {
				std::cout << m_instances(i,j)<<" ";
			}
			std::cout<<std::endl;
		}
	}
}

Dataset::Dataset(const std::string & dataset_fname) {
	std::cerr << "Reading datasets \"representation\" and dataset \"true_labels\" from file " << dataset_fname << " into memory..." << std::endl;
	
	H5Easy::File hf(dataset_fname, H5Easy::File::ReadOnly);
	std::cout << "Will load datapoints" << std::endl;
	m_instances = H5Easy::load<RMatrixXd>(hf, "representation");
	m_labels = H5Easy::load<Eigen::VectorXi>(hf, "true_labels");
	
	assert(m_instances.rows() == m_labels.rows());
	
	std::cerr << "Read of " << m_instances.rows() << " rows done!" << std::endl;
	
	max_label = m_labels.maxCoeff();
}
