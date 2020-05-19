
#include "KnnClassification.hpp"

#include <iostream>
#include <sstream>
#include <string>

#include <ANN/ANN.h>

 
KnnClassification::KnnClassification(int k, std::shared_ptr<Dataset> dataset)
: Classification{std::move(dataset)}, m_k{k}  {
  auto const n = m_dataset->getNbrSamples();
  auto const d = m_dataset->getDim();
  
  // APPROACH 1: no copy ! Make kd-tree point to vectors in the dataset.
  // Save the vector of pointers as a class member : it should outlive the 
  // kd-tree.
  /*
  datapoints.resize(n,nullptr);
  
  std::cerr << "Building datapoints array for kd-tree..." << std::endl;
  for(int i = 0; i < n ; i++) {
    Eigen::Ref<Eigen::VectorXd const> const & v = dataset->getInstance(i);
      datapoints[i] = v.data();
  }
  */
  
   // APPROACH 2: Copy data points to ANNpointArray
   datapoints = annAllocPts(n, d);
   //std::memcpy(datapoints, dataset.data(), n * d * sizeof(double))
  //std::cerr << "Building datapoints array for kd-tree..." << std::endl;
  for(int i = 0; i < n ; i++) {
    Eigen::Ref<Eigen::VectorXd const> const v = m_dataset->getInstance(i);
    //std::memcpy(datapoints[i], v.data(), d * sizeof(double));
    for(int j=0 ; j < d; j++) {
      datapoints[i][j] = v(j);
    }
  }
  
  //std::cerr << "Building kd-tree..." << std::endl;
  m_kdTree = std::make_unique<ANNkd_tree>(datapoints, n, d);
  //std::cerr << "k-NN classifier has been trained!" << std::endl;
}
KnnClassification::~KnnClassification() {
    annDeallocPts(datapoints);
}

int KnnClassification::getK() {
    return m_k;
}

std::string KnnClassification::print_kd_stats() {
  std::stringstream ss;
	ANNkdStats stats;
    m_kdTree->getStats(stats);
    ss << stats.dim << " : dimension of space (e.g. 1899 for mail_train)" << std::endl;
	ss << stats.n_pts << " : no. of points (e.g. 4000 for mail_train)" << std::endl;
	ss << stats.bkt_size << " : bucket size" << std::endl;
	ss << stats.n_lf << " : no. of leaves (including trivial)" << std::endl;
	ss << stats.n_tl << " : no. of trivial leaves (no points)" << std::endl;
	ss << stats.n_spl << " : no. of splitting nodes" << std::endl;
	ss << stats.n_shr << " : no. of shrinking nodes (for bd-trees)" << std::endl;
	ss << stats.depth << " : depth of tree" << std::endl;
	ss << stats.sum_ar << " : sum of leaf aspect ratios" << std::endl;
	ss << stats.avg_ar << " : average leaf aspect ratio" << std::endl;
  return ss.str();
}
