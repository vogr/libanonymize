
#include "KnnClassification.hpp"
#include <iostream>
#include <ANN/ANN.h>

 
KnnClassification::KnnClassification(int k, std::shared_ptr<Dataset> dataset)
: Classification{dataset}, m_k{k}  {
  auto const n = dataset->getNbrSamples();
  auto const d = dataset->getDim();
  
  // APPROACH 1: no copy ! Make kd-tree point to vectors in the dataset.
  // Save the vector of pointers as a class member : it should outlive the 
  // kd-tree.
  
  datapoints.resize(n,nullptr);
  
  std::cerr << "Building datapoints array for kd-tree..." << std::endl;
  for(int i = 0; i < n ; i++) {
    Eigen::Ref<Eigen::VectorXd> v = dataset->getInstance(i);
      datapoints[i] = v.data();
  }
  
  /*
   // APPROACH 2: Copy data points to ANNpointArray
   datapoints = annAllocPts(n, d);

  std::cerr << "Building datapoints array for kd-tree..." << std::endl;
  for(size_t i = 0; i < n ; i++) {
    Eigen::Ref<Eigen::VectorXd> v = dataset->getInstance(i);
    for(size_t j=0 ; j < d; j++) {
      datapoints[i][j] = v(j);
    }
  }
  */
  
  std::cerr << "Building kd-tree..." << std::endl;
  m_kdTree = std::make_unique<ANNkd_tree>(datapoints.data(), n, d);
  std::cerr << "k-NN classifier has been trained!" << std::endl;
}


int KnnClassification::Estimate(Eigen::Ref<Eigen::VectorXd> x, double threshold) {
  //std::cout << "Input vector size:" << x.size() << std::endl;
  
  // index of nearest neighbors returned by the search
  std::vector<int> nn_idx(m_k, 0);
  std::vector<double> dists(m_k, 0.);


	// NOTE : we directly pass a vector to the data underlying
	// the Eigen::Ref<Eigen::VectorXd> to ANN. A bit hacky, and
	// does not accept const qualifier... but saves a copy.

  //std::cout << "Will search in kd-tree..." << std::endl;
  m_kdTree->annkSearch(
      x.data(),      // query point
      m_k,    // number of near neighbors to find
      nn_idx.data(), // MODIFIED : id of nearest neighbors
      dists.data() // MODIFIED : array of dists to these neighbors
  );
  //std::cout << "Done!" << std::endl;
  
  /*
  std::cout << "nn_idx: ";
  for(auto f : nn_idx) {
	std::cout << f << " ";
  }
  std::cout << std::endl;

  
  std::cout << "labels: ";
  for(auto i : nn_idx) {
	std::cout << label << " ";
  }
  std::cout << std::endl;
  */
  
  std::vector<double> p_label(m_dataset->getMaxLabel() + 1, 0.);
  for(auto i : nn_idx) {
    auto const label = m_dataset->getLabel(i);
    p_label[label] += 1. / m_k;
  }

  /*
   * CONVENTION :
   * Le label 0 correspond au cas sans étiquette !
   * On cherche le label (donc non 0) qui a le plus haut score.
   * Si le threshold n'est pas atteint, on renvoie 0.
   */
   /*
    * OTHER POSSIBLE CHOICE:
    * Return best label found, even if 0
    */
   
  size_t lmax = 1;
  double pmax = p_label[1];
  for (size_t i = 1; i < p_label.size(); i++) {
	if(p_label[i] > pmax) {
		lmax = i;
		pmax = p_label[i];
	}
  }
  /*
  std::cout << "p_label: ";
  for(auto f : p_label) {
	std::cout << f << " ";
  }
  std::cout << std::endl;
  
  std::cout << "lmax " << lmax << " pmax " << pmax << std::endl;
  */
  
  if (pmax > threshold) {
    return lmax;
  }
  else {
    return 0;
  }

}

int KnnClassification::getK() {
    return m_k;
}

void KnnClassification::print_kd_stats() {
	ANNkdStats stats;
    m_kdTree->getStats(stats);
    std::cout << stats.dim << " : dimension of space (e.g. 1899 for mail_train)" << std::endl;
	std::cout << stats.n_pts << " : no. of points (e.g. 4000 for mail_train)" << std::endl;
	std::cout << stats.bkt_size << " : bucket size" << std::endl;
	std::cout << stats.n_lf << " : no. of leaves (including trivial)" << std::endl;
	std::cout << stats.n_tl << " : no. of trivial leaves (no points)" << std::endl;
	std::cout << stats.n_spl << " : no. of splitting nodes" << std::endl;
	std::cout << stats.n_shr << " : no. of shrinking nodes (for bd-trees)" << std::endl;
	std::cout << stats.depth << " : depth of tree" << std::endl;
	std::cout << stats.sum_ar << " : sum of leaf aspect ratios" << std::endl;
	std::cout << stats.avg_ar << " : average leaf aspect ratio" << std::endl;
}
