
#include "KnnClassification_TD6.hpp"
#include <iostream>
// #include <fstream>
#include <ANN/ANN.h>

 
KnnClassification_TD6::KnnClassification_TD6(int k, Dataset_TD6* dataset, int col_class)
: Classification_TD6(dataset, col_class), m_k{k}  {
  auto const n = dataset->getNbrSamples();
  auto const d = dataset->getDim() - 1;
  auto points_array = annAllocPts(n, d);
  
  for(size_t i = 0; i < n ; i++) {
    auto const v = dataset->getInstance(i);
    for(size_t j=0 ; j < col_class; j++) {
      points_array[i][j] = v[j];
    }
    for(size_t j = (col_class + 1) ; j < d + 1 ; j++) {
      points_array[i][j-1] = v[j];
    }
  }

  m_kdTree = new ANNkd_tree(points_array, n, d);
}

KnnClassification_TD6::~KnnClassification_TD6() {
    delete m_kdTree;
    annClose();
}

int KnnClassification_TD6::Estimate(const Eigen::VectorXd & x, double threshold) {
  //auto const n = m_dataset->getNbrSamples();
  auto const d = m_dataset->getDim() - 1;

  // index of nearest neighbors returned by the search
  std::vector<int> nn_idx(m_k, 0);
  std::vector<double> dists(m_k, 0.);

  auto query_point = annAllocPt(d);
  for(size_t j = 0; j < d; j++) {
    query_point[j] = x[j];
  }

  m_kdTree->annkSearch(
      query_point,      // query point
      m_k,    // numbe of near neighbors to find
      nn_idx.data(), // MODIFIED : id of nearest neighbors
      dists.data() // MODIFIED : array of dists (can it take nullptr ?)
  );
  annDeallocPt(query_point);

  auto const label_col = getColClass();

  double s = 0.;
  for(int j = 0; j < nn_idx.size(); j++) {
    int i = nn_idx[j];
    auto const v = m_dataset->getInstance(i);
    //std::cout << v[label_col] << "(" << i << "," << dists[j] << ") ";
    s += v[label_col] / m_k;
  }
  //std::cout << " -> " << s << std::endl;


  if (s > threshold) {
    return 1;
  }
  else {
    return 0;
  }
}

int KnnClassification_TD6::getK() {
    return m_k;
}

ANNkd_tree* KnnClassification_TD6::getKdTree() {
    return m_kdTree;
}
