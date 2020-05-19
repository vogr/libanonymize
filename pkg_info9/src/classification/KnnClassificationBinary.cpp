#include "KnnClassificationBinary.hpp"


int KnnClassificationBinary::Estimate(Eigen::Ref<Eigen::VectorXd const> const & x) {
  auto const d = m_dataset->getDim();
  auto const k = getK();

  // index of nearest neighbors returned by the search
  std::vector<int> nn_idx(k, 0);
  std::vector<double> dists(k, 0.);


  auto query_point = annAllocPt(d);
  for(int j = 0; j < d; j++) {
    query_point[j] = x(j);
  }

  // NOTE : we could directly pass a vector to the data underlying
  // the Eigen::Ref<Eigen::VectorXd> to ANN. A bit hacky, and
  // does not accept const qualifier... but saves a copy.

  m_kdTree->annkSearch(
      query_point,      // query point
      k,    // number of near neighbors to find
      nn_idx.data(), // MODIFIED : id of nearest neighbors
      dists.data() // MODIFIED : array of dists to these neighbors
  );
  annDeallocPt(query_point);

  double s = 0.;
  for(size_t j = 0 ; j < nn_idx.size(); j++) {
    int i = nn_idx[j];
    auto const label = m_dataset->getLabel(i);
    s += static_cast<double>(label) / k;
  }

  if (s > m_threshold) {
    return 1;
  }
  else {
    return 0;
  }
}


ConfusionMatrix KnnClassificationBinary::EstimateAll(Dataset const & test_dataset) {
  ConfusionMatrix cm;

  int const n = test_dataset.getNbrSamples();

  for (int i = 0; i < n ; i++) {
    int l = Estimate(test_dataset.getInstance(i));
    cm.AddPrediction(test_dataset.getLabel(i), l);
  }
   return cm;
}