#include "KnnClassificationMulticlass.hpp"


KnnClassificationMulticlass::KnnClassificationMulticlass(int k, std::shared_ptr<Dataset> dataset, std::vector<std::string> labels) 
	: KnnClassification(k, std::move(dataset)), m_labels{std::move(labels)}
	{
		total_label_weight.resize(m_dataset->getMaxLabel() + 1);
		int const n = m_dataset->getNbrSamples();
		for(int i = 0; i < n; i++) {
			total_label_weight[m_dataset->getLabel(i)] += 1. / n;
		}
	}


int KnnClassificationMulticlass::Estimate(Eigen::Ref<Eigen::VectorXd const> const & x) {
  //std::cout << "Input vector size:" << x.size() << std::endl;
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



  // On donne un poids 1 / R a un point dont le label apparait avec un ratio R dans
  // le jeu de donnée : un label très représenté à un poids moins important qu'un label
  // peu représenté.
  std::vector<double> label_value(m_dataset->getMaxLabel() + 1, 0.);
  for(auto i : nn_idx) {
    auto const label = m_dataset->getLabel(i);
    label_value[label] +=  1. / total_label_weight[label];
  }


  size_t label_max = 0;
  double max_value = label_value[0];

  for (size_t i = 1; i < label_value.size(); i++) {
    if(label_value[i] > max_value) {
      label_max = i;
      max_value = label_value[i];
    }
  }
  return label_max;

}

ConfusionMatrixMulticlass KnnClassificationMulticlass::EstimateAll(Dataset const & test_dataset) {
	ConfusionMatrixMulticlass cm{m_labels};

	int const n = test_dataset.getNbrSamples();

	for (int i = 0; i < n ; i++) {
    int l = Estimate(test_dataset.getInstance(i));
    cm.AddPrediction(test_dataset.getLabel(i), l);
  }
   return cm;
}