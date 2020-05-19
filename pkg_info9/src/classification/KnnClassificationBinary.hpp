#pragma once

#include "KnnClassification.hpp"
#include "ConfusionMatrix.hpp"
#include <vector>


class KnnClassificationBinary : public KnnClassification {
private:
	double m_threshold {0.5};
public:
	KnnClassificationBinary(int k, std::shared_ptr<Dataset> dataset, double threshold)
	: KnnClassification(k, std::move(dataset)), m_threshold{threshold} 
	{ };

  int Estimate(Eigen::Ref<Eigen::VectorXd const> const & x) override;
  ConfusionMatrix EstimateAll(Dataset const & test_dataset);
};