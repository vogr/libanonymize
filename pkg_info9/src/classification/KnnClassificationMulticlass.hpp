#pragma once

#include "KnnClassification.hpp"
#include "ConfusionMatrixMulticlass.hpp"

#include <string>
#include <vector>


class KnnClassificationMulticlass : public KnnClassification {
private:
	std::vector<double> total_label_weight;
	std::vector<std::string> m_labels;
public:
	KnnClassificationMulticlass(int k, std::shared_ptr<Dataset> dataset, std::vector<std::string> labels);
	KnnClassificationMulticlass() = default;

  int Estimate(Eigen::Ref<Eigen::VectorXd const> const & x) override;
  ConfusionMatrixMulticlass EstimateAll(Dataset const & test_dataset);

};