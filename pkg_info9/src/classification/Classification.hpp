#pragma once

#include <memory>

#include "Dataset.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>


/** 
	The Classification class is an abstract class that will be the basis of the KnnClassification classe.
*/
class Classification{
protected:
    /**
      The pointer to a dataset.
    */
	std::shared_ptr<Dataset> m_dataset;
public:
    /**
      The constructor sets private attributes dataset (as a shared pointer).
    */
	Classification(std::shared_ptr<Dataset> dataset) : m_dataset{std::move(dataset)} { };
    /**
      The dataset getter.
    */
	std::shared_ptr<Dataset> getDataset() { return m_dataset; };
    /**
      The Estimate method is virtual: it (could) depend(s) on the Classification model(s) implemented (here we use only the KnnClassification class).
      x should be an Eigen::Ref<Eigen::VectorXd const> const & but this required copying the vector before passing it to ANN.
    */
	virtual int Estimate(Eigen::Ref<Eigen::VectorXd> x, double threshold=0.5) = 0;
};
