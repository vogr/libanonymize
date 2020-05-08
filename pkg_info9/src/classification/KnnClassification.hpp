#pragma once

#include <memory>

#include <Eigen/Dense>
#include <Eigen/Core>
#include "Dataset.hpp"
#include "Classification.hpp"
#include <ANN/ANN.h>
#include <ANN/ANNperf.h>



/**
  The KnnClassification class inherits from the Classification class, adds the number of neighbours k, the kdTree obtained from the ANN library, and a bunch of methods.
*/
class KnnClassification : public Classification {
private:
    /**
      The number of neighbours k to compute.
    */
	int m_k {2};
    /**
      The kdTree obtained from the ANN library.
    */
    std::shared_ptr<Dataset> dataset;
  // Change this line to compare ANNkd_tree, ANNbd tree and ANNbruteForce
	std::unique_ptr<ANNkd_tree> m_kdTree;
	std::vector<double*> datapoints;
	
public:
    /**
      The constructor needs:
     @param k the number of neighbours
     @param dataset the pointer to a dataset of class Dataset
     @param col_class the integer that defines the column index that represents the label
    */
	KnnClassification(int k, std::shared_ptr<Dataset> dataset);
	/**
	 * Default destructor (rule of zero)
	 */
    /**
      The predicted label for a new instance depending on the chosen thresholds:
     @param x the new instance which output we wish to predict, as a VectorXd (or equivalent Eigen expression) of class Eigen.
     @param threshold the threshold in majority vote.
     @returns the prediction as an integer
    */
  int Estimate(Eigen::Ref<Eigen::VectorXd> x, double threshold=0.5);
  /**
   * Predict label in a dataset with only binary labels (0 or 1)
   */
  int EstimateBinary(Eigen::Ref<Eigen::VectorXd> x, double threshold=0.5);

    /**
      The getter for the number of neighbors
    */
  int getK();
  void print_kd_stats();
};
