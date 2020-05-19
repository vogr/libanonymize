#pragma once

#include <memory>
#include <string>

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
protected:
  /**
    The number of neighbours k to compute.
  */
	int m_k {0};
  /**
    The kdTree obtained from the ANN library.
  */
  // Change this line to compare ANNkd_tree, ANNbd tree and ANNbruteForce
	std::unique_ptr<ANNkd_tree> m_kdTree;
	//std::vector<double*> datapoints;
	ANNpointArray datapoints {nullptr};
public:
    /**
      The constructor needs:
     @param k the number of neighbours
     @param dataset the shared pointer to a dataset of class Dataset
    */
  KnnClassification(int k, std::shared_ptr<Dataset> dataset);
  // Empty classification constructor.
  KnnClassification() = default;

  // make KnnClassification movable but not copyable
  KnnClassification(KnnClassification const & other) = delete;
  KnnClassification& operator=(KnnClassification const & other) noexcept = delete;
  KnnClassification(KnnClassification && other) = default;
  KnnClassification& operator=(KnnClassification && other) noexcept = default;


	/**
	 * Destructor
	 */
  ~KnnClassification();
    /**
      The getter for the number of neighbors
    */
  int getK();
  std::string print_kd_stats();
  /**
   * Getter for dim in dataset.
   */
   int getDim() const { return m_dataset->getDim();};

   void get_kd_stats(ANNkdStats & stats) {m_kdTree->getStats(stats);}
        
};
