#include <Eigen/Dense>
#include <Eigen/Core>
#include "Dataset_TD6.hpp"
#include "Classification_TD6.hpp"
#include <ANN/ANN.h>

#ifndef KNNCLASSIFICATION_HPP
#define KNNCLASSIFICATION_HPP

/**
  The KnnClassification_TD6 class inherits from the Classification_TD6 class, adds the number of neighbours k, the kdTree obtained from the ANN library, and a bunch of methods.
*/
class KnnClassification_TD6 : public Classification_TD6 {
private:
    /**
      The number of neighbours k to compute.
    */
	int m_k;
    /**
      The kdTree obtained from the ANN library.
    */
  // Change this line to compare ANNkd_tree, ANNbd tree and ANNbruteForce
	ANNkd_tree* m_kdTree;
public:
    /**
      The constructor needs:
     @param k the number of neighbours
     @param dataset the pointer to a dataset of class Dataset_TD6
     @param col_class the integer that defines the column index that represents the label
    */
	KnnClassification_TD6(int k, Dataset_TD6* dataset, int col_class);
    /**
      The standard destructor.
    */
	~KnnClassification_TD6();
    /**
      The predicted label for a new instance depending on the chosen thresholds:
     @param x the new instance which output we wish to predict, as a VectorXd of class Eigen.
     @param threshold the threshold in majority vote.
     @returns the prediction as an integer
    */
  int Estimate(const Eigen::VectorXd & x, double threshold=0.5);
    /**
      The getter for the number of neighbors
    */
  int getK();
    /**
      The getter for the kdtree
    */
  ANNkd_tree* getKdTree();
};

#endif //KNNCLASSIFICATION_HPP
