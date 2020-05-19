#include "Dataset_TD6.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>

#ifndef CLASSIFICATION_HPP
#define CLASSIFICATION_HPP

/** 
	The Classification_TD6 class is an abstract class that will be the basis of the KnnClassification_TD6 classe.
*/
class Classification_TD6{
protected:
    /**
      The pointer to a dataset.
    */
	Dataset_TD6* m_dataset;
    /**
      The column to do classification on.
    */
	int m_col_class;
public:
    /**
      The constructor sets private attributes dataset (as a pointer) and the column to do classification on (as an int).
    */
	Classification_TD6(Dataset_TD6* dataset, int col_class);
    /**
      The dataset getter.
    */
	Dataset_TD6* getDataset_TD6();
    /**
      The col_class getter.
    */
	int getColClass();
    /**
      The Estimate method is virtual: it (could) depend(s) on the Classification_TD6 model(s) implemented (here we use only the KnnClassification_TD6 class).
    */
	virtual int Estimate(const Eigen::VectorXd & y , double threshold=0.5) = 0;
};

#endif //CLASSIFICATION_HPP
