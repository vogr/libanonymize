#pragma once


#include <iostream>
#include <memory>
#include <vector>
#include <string>

#include "../helpers.h"


#include <Eigen/Dense>
// Load after Eigen
#include <highfive/H5Easy.hpp>


//using RMatrixXf = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;


/**
  The Dataset class encapsulates a dataset in an Eigen::MatrixXd and provides a kind of interface to manipulate it.
*/
class Dataset {
    public:
		/**
		 * Restore default constructor.
		 */
		 //Dataset() = default;
        /**
          The constructor needs the path of the file as a string.
        */
       
        explicit Dataset(const std::string & file);
        /**
          Dataset list-initialization (needed for RandomProjection).
        */
        Dataset(RMatrixXd instances, Eigen::VectorXi labels)
        : m_instances{instances}, m_labels{labels}
        { max_label = m_labels.maxCoeff(); };
          
        /**
          The Show method displays the number of instances and columns of the Dataset.
          @param verbose If set to True, the Dataset is also printed.
        */
        void Show(bool verbose) const;
        /**
         Returns a view of the instance.
        @param i Instance number (= row) to get.
        */
    	Eigen::Ref<Eigen::VectorXd> getInstance(int i) { return m_instances.row(i); };
    	/**
    	 * Return the label of the instance
    	 */
    	 int getLabel(int i) const { return m_labels(i); };
        /**
          The getter to the number of instances / samples.
        */
    	  int getNbrSamples() const {return m_instances.rows();};
        /**
          The getter to the dimension of the dataset.
        */
    	  int getDim() const { return m_instances.cols();};
		/**
		 * Get maximum label value
		 */
		 int getMaxLabel() const { return max_label; };
    private:
		/**
          The dataset is stored as a MatrixXd.
        */
        int max_label {0};
        RMatrixXd m_instances;
		Eigen::VectorXi m_labels;

};
