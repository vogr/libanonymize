
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <tuple>

#include <Eigen/Dense>
#include <ANN/ANN.h>

// Load after Eigen
#include <highfive/H5Easy.hpp>

#include "helpers.h"
#include "classification/KnnClassification.hpp"
#include "classification/Dataset.hpp"
#include "classification/ConfusionMatrix.hpp"
#include "classification/RandomProjection.hpp"


// List of labels
std::vector<std::string> const LABELS {"O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"};

std::vector<int> const POSITIVE_LABELS {3, 4};
int const K_NN = 10;
bool const DO_PROJECTION = true;
int const PROJECTION_DIM = 128;
//std::string SAMPLE_TYPE = "Rademacher";
std::string const SAMPLE_TYPE = "Gaussian";
double const threshold = 0.2;
int const MAX_ITER = 500;


Eigen::VectorXi relabel(Eigen::Ref<Eigen::VectorXi const> const labels, std::vector<int> const & positive_labels) {
  std::cerr << "Relabeling..." << std::endl;
  Eigen::VectorXi new_labels = Eigen::VectorXi::Zero(labels.size());
  for (auto p : positive_labels) {
    for(int i = 0; i < labels.size(); i++) {
      if (labels[i] == p) {
        new_labels[i] = 1;
      }
    }
  }
  return new_labels;
}

void build_classifier_and_projection(std::string training_dset_fname, KnnClassification & classifier, RandomProjection & projecter ) {
    // Read training file
    H5Easy::File hf(training_dset_fname, H5Easy::File::ReadOnly);
    std::cout << "Will load training datapoints" << std::endl;
    RMatrixXd datapoints = H5Easy::load<RMatrixXd>(hf, "representation");
    std::cout << "Will load training true labels" << std::endl;
    auto true_labels =  H5Easy::load<Eigen::VectorXi>(hf, "true_labels");
    std::cerr << "Matrix of size (" << datapoints.rows() <<", " <<  datapoints.cols() << ") has been read!" << std::endl;
    auto binary_labels = relabel(std::move(true_labels), POSITIVE_LABELS);
    
    
    // Build projecter and evaluate projection
    std::cerr << "We will project datapoints from space of dimension d = " << datapoints.cols() << " to space of dimension l = " << PROJECTION_DIM << "." << std::endl;
    std::cerr << "Generating random projection matrix..." << std::endl;
    projecter = RandomProjection{static_cast<int>(datapoints.cols()), PROJECTION_DIM, SAMPLE_TYPE};
    std::cerr << "Evaluation of the projection quality:" << std::endl;
    projecter.ProjectionQuality(datapoints.topRows(500));
    
    
    std::shared_ptr<Dataset> training_dataset;
    if(!DO_PROJECTION) {
      training_dataset = std::make_shared<Dataset>(std::move(datapoints), std::move(binary_labels));
    }
    else {
      // Project dataset
      std::cerr << "Projecting datapoints in smaller space..." << std::endl;
      training_dataset = std::make_shared<Dataset>(projecter.Project(std::move(datapoints)), std::move(binary_labels));
      std::cerr << "Projection done." << std::endl;
    }

    // Training Knn-classifier
    std::cerr << "Training knn classifier with k = " << K_NN << " (i.e. building kd-tree)" << std::endl;
    classifier = KnnClassification{K_NN, training_dataset};
    std::cerr << "Training done." << std::endl;
  }


int main(int argc, char **argv) {
	// Disable sync between iostreams and C-style IO.
	std::ios::sync_with_stdio(false);
	

	
	if (argc < 3) {
		std::cerr << "Usage:\n\t" << argv[0] << " <training_dataset.hdf5> <testing_dataset.hdf5>" << std::endl;
		return 0;
	}
	
	std::string training_dset_fname {argv[1]};
	std::string testing_dset_fname {argv[2]};
	
  KnnClassification classifier;
  RandomProjection projecter;
  build_classifier_and_projection(training_dset_fname, classifier, projecter);


  RMatrixXd testing_datapoints;
  Eigen::VectorXi testing_true_labels;
  Eigen::VectorXi testing_binary_labels;
  {
    std::cerr << std::endl;
    std::cerr << "Reading testing dataset into memory" << std::endl;

    H5Easy::File hf(testing_dset_fname, H5Easy::File::ReadOnly);
    std::cout << "Will load datapoints..." << std::endl;
    auto datapoints = H5Easy::load<RMatrixXd>(hf, "representation");
    std::cout << "Done." << std::endl;
    if(! DO_PROJECTION) {
      testing_datapoints = std::move(datapoints);
    }
    else {
      std::cerr << "Projecting testing dataset..." << std::endl;
      testing_datapoints = projecter.Project(std::move(datapoints));
      std::cerr << "Done." << std::endl;
    }

    std::cout << "Will load true labels" << std::endl;
    testing_true_labels =  H5Easy::load<Eigen::VectorXi>(hf, "true_labels");
    std::cerr << "Read of " << datapoints.rows() << " rows done!" << std::endl;
    testing_binary_labels = relabel(testing_true_labels, POSITIVE_LABELS);
  }
  
  
	assert(testing_datapoints.rows() == testing_true_labels.rows());
  assert(testing_datapoints.cols() == classifier.getDim());

	
	ConfusionMatrix confusion_matrix;
	int const n = testing_datapoints.rows();
  
	for (int i = 0; i < std::min(n, MAX_ITER) ; i++) {
		if (i % 50 == 0) {
			std::cout << "Processing point " << i << std::endl;
		}
		int l = classifier.EstimateBinary(testing_datapoints.row(i), threshold);
		confusion_matrix.AddPrediction(testing_binary_labels(i), l);
		
		std::cout << "\t" << LABELS[testing_true_labels(i)] << "\t->\t" << ((l == 1) ? "I-PER" : "neg") << std::endl;
	}
	
	std::cout << std::endl << "Confusion matrix:" << std::endl;
	confusion_matrix.PrintEvaluation();
}
