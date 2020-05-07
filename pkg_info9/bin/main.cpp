
#include <iostream>
#include <memory>
#include <vector>
#include <string>

#include <Eigen/Dense>
#include <ANN/ANN.h>

// Load after Eigen
#include <highfive/H5Easy.hpp>

#include "../src/helpers.h"
#include "../src/classification/KnnClassification.hpp"
#include "../src/classification/Dataset.hpp"
#include "../src/classification/ConfusionMatrix.hpp"
#include "../src/classification/RandomProjection.hpp"

int const K_NN = 10;
bool DO_PROJECTION = true;
int const PROJECTION_DIM = 30;
//std::string type_sample = "Rademacher";
std::string const type_sample = "Gaussian";
double const threshold = 0.2;
int const MAX_ITER = 5000;


KnnClassification build_classifier(std::string training_dset_fname) {
	std::shared_ptr<Dataset> training_dataset;
	if(! DO_PROJECTION) {
		training_dataset = std::make_shared<Dataset>(training_dset_fname);
	}
	else {
		H5Easy::File hf(training_dset_fname, H5Easy::File::ReadOnly);
		std::cout << "Will load training datapoints" << std::endl;
		RMatrixXd datapoints = H5Easy::load<RMatrixXd>(hf, "representation");
		std::cout << "Will load training true labels" << std::endl;
		auto true_labels =  H5Easy::load<Eigen::VectorXi>(hf, "true_labels");
		std::cerr << "Matrix of size (" << datapoints.rows() <<", " <<  datapoints.cols() << ") has been read!" << std::endl;
		
		
		std::cerr << "We will project datapoints from space of dimension d = " << datapoints.cols() << " to space of dimension l = " << PROJECTION_DIM << "." << std::endl;
		
		std::cerr << "Generating random projection matrix..." << std::endl;
		RandomProjection const projecter {static_cast<int>(datapoints.cols()), PROJECTION_DIM, type_sample};
		std::cerr << "Evaluation of the projection quality:" << std::endl;
		projecter.ProjectionQuality(datapoints.topRows(500));
		
		std::cerr << "Projecting datapoints in smaller space..." << std::endl;
		training_dataset = std::make_shared<Dataset>(projecter.Project(datapoints), std::move(true_labels));
		std::cerr << "Projection done." << std::endl;
	}
	// Training Knn-classifier
	std::cerr << "Training knn classifier with k = " << K_NN << " (i.e. building kd-tree)" << std::endl;
	KnnClassification classifier{K_NN, training_dataset};
	std::cerr << "Training done." << std::endl;
	return classifier;
}



int main(int argc, char **argv) {
	// Disable sync between iostreams and C-style IO.
	std::ios::sync_with_stdio(false);
	
	// List of labels
	std::vector<std::string> const LABELS {"O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"};
	size_t const N_LABELS = LABELS.size();
	
	
	// Positive labels = the labels of interest. All other are negatives.
	const std::vector<int> POSITIVE_LABELS {3, 4};
	
	
	if (argc < 3) {
		std::cerr << "Usage:\n\t" << argv[0] << " <training_dataset.hdf5> <testing_dataset.hdf5>" << std::endl;
		return 0;
	}
	
	std::string training_dset_fname {argv[1]};
	std::string testing_dset_fname {argv[2]};
	
	
	// Read training file and build knn-classifier
	KnnClassification classifier {build_classifier(training_dset_fname)};

	std::cerr << std::endl;
	std::cerr << "Reading testing dataset into memory" << std::endl;

	H5Easy::File hf(testing_dset_fname, H5Easy::File::ReadOnly);

	std::cout << "Will load datapoints" << std::endl;
	auto datapoints = H5Easy::load<RMatrixXd>(hf, "representation");
	std::cout << "Will load true labels" << std::endl;
	auto true_labels =  H5Easy::load<Eigen::VectorXi>(hf, "true_labels");
	std::cerr << "Read of " << datapoints.rows() << " rows done!" << std::endl;
	
	assert(datapoints.rows() == true_labels.rows());
	
	
	ConfusionMatrix confusion_matrix;
	int const n = datapoints.rows();
	for (int i = 0; i < std::min(n, MAX_ITER) ; i++) {
		if (i % 50 == 0) {
			std::cout << "Processing point " << i << std::endl;
		}
		int l = classifier.Estimate(datapoints.row(i), threshold);
		confusion_matrix.AddPrediction(true_labels(i), l, POSITIVE_LABELS);
		
		//std::cout << "\t" << LABELS[true_labels(i)] << " -> " << LABELS[l] << std::endl;
	}
	
	std::cout << std::endl << "Confusion matrix:" << std::endl;
	confusion_matrix.PrintEvaluation();
}
