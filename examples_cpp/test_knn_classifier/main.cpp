
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
#include "classification/KnnClassificationBinary.hpp"
#include "classification/KnnClassificationMulticlass.hpp"
#include "classification/Dataset.hpp"
#include "classification/ConfusionMatrix.hpp"
#include "classification/ConfusionMatrixMulticlass.hpp"
#include "classification/RandomProjection.hpp"


// List of labels
std::vector<std::string> const LABELS {"O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"};

int K_NN = 10;
int PROJECTION_DIM = 128;
//std::string SAMPLE_TYPE = "Rademacher";
std::string const SAMPLE_TYPE = "Gaussian";
double const threshold = 0.2;
int TRAIN_LINES = 10000;
int TEST_LINES = 5000;



void build_classifier_and_projection(std::string training_dset_fname, KnnClassificationBinary & classifier, RandomProjection & projecter );
Eigen::VectorXi relabel(Eigen::Ref<Eigen::VectorXi const> const labels, std::vector<int> const & positive_labels);



int main(int argc, char **argv) {
	// Disable sync between iostreams and C-style IO.
	std::ios::sync_with_stdio(false);
	
	if (argc < 3) {
		std::cerr << "Usage:\n\t" << argv[0] << " <training_dataset.hdf5> <testing_dataset.hdf5> [N train lines=10000] [N test lines=5000] [K number of neighbors] [D projection dimension]" << std::endl;
		return 0;
	}
	
	std::string const training_dset_fname {argv[1]};
	std::string const testing_dset_fname {argv[2]};
	if (argc >= 4) {
    TRAIN_LINES = std::atoi(argv[3]);
  }
  if (argc >= 5) {
    TEST_LINES = std::atoi(argv[4]);
  }
  if (argc >= 6) {
    K_NN = std::atoi(argv[5]);
  }
  if (argc >= 7) {
    PROJECTION_DIM = std::atoi(argv[5]);
  }

  // Read training and testing file
  H5Easy::File hf_train(training_dset_fname, H5Easy::File::ReadOnly);
  H5Easy::File hf_test(testing_dset_fname, H5Easy::File::ReadOnly);

  std::cout << "Will load training and testing datapoints" << std::endl;
  
  RMatrixXd datapoints_train = H5Easy::load<RMatrixXd>(hf_train, "representation").topRows(TRAIN_LINES);
  std::cerr << "Training matrix of size (" << datapoints_train.rows() <<", " <<  datapoints_train.cols() << ") has been read!" << std::endl;    

  RMatrixXd datapoints_test = H5Easy::load<RMatrixXd>(hf_test, "representation").topRows(TEST_LINES);
  std::cerr << "Testing matrix of size (" << datapoints_test.rows() <<", " <<  datapoints_test.cols() << ") has been read!" << std::endl;    


  std::cout << "Will load training and testing true labels" << std::endl;
  auto true_labels_train =  H5Easy::load<Eigen::VectorXi>(hf_train, "true_labels");
  auto true_labels_test =  H5Easy::load<Eigen::VectorXi>(hf_test, "true_labels");



  
  // Build projecter and evaluate projection
  std::cerr << "We will project datapoints from space of dimension d = " << datapoints_test.cols() << " to space of dimension l = " << PROJECTION_DIM << "." << std::endl;
  std::cerr << "Generating random projection matrix..." << std::endl;
  RandomProjection projecter {static_cast<int>(datapoints_train.cols()), PROJECTION_DIM, SAMPLE_TYPE};
  std::cerr << "Evaluation of the projection quality on first 500 rows:" << std::endl;
  projecter.ProjectionQuality(datapoints_train.topRows(std::min(500,TRAIN_LINES)));
  
  std::cerr << "Projecting training datapoints..." << std::endl;
  RMatrixXd datapoints_train_projected = projecter.Project(std::move(datapoints_train));
  std::cerr << "Done.\nProjecting testing datapoints..." << std::endl;
  RMatrixXd datapoints_test_projected = projecter.Project(std::move(datapoints_test));
  std::cerr << "Done." << std::endl;

  {
    std::cerr << "\n\nEVALUATING BINARY KNN CLASSIFIER\n\n";

    auto binary_labels_train = relabel(true_labels_train, {3,4});
    auto binary_labels_test = relabel(true_labels_test, {3,4});

    std::shared_ptr<Dataset> training_dataset = std::make_shared<Dataset>(datapoints_train_projected, std::move(binary_labels_train));

  // Training Knn-classifier
    std::cerr << "Training knn classifier with k = " << K_NN << " (i.e. building kd-tree)" << std::endl;
    KnnClassificationBinary classifier{K_NN, training_dataset, threshold};
    std::cerr << "Training done." << std::endl;

	
  	ConfusionMatrix confusion_matrix;
  	int const n = datapoints_test_projected.rows();
    
  	for (int i = 0; i < n ; i++) {
  		if (i % 1000 == 0) {
  			std::cout << "Processing point " << i << std::endl;
  		}
      int l = classifier.Estimate(datapoints_test_projected.row(i));
  		confusion_matrix.AddPrediction(binary_labels_test(i), l);
  		
  		//std::cout << "\t" << LABELS[true_labels_test(i)] << "\t->\t" << ((l == 1) ? "I-PER" : "neg") << std::endl;
  	}
  	
  	std::cout << std::endl << "Confusion matrix:" << std::endl;
  	std::cerr << confusion_matrix.PrintEvaluation();
  }


  {
    std::cerr << "\n\nEVALUATING MULTICLASS KNN CLASSIFIER\n\n";

    std::shared_ptr<Dataset> training_dataset = std::make_shared<Dataset>(std::move(datapoints_train_projected), std::move(true_labels_train));

  // Training Knn-classifier
    std::cerr << "Training knn classifier with k = " << K_NN << " (i.e. building kd-tree)" << std::endl;
    KnnClassificationMulticlass classifier{K_NN, training_dataset, LABELS};
    std::cerr << "Training done." << std::endl;

  
    ConfusionMatrixMulticlass confusion_matrix(LABELS);
    int const n = datapoints_test_projected.rows();
    
    for (int i = 0; i < n ; i++) {
      if (i % 1000 == 0) {
        std::cout << "Processing point " << i << std::endl;
      }
      int l = classifier.Estimate(datapoints_test_projected.row(i));

      confusion_matrix.AddPrediction(true_labels_test(i), l);
      
      //std::cout << "\t" << LABELS[true_labels_test(i)] << "\t->\t" << LABELS[l] << std::endl;
    }
    
    std::cerr << std::endl << "Multiclass confusion matrix:" << std::endl;
    std::cerr << confusion_matrix.PrintMatrix();
    std::cerr << "\nOne vs all confusion matrix for I-PER:\n";
    std::cerr << confusion_matrix.OneVsAllConfusionMatrix(4).PrintEvaluation();
  }
}



Eigen::VectorXi relabel(Eigen::Ref<Eigen::VectorXi const> const labels, std::vector<int> const & positive_labels) {
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