#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

#include <Eigen/Dense>
// Load after Eigen
#include <highfive/H5Easy.hpp>

#include "ConfusionMatrix.hpp"
#include "LogisticReg.hpp"
#include "RandomProjection.hpp"

#include <cstdlib>
#include <cassert>
using namespace std;

double lambda = 10.;
int N_TRAIN = 200;
int N_TEST = 200;

// Stop iterating when difference is smaller than threshold
double EPSILON_NEWTON = 0.1;
double EPSILON_GD = 0.001;

// learning rate for gradient descent
double ALPHA = 0.001;
double DECISION_THRESHOLD = 0.5;

std::string GD_TYPE = "Simple";

Eigen::VectorXi relabel(Eigen::Ref<Eigen::VectorXi const> const labels, std::vector<int> const & positive_labels) {
  // Transform labels to a number in {0,1}, where 1 indicates positive instances. 
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


int main(int argc, char** argv) {
    if(argc < 3) {
        std::cerr << "Usage\n\t";
        std::cerr << argv[0] << " <train_file.hdf5> <test_file.hdf5> [N training rows] [N testing rows] [lambda] [gradient descent | \"Newton\" or \"Simple\"]\n";
        std::cerr << "lambda: the regularizing parameter.\n";
        return 1;
    }
    if(argc >= 4) {
        N_TRAIN = std::atoi(argv[3]);
    }
    if(argc >= 5) {
        N_TEST = std::atoi(argv[4]);
    }
    if(argc >= 6) {
        lambda = std::atof(argv[5]);
    }
    if(argc >= 7) {
      GD_TYPE = argv[6];
      if (GD_TYPE != "Newton" && GD_TYPE != "Simple") {
        std::cerr << "Gradient descent type not recognised.\n";
        return 1;
      }
    }
    std::cout << "Loading datasets..." << std::endl;

    H5Easy::File hf_train(argv[1], H5Easy::File::ReadOnly);
    RMatrixXd datapoints_train = H5Easy::load<RMatrixXd>(hf_train, "representation").topRows(N_TRAIN);
    Eigen::VectorXi true_labels_train =  relabel(H5Easy::load<Eigen::VectorXi>(hf_train, "true_labels").topRows(N_TRAIN), {3,4});
    
    H5Easy::File hf_test(argv[2], H5Easy::File::ReadOnly);
    RMatrixXd datapoints_test = H5Easy::load<RMatrixXd>(hf_test, "representation").topRows(N_TEST);
    Eigen::VectorXi true_labels_test =  relabel(H5Easy::load<Eigen::VectorXi>(hf_test, "true_labels").topRows(N_TEST), {3,4});

    // Random projection
    int const projection_dim = 128;
    std::string const sampling = "Gaussian";
    std::cout << "Performing Random Projection" << std::endl;

    RandomProjection projecter {static_cast<int>(datapoints_train.cols()), projection_dim, sampling};
    auto training_dataset_projected = std::make_shared<Dataset>(projecter.Project(datapoints_train), true_labels_train);
    auto testing_dataset_projected = std::make_shared<Dataset>(projecter.Project(datapoints_test), true_labels_test);

    std::cout << "Random projection done." << std::endl;

    std::cout << "Building classifier.\n";
    
    LogisticReg LG(training_dataset_projected, lambda, DECISION_THRESHOLD);
    /* Train model */
    if (GD_TYPE == "Newton") {
        std::cout << "Training classifier with Newton method..." << std::endl;
        LG.fit_newton(EPSILON_NEWTON);
    }
    else if (GD_TYPE == "Simple") {
        std::cout << "Training classifier with a simple fixed-step gradient descent method..." << std::endl;
        LG.fit_gd(EPSILON_GD, ALPHA);
    }
    else{
        // Unreachable.
        return 1;
    }

    std::cout << "Done training." << std::endl;

    std::cout << "Start testing." << std::endl;
    ConfusionMatrix cm = LG.EstimateAll(*testing_dataset_projected);
    std::cout << "Classification done. Confusion matrix:\n";
    std::cout << cm.PrintEvaluation();

    return 0;
}
