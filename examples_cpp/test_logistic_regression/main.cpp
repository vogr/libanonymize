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


// Random projection params
int const projection_dim = 256;
std::string sampling = "None";


double lambda = 0.1;
int N_TRAIN = 200;
int N_TEST = 200;

// Stop iterating when difference is smaller than threshold
double EPSILON_NEWTON = 0.1;

// learning rate for gradient descent
double EPSILON_GD = 0.001;
double ALPHA_GD = 0.001;

// learning rate for stochastic descent
// /!\ here epsilon is for the objective function
double EPSILON_SGD = 0.05;
double ALPHA_SGD = 0.05;

double EPSILON_RMSPROP = 0.05;

double DECISION_THRESHOLD = 0.5;

std::string GD_TYPE = "RMSProp";

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
        std::cerr << argv[0] << " <train_file.hdf5> <test_file.hdf5> [N training rows, 0 for all] [N testing rows, 0 or all] [lambda default=0.1] [gradient descent | \"Newton\" \"Simple\" \"SGD\" default=\"RMSProp\"] [Random projection | \"Gaussian\" \"Rademacher\" default=\"None\"\n";
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
      if (GD_TYPE != "Newton" && GD_TYPE != "Simple" && GD_TYPE != "SGD" && GD_TYPE != "RMSProp") {
        std::cerr << "Gradient descent type not recognised.\n";
        return 1;
      }
    }
    if(argc >= 8) {
      sampling = argv[7];
      if (sampling != "None" && sampling != "Rademacher" && sampling != "Gaussian") {
        std::cerr << "Sampling type not recognised.\n";
        return 1;
      }
    }
    
      std::cout << "Loading datasets..." << std::endl;
      H5Easy::File hf_train(argv[1], H5Easy::File::ReadOnly);

      RMatrixXd datapoints_train = [&]() -> RMatrixXd {
        RMatrixXd d = H5Easy::load<RMatrixXd>(hf_train, "representation");
        if (N_TRAIN <= 0 || d.rows() <= N_TRAIN) {
          return d;
        }
        else {
          return d.topRows(N_TRAIN);
        }
      }();
      Eigen::VectorXi true_labels_train =  [&]() -> Eigen::VectorXi {
        Eigen::VectorXi d = H5Easy::load<Eigen::VectorXi>(hf_train, "true_labels");
        if (N_TRAIN <= 0 || d.rows() <= N_TRAIN) {
          return relabel(d, {3,4});
        }
        else {
          return relabel(d.topRows(N_TRAIN), {3,4});
        }
      }();

      std::cout << "Loaded " << datapoints_train.rows() << " rows.\n";

      RandomProjection projecter {static_cast<int>(datapoints_train.cols()), projection_dim, sampling};

      std::cout << "Projecting instances in dataset from dimension " << datapoints_train.cols() << " to " << projection_dim << " with sampling: " << sampling << ".\n";

      auto training_dataset_projected = std::make_shared<Dataset>(projecter.Project(std::move(datapoints_train)), std::move(true_labels_train));

      std::cout << "Random projection done." << std::endl;

      std::cout << "Building classifier.\n";
      
      LogisticReg LG {training_dataset_projected, lambda, DECISION_THRESHOLD};
      /* Train model */
      if (GD_TYPE == "Newton") {
          std::cout << "Training classifier with Newton method..." << std::endl;
          LG.fit_newton(EPSILON_NEWTON);
      }
      else if (GD_TYPE == "Simple") {
          std::cout << "Training classifier with a simple fixed-step gradient descent method..." << std::endl;
          LG.fit_gd(EPSILON_GD, ALPHA_GD);
      }
      else if (GD_TYPE == "SGD") {
          std::cout << "Training classifier with fixed step stochastic gradient descent..." << std::endl;
          LG.fit_sgd(EPSILON_SGD, ALPHA_SGD);
      }
      else if (GD_TYPE == "RMSProp") {
          std::cout << "Training classifier with fixed step stochastic gradient descent with RMSProp convergence optimizer..." << std::endl;
          LG.fit_sgd_rmsprop(EPSILON_RMSPROP);
      }
      else {
          // Unreachable.
          return 1;
      }

      std::cout << "Done training." << std::endl;

    {
      std::cout << "Loading testing dataset...\n";
      H5Easy::File hf_test(argv[2], H5Easy::File::ReadOnly);


      RMatrixXd datapoints_test = [&]() -> RMatrixXd {
        RMatrixXd d = H5Easy::load<RMatrixXd>(hf_test, "representation");
        if (N_TEST <= 0 || d.rows() <= N_TEST) {
          return d;
        }
        else {
          return d.topRows(N_TEST);
        }
      }();
      Eigen::VectorXi true_labels_test =  [&]() -> Eigen::VectorXi {
        Eigen::VectorXi d = H5Easy::load<Eigen::VectorXi>(hf_test, "true_labels");
        if (N_TEST <= 0 || d.rows() <= N_TEST) {
          return relabel(d, {3,4});
        }
        else {
          return relabel(d.topRows(N_TEST), {3,4});
        }
      }();
      std::cout << "Loaded " << datapoints_test.rows() << " rows.\n";


      std::cout << "Projecting testing dataset..." << std::endl;
      auto testing_dataset_projected = std::make_shared<Dataset>(projecter.Project(datapoints_test), true_labels_test);

      std::cout << "Start testing." << std::endl;
      ConfusionMatrix cm = LG.EstimateAll(*testing_dataset_projected);
      std::cout << "Classification done. Confusion matrix:\n";
      std::cout << cm.PrintEvaluation();
    }

    return 0;
}
