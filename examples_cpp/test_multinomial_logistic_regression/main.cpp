#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

#include <Eigen/Dense>
// Load after Eigen
#include <highfive/H5Easy.hpp>

#include "ConfusionMatrix.hpp"
#include "LogisticRegMultinomial.hpp"
#include "RandomProjection.hpp"

#include <cstdlib>
#include <cassert>
using namespace std;

std::vector<std::string> const LABELS {"O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"};

// Random projection params
int const projection_dim = 256;
std::string sampling = "None";


//double lambda = 0.1;
int N_TRAIN = 2000;
int N_TEST = 2000;

double EPSILON_RMSPROP = 0.05;

double EPSILON_SGD = 0.05;
double ALPHA_SGD = 0.05;

double LAMBDA = 1;

std::string GD_TYPE = "RMSProp";

int main(int argc, char** argv) {
    if(argc < 3) {
        std::cerr << "Usage\n\t";
        std::cerr << argv[0] << " <train_file.hdf5> <test_file.hdf5> [N training rows, 0 for all] [N testing rows, 0 or all] [lambda] [GD type | \"SGD\" default=\"RMSProp\"] [epsion] [alpha] \n";
        return 1;
    }
    if(argc >= 4) {
        N_TRAIN = std::atoi(argv[3]);
    }
    if(argc >= 5) {
        N_TEST = std::atoi(argv[4]);
    }
    if(argc >= 6) {
      LAMBDA = std::atof(argv[5]);
    }
    if(argc >= 7) {
      GD_TYPE = argv[6];
      if (GD_TYPE != "SGD" && GD_TYPE != "RMSProp") {
        std::cerr << "Gradient descent type not recognised.\n";
        return 1;
      }
    }
    if(argc >= 8) {
      double eps = std::atof(argv[7]);
      EPSILON_SGD = eps;
      EPSILON_RMSPROP = eps;
    }
    if(argc >= 9) {
      ALPHA_SGD = std::atof(argv[8]);
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
          return d;
        }
        else {
          return d.topRows(N_TRAIN);
        }
      }();

      std::cout << "Loaded " << datapoints_train.rows() << " rows.\n";

      RandomProjection projecter {static_cast<int>(datapoints_train.cols()), projection_dim, sampling};

      if (sampling != "None") {
        std::cout << "Projecting instances in dataset from dimension " << datapoints_train.cols() << " to " << projection_dim << " with sampling: " << sampling << ".\n";
      }

      auto training_dataset_projected = std::make_shared<Dataset>(projecter.Project(std::move(datapoints_train)), std::move(true_labels_train));
      
      if (sampling != "None") {
        std::cout << "Random projection done." << std::endl;
      }
      
      std::cout << "Building classifier.\n";
      
      LogisticRegMultinomial LG {training_dataset_projected, LAMBDA, LABELS};
      /* Train model */

      if (GD_TYPE == "SGD") {
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
          return d;
        }
        else {
          return d.topRows(N_TEST);
        }
      }();
      std::cout << "Loaded " << datapoints_test.rows() << " rows.\n";


      std::cout << "Projecting testing dataset..." << std::endl;
      auto testing_dataset_projected = std::make_shared<Dataset>(projecter.Project(datapoints_test), true_labels_test);

      std::cout << "Start testing." << std::endl;
      ConfusionMatrixMulticlass cm = LG.EstimateAll(*testing_dataset_projected);

      std::cerr << std::endl << "Multiclass confusion matrix:" << std::endl;
      std::cerr << cm.PrintMatrix();
      std::cerr << "\nOne vs all confusion matrix for I-PER:\n";
      std::cerr << cm.OneVsAllConfusionMatrix(4).PrintEvaluation();
    }

    return 0;
}
