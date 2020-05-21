#pragma once

#include <Eigen/Dense> // for MatrixXd
#include <Eigen/Core> // for MatrixXd

#include <memory>

#include "../helpers.h"
#include "Dataset.hpp"
#include "Classification.hpp"
#include "ConfusionMatrixMulticlass.hpp"


class LogisticRegMultinomial : public Classification {
    private:
        /*
        W = (k x (d+1)) row matrix
        | beta_1 |
        | beta_2 |
        |  ....  |
        | beta_k |

        beta_i = [b0, ..., bk, bk+1]
        /!\ b0 is the intercept

        Activation of observation x is a(x) = W * [1 x]
        Estimation of the class y that x belongs to by ~one-hot vector
            y_hat(x) = softmax(a(x))
        y_hat(x)_j = P[Y = j |X = x]
        */

        RMatrixXd W;
        Eigen::VectorXd b;
        int m_dim {0}; // d size of observations x
        int m_nb_instances {0}; // n number of observations

        std::vector<std::string>  m_labels;
        int m_nb_labels {0}; // k number of labels, y takes values in {0, ..., k-1}

        double m_lambda {0.};
        Eigen::VectorXd y_hat(Eigen::Ref<Eigen::VectorXd const> const & x);
    public:
        LogisticRegMultinomial(std::shared_ptr<Dataset> dataset, double _lambda, std::vector<std::string> _labels);    

        //Softmax function used in Logistic Regression.
        Eigen::VectorXd softmax(Eigen::Ref<Eigen::VectorXd const> const & Z);

        // Regularized cost function
        double J();

        /* SGD with mini-batching using fixed learning rate alpha */
        void fit_sgd(double epsilon, double alpha);
        
        /* SGD with mini-batching using RMSProp convergence optimizer */
        void fit_sgd_rmsprop(double epsilon);

        int Estimate(Eigen::Ref<Eigen::VectorXd const> const &x) override;
        // Estimate label of all observables in a dataset and compare then to the true labels (found in the dataset).
        // All the labels should be binary.
        ConfusionMatrixMulticlass EstimateAll(Dataset const & test_dataset);
};
