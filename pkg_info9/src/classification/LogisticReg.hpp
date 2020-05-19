#pragma once

#include <Eigen/Dense> // for MatrixXd
#include <Eigen/Core> // for MatrixXd

#include <memory>

#include "Dataset.hpp"
#include "Classification.hpp"
#include "ConfusionMatrix.hpp"


class LogisticReg : public Classification {
    private:
        Eigen::VectorXd beta_1;
        std::shared_ptr<Dataset> m_dataset;
        int m_dim {0};
        int m_nb_instances {0};
        double m_lambda {0.};

        double m_decision_threshold {0.5};

        // Regularized cost function
        double J();
    public:
        LogisticReg(std::shared_ptr<Dataset> dataset, double lambda, double decision_threshold);
    

        //Sigmoid function used in Logistic Regression. Values in [0,1]
        double sigmoid(double t);
        
        /* Training method. Uses Newton-Raphson's method to fit the parameter vector Beta
           Computes Gradiant and Hessian to do so */
        void fit_newton(double epsilon);
        
        /* Simpler gradient descent iteration to minimize regularized cost function
           with learning rate alpha. */
        void fit_gd(double epsilon, double alpha);
        
        //Computes the probability of label being 1. If higehr than threshold, then label is predicted to be 1
        int Estimate(Eigen::Ref<Eigen::VectorXd const> const &x) override;
        // Estimate label of all observables in a dataset and compare then to the true labels (found in the dataset).
        // All the labels should be binary.
        ConfusionMatrix EstimateAll(Dataset const & test_dataset);
};
