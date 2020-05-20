#include "LogisticReg.hpp"
#include <cmath> // exp

#include <vector>

/*
 * Formulas taken from:
 *     - http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex5/ex5.html
 *     - https://mlelarge.github.io/dataflowr-slides/X/lesson4.html
 */


LogisticReg::LogisticReg(std::shared_ptr<Dataset> dataset, double lambda, double decision_threshold)
: Classification{std::move(dataset)}, m_lambda{lambda}, m_decision_threshold{decision_threshold}
{
    m_dim = m_dataset->getDim();
    m_nb_instances = m_dataset->getNbrSamples();
    // Set beta_1 to zeros
    beta_1 = Eigen::VectorXd::Zero(m_dim+1);  // one more dimension for beta0
}


double LogisticReg::J() {
  /**
   * Regularized loss function of the logistic regression.
   */

  double s = 0;
  for (int i = 0; i < m_nb_instances; i++) {
    // extend observation x into [1 x] = X
    Eigen::VectorXd X(1 + m_dim);
    X << 1, m_dataset->getInstance(i);
    // y takes its values in {0,1}
    int y = m_dataset->getLabel(i);

    double delta = sigmoid(X.transpose() * beta_1);
    s += - 1. / m_dim * ( y * std::log(delta) + (1 - y) * std::log(1 - delta) );
  }
  // Add regularization term, exclude intercept.
  s += m_lambda / (2. * m_dim) * beta_1.bottomRows(m_dim).squaredNorm();
  return s;
}

double LogisticReg::sigmoid(double t){
    /**
     * The logistic sigmoid function.
     */
    return 1.0/(1.0 + std::exp(-t));
}


void LogisticReg::fit_gd(double epsilon, double alpha){
    /**
     * Fit logistic regression coefficients Beta_1 using a simple fixed-step gradient descent.
     */
    std::cout << "Before:\n";
    std::cout << "\tbeta_1(0) = " << beta_1(0) << std::endl;
    std::cout << "\tJ(beta1) = " << J() << std::endl;
    std::cout << "beginning Gradient Descent with alpha = " << alpha << "." << std::endl;
    double difference = 2 * epsilon; // Used to determine when to get out of the loop
    do{
        //Compute gradient
        Eigen::VectorXd gradient = Eigen::VectorXd::Zero(m_dim+1);
        // loop over all instances to compute the gradient of the regularized loss function
        for(int i = 0; i < m_nb_instances; i++){
            // extend observation x into [1 x] = X
            Eigen::VectorXd X(1 + m_dim);
            X << 1, m_dataset->getInstance(i);
            // y takes its values in {0,1}
            int y = m_dataset->getLabel(i);

            double delta_1 = sigmoid(X.transpose() * beta_1);
            gradient += 1. / m_dim * (delta_1 - y) * X;
        }
        // Regularize : exclude intercept
        gradient.bottomRows(m_dim) += m_lambda / m_dim * beta_1.bottomRows(m_dim);

        
        // Next step to take
        Eigen::VectorXd d = - alpha * gradient;
        beta_1 += d;
        // Compute norm of change
        difference = d.norm();
        std::cout << "After one iteration:\n";
        std::cout << "\tdifference = " << difference << std::endl;
        std::cout << "\tbeta_1(0) = " << beta_1(0) << std::endl;
        std::cout << "\tJ(beta1) = " << J() << std::endl;

        //repeat descent if we did not converge.
    } while(difference > epsilon); 
}



void LogisticReg::fit_newton(double epsilon){
    /**
     * Uses Newton-Raphson's method to compute an estimator of the parameter vector Beta, by minimizing the loss function.
     * This vector will then be used to classify.
     */

    std::cout << "Before:\n";
    std::cout << "\tbeta_1(0) = " << beta_1(0) << std::endl;
    std::cout << "\tJ(beta1) = " << J() << std::endl;
    std::cout << "beginning Newton-Raphson's Method" << std::endl;


    double difference = 2 * epsilon; //Used to determine when to get out of the loop
    do{
    
        // Compute gradient and Hessian
        Eigen::VectorXd gradient = Eigen::VectorXd::Zero(m_dim+1);
        Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(m_dim+1, m_dim+1);
        // loop over all instances
        for(int i = 0; i < m_nb_instances; i++){
            // extend observation x into [1 x] = X
            Eigen::VectorXd X(1 + m_dim);
            X << 1, m_dataset->getInstance(i);
            // y takes its values in {0,1}
            int y = m_dataset->getLabel(i);

            double delta_1 = sigmoid(X.transpose() * beta_1);
            gradient += 1. / m_dim * (delta_1 - y) * X;
            hessian += 1. / m_dim * delta_1 * (1 - delta_1) * X * X.transpose();
        }

        // Regularize gradient : exclude intercept
        gradient.bottomRows(m_dim) += m_lambda / m_dim * beta_1.bottomRows(m_dim);

        {
            // Regularize Hessian : exclude intercept
            Eigen::MatrixXd matr = Eigen::MatrixXd::Identity(1 + m_dim, 1 + m_dim);
            matr(0,0) = 0;
            hessian += m_lambda / m_dim * matr;
        }

        std::cout << "Gradient and Hessian computed" << std::endl;

        
        // Next step to take in Newton-Raphson's method
        Eigen::VectorXd d = - hessian.inverse() * gradient;
        beta_1 += d;
        // Compute norm of change
        difference = d.norm();
        std::cout << "After one iteration:\n";
        std::cout << "\tdifference = " << difference << std::endl;
        std::cout << "\tbeta_1(0) = " << beta_1(0) << std::endl;
        std::cout << "\tJ(beta1) = " << J() << std::endl;

        //repeat Newton-Raphson if we still did not converge.
    } while(difference > epsilon); 
    //Else, we stop and have found a good estimator
}

void LogisticReg::fit_sgd(double epsilon, double alpha){
    /**
     * Fit logistic regression coefficients Beta_1 using a fixed-step stochastic gradient descent, with
     * mini-batching (512 entries per batch). 
     */
    std::cout << "Before:\n";
    std::cout << "\tbeta_1(0) = " << beta_1(0) << std::endl;
    std::cout << "\tJ(beta1) = " << J() << std::endl;
    std::cout << "Beginning Stochastic Gradient Descent with alpha = " << alpha << "." << std::endl;
    
    std::vector<int> shuffled_indexes(m_nb_instances);

    for (int i = 0; i < m_nb_instances; i ++) {
        shuffled_indexes[i] = i;
    }

    // Stop after max_n_iter_no_change iterations with little improvement to objective function
    int const max_n_iter_no_change = 3;
    int n_iter_no_change = 0;
    double prev_J = J();
    do{
        // one epoch: iterate over random permutation of the obervations
        std::random_shuffle(shuffled_indexes.begin(), shuffled_indexes.end());
        
        // t : id of current instance in full batch
        int t = 0;
        while(t < m_nb_instances) {
            
            Eigen::VectorXd approx_gradient = Eigen::VectorXd::Zero(m_dim + 1);

            // Approximate gradient over a mini-batch of 512 observations. All the operations on
            // the batch are performed by Eigen, and are therefore vectorised.
            for(int t0 = t ; t < std::min(t0 + 512, m_nb_instances); t++) {
                // i : id of the observation in the dataset.
                int i = shuffled_indexes[t];
                // extend observation x_i into [1 x] = X
                Eigen::VectorXd X(1 + m_dim);
                X << 1, m_dataset->getInstance(i);
                // y takes its values in {0,1}
                int y = m_dataset->getLabel(i);

                double delta_1 = sigmoid(X.transpose() * beta_1);
                approx_gradient += 1. / m_dim * (delta_1 - y) * X;
            }

            // Regularize (exclude the intercept)
            approx_gradient.bottomRows(m_dim) += m_lambda / m_dim * beta_1.bottomRows(m_dim);

            // Take next step (fixed size learning rate alpha)
            beta_1 += - alpha * approx_gradient;
        }
    
        // Compute loss after the epoch
        double new_J = J();
        double J_diff = std::abs(new_J - prev_J);
        prev_J = new_J;

        std::cout << "After one epoch:\n";
        std::cout << "\tJ(beta1) = " << new_J << std::endl;
        std::cout << "\t|new_J - prev_J| = " << J_diff << std::endl;
        std::cout << "\tbeta_1(0) = " << beta_1(0) << std::endl;

        if(J_diff < epsilon) {
            n_iter_no_change++;
        }
        else {
            n_iter_no_change = 0;
        }

    } while(n_iter_no_change < max_n_iter_no_change); 
}


void LogisticReg::fit_sgd_rmsprop(double epsilon){
    /**
     * Fit logistic regression coefficients Beta_1 using an adaptative-step stochastic gradient descent, with
     * mini-batching (512 entries per batch). The step size is adapted per coefficient at each step using
     * the RMSProp convergence accelerator.
     */
    std::cout << "Before:\n";
    std::cout << "\tbeta_1(0) = " << beta_1(0) << std::endl;
    std::cout << "\tJ(beta1) = " << J() << std::endl;
    std::cout << "Beginning SGD with RMSProp convergence accelerator." << std::endl;
    

    // S will hold the "exponential averages" used (and updated) by the RMSProp convergence
    // optimizer 
    // See https://mlelarge.github.io/dataflowr-slides/X/lesson4.html#27 for a formula
    // Here S[i] holds s_t,i (at iteration t).
    // S is an Array, meaning that it will only use coefficient-wise operations
    // (in particular : coefficient-wise product)
    double const gamma = 0.9;
    double const eta = 0.001;
    double const s_eps = 1e-7;
    Eigen::ArrayXd S(1 + m_dim);
    for (int i = 0; i < S.rows(); i ++) {
        S(i) = 0.;
    }

    std::vector<int> shuffled_indexes(m_nb_instances);
    for (int i = 0; i < m_nb_instances; i ++) {
        shuffled_indexes[i] = i;
    }

    // Stop after max_n_iter_no_change iterations with little improvement to objective function
    int const max_n_iter_no_change = 3;
    int n_iter_no_change = 0;
    double prev_J = J();
    do{
        // one epoch: iterate over random permutation of the obervations
        std::random_shuffle(shuffled_indexes.begin(), shuffled_indexes.end());
        // t : id of current instance in full batch
        int t = 0;
        while(t < m_nb_instances) {
            // Approximate gradient over a mini-batch of 512 observations. All the operations on
            // the batch are performed by Eigen, and are therefore vectorised.
            Eigen::VectorXd approx_gradient = Eigen::VectorXd::Zero(m_dim + 1);
            for(int t0 = t ; t < std::min(t0 + 512, m_nb_instances); t++) {
                // i : id of the observation in the dataset.
                int i = shuffled_indexes[t];
                // extend observation x_i into [1 x] = X
                Eigen::VectorXd X(1 + m_dim);
                X << 1, m_dataset->getInstance(i);
                // y takes its values in {0,1}
                int y = m_dataset->getLabel(i);

                double delta_1 = sigmoid(X.transpose() * beta_1);
                approx_gradient += 1. / m_dim * (delta_1 - y) * X;
            }
           
            // Regularize (exclude the intercept)
            approx_gradient.bottomRows(m_dim) += m_lambda / m_dim * beta_1.bottomRows(m_dim);

            // Update S:
            // approx_gradient as an array, so .square() works on coefficents.
            S = gamma * S + (1 - gamma) * approx_gradient.array().square();

            // Take next step (with adaptative learning)
            // /!\ addition, sqrt, inverse and product are coefficient-wise operations (because we use S
            // and approx_gradient as Eigen::Arrays)
            beta_1 +=  - (eta * (S + s_eps).sqrt().inverse() * approx_gradient.array()).matrix();
        }
    
        // Compute loss after the epoch
        double new_J = J();
        double J_diff = std::abs(new_J - prev_J);
        prev_J = new_J;
        // next step
        // Compute norm of change
        std::cout << "After one epoch:\n";
        std::cout << "\tJ(beta1) = " << new_J << std::endl;
        std::cout << "\t|new_J - prev_J| = " << J_diff << std::endl;
        std::cout << "\tbeta_1(0) = " << beta_1(0) << std::endl;

        if(J_diff < epsilon) {
            n_iter_no_change++;
        }
        else {
            n_iter_no_change = 0;
        }

    } while(n_iter_no_change < max_n_iter_no_change); 
}



int LogisticReg::Estimate(Eigen::Ref<Eigen::VectorXd const> const & x){
    /**
     * Compute probabilty that label == 1
     */

    // extend observation x_i into [1 x] = X
    Eigen::VectorXd X = Eigen::VectorXd(m_dim+1);
    X << 1. , x;

    double prob = sigmoid(X.transpose() * beta_1);

    // Predict label to be 1 if probability is higher than threshold
    return (prob > m_decision_threshold) ? 1 : 0; 
}

ConfusionMatrix LogisticReg::EstimateAll(Dataset const & test_dataset) {
  /**
   * Estimate label of all observations in the test_dataset, and compare the predicted
   * label with the true label of the observation (also found in the dataset). Uses these
   * labels to fill a confusion matrix, returned by the function.
   */
  ConfusionMatrix cm;

  int const n = test_dataset.getNbrSamples();

  for (int i = 0; i < n ; i++) {
    int l = Estimate(test_dataset.getInstance(i));
    cm.AddPrediction(test_dataset.getLabel(i), l);
  }
   return cm;
}
