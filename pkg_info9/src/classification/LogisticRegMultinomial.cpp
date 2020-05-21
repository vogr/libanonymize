#include "LogisticRegMultinomial.hpp"
#include <cmath> // exp

#include <vector>

/*
 * Formulas for multinomial logistic regression taken from:
 *     https://cedar.buffalo.edu/~srihari/CSE574/Chap4/4.3.4-MultiLogistic.pdf
 * Formulas for RMSProp:
 *     - https://mlelarge.github.io/dataflowr-slides/X/lesson4.html
 */


LogisticRegMultinomial::LogisticRegMultinomial(std::shared_ptr<Dataset> dataset, double _lambda, std::vector<std::string> _labels)
: Classification{std::move(dataset)}, m_labels{std::move(_labels)}, m_lambda{_lambda}
{
    m_nb_labels = m_labels.size();
    m_dim = m_dataset->getDim();
    m_nb_instances = m_dataset->getNbrSamples();
    // Set W to zeros
    W = RMatrixXd::Zero(m_nb_labels, 1 + m_dim);  // one more dimension for beta0
}


Eigen::VectorXd LogisticRegMultinomial::softmax(Eigen::Ref<Eigen::VectorXd const> const & Z){
    /**
     * The softmax function (vectorized):
     *  [..., exp(Z_i), ...] / sum(exp(Z_i))
     */
    // .array() for coefficient-wise operations.
    Eigen::VectorXd eZ = exp(Z.array());
    return eZ /  eZ.sum();
}


double LogisticRegMultinomial::J() {
  /**
   * Regularized loss function of the multinomial logistic regression.
   */

  double s = 0;
  for (int i = 0; i < m_nb_instances; i++) {
    
    // k class x belongs to
    // y = one-hot vector indicating k
    int k = m_dataset->getLabel(i);

    Eigen::VectorXd yh = y_hat(m_dataset->getInstance(i));


    // y * log(yh), but y is one-hot in k
    // prevent log(0) ...
    s += - 1. / m_dim * std::log(yh(k) + 1e-6 );
  }
  
  // TODO : regularize !
  // Add regularization term, exclude intercept.
  s += m_lambda / (2. * m_dim) * W.squaredNorm();

  return s;
}


void LogisticRegMultinomial::fit_sgd(double epsilon, double alpha){
    /**
     * Fit logistic regression coefficients W using an fixed-step stochastic gradient descent, with
     * mini-batching (512 entries per batch).
     */
    std::cout << "Before:\n";
    std::cout << "\tW(0,0) = " << W(0,0) << std::endl;
    std::cout << "\tJ(W) = " << J() << std::endl;
    std::cout << "Beginning SGD with with alpha = " << alpha << " and epsilon = " << epsilon << std::endl;
    


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
            RMatrixXd approx_gradients = RMatrixXd::Zero(m_nb_labels, m_dim + 1);
            for(int t0 = t ; t < std::min(t0 + 512, m_nb_instances); t++) {
                // i : id of the observation in the dataset.
                int i = shuffled_indexes[t];

                // extend x_i in [1 x_i] = X
                Eigen::VectorXd X(1 + m_dim);
                X << 1, m_dataset->getInstance(i);

                Eigen::VectorXd yh = softmax(W * X);

                // k class x belongs to
                // y = one-hot vector indicating k
                int k = m_dataset->getLabel(i);
                Eigen::VectorXd y = Eigen::VectorXd::Zero(m_nb_labels);
                y(k) = 1;

                for (int l = 0; l < m_nb_labels; l++) {
                    approx_gradients.row(l) += 1. / m_dim * (yh(l) - y(l)) * X;


                }
            }
           
           
            // Regularize (exclude the intercept)
            approx_gradients.rightCols(m_dim) += m_lambda / m_dim * W.rightCols(m_dim);

            // Make next step
            W += - alpha * approx_gradients;

        }
    
        // Compute loss after the epoch
        double new_J = J();
        double J_diff = std::abs(new_J - prev_J);
        prev_J = new_J;
        // next step
        // Compute norm of change
        std::cout << "After one epoch:\n";
        std::cout << "\tJ(W) = " << new_J << std::endl;
        std::cout << "\t|new_J - prev_J| = " << J_diff << std::endl;
        std::cout << "\tW(0) = " << W(0,0) << std::endl;

        if(J_diff < epsilon) {
            n_iter_no_change++;
        }
        else {
            n_iter_no_change = 0;
        }

    } while(n_iter_no_change < max_n_iter_no_change); 
}



void LogisticRegMultinomial::fit_sgd_rmsprop(double epsilon){
    /**
     * Fit logistic regression coefficients W using an adaptative-step stochastic gradient descent, with
     * mini-batching (512 entries per batch). The step size is adapted per coefficient at each step using
     * the RMSProp convergence accelerator.
     */
    std::cout << "Before:\n";
    std::cout << "\tW(0,0) = " << W(0,0) << std::endl;
    std::cout << "\tJ(W) = " << J() << std::endl;
    std::cout << "Beginning SGD with with  adaptative step using RMSProp and epsilon = " << epsilon << std::endl;
    


    // S will hold the "exponential averages" used (and updated) by the RMSProp convergence
    // optimizer 
    // See https://mlelarge.github.io/dataflowr-slides/X/lesson4.html#27 for a formula
    // Here S.row(k) = s_t holds coefficient s_t,i (at iteration t).
    // S is an Array, meaning that it will only use coefficient-wise operations
    // (in particular : coefficient-wise product)
    double const gamma = 0.9;
    double const eta = 0.001;
    double const s_eps = 1e-7;
    Eigen::ArrayXd S = Eigen::ArrayXd::Zero(m_nb_labels, 1 + m_dim);


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
            // Approximate gradients over a mini-batch of 512 observations. All the operations on
            // the batch are performed by Eigen, and are therefore vectorised.
            RMatrixXd approx_gradients = RMatrixXd::Zero(m_nb_labels, m_dim + 1);
            for(int t0 = t ; t < std::min(t0 + 512, m_nb_instances); t++) {
                // i : id of the observation in the dataset.
                int i = shuffled_indexes[t];

                // extend x_i in [1 x_i] = X
                Eigen::VectorXd X(1 + m_dim);
                X << 1, m_dataset->getInstance(i);

                Eigen::VectorXd yh = softmax(W * X);

                // k class x belongs to
                // y = one-hot vector indicating k
                int k = m_dataset->getLabel(i);
                Eigen::VectorXd y = Eigen::VectorXd::Zero(m_nb_labels);
                y(k) = 1;

                for (int l = 0; l < m_nb_labels; l++) {
                    approx_gradients.row(l) += 1. / m_dim * (yh(l) - y(l)) * X;


                }
            }

            // Regularize (exclude the intercept)
            approx_gradients.rightCols(m_dim) += m_lambda / m_dim * W.rightCols(m_dim);
    
            

            // Update S:
            // approx_gradient as an array, so .square() works on coefficents.
            S = gamma * S + (1 - gamma) * approx_gradients.array().square();

            // Take next step (with adaptative learning)
            // /!\ addition, sqrt, inverse and product are coefficient-wise operations (because we use S
            // and approx_gradient as Eigen::Arrays)
            
            // Make next step
            W += - (eta * (S + s_eps).sqrt().inverse() * approx_gradients.array()).matrix();
            
        }
    
        // Compute loss after the epoch
        double new_J = J();
        double J_diff = std::abs(new_J - prev_J);
        prev_J = new_J;
        // next step
        // Compute norm of change
        std::cout << "After one epoch:\n";
        std::cout << "\tJ(W) = " << new_J << std::endl;
        std::cout << "\t|new_J - prev_J| = " << J_diff << std::endl;
        std::cout << "\tW(0) = " << W(0,0) << std::endl;

        if(J_diff < epsilon) {
            n_iter_no_change++;
        }
        else {
            n_iter_no_change = 0;
        }

    } while(n_iter_no_change < max_n_iter_no_change); 
}




Eigen::VectorXd LogisticRegMultinomial::y_hat(Eigen::Ref<Eigen::VectorXd const> const & x){
    /**
     * y is a one-hot vector indicating the label of x.
     * Return estimator y_hat of y, using multinomial logistic regression. 
     */
    // extend x_i in [1 x_i] = X
    Eigen::VectorXd X(1 + m_dim);
    X << 1, x;
    return softmax(W * X);
}


int LogisticRegMultinomial::Estimate(Eigen::Ref<Eigen::VectorXd const> const & x){
    /**
     * Return most probable class from y_hat estimator.
     */

    Eigen::VectorXd yh = y_hat(x);
    int k = 0;
    double p_k = yh(k);
    for (int i = 0; i < m_nb_labels; i++) {
        if (yh(i) > p_k) {
            k = i;
            p_k = yh(i);
        }
    }
    return k;
}



ConfusionMatrixMulticlass LogisticRegMultinomial::EstimateAll(Dataset const & test_dataset) {
  /**
   * Estimate label of all observations in the test_dataset, and compare the predicted
   * label with the true label of the observation (also found in the dataset). Uses these
   * labels to fill a confusion matrix, returned by the function.
   */
    ConfusionMatrixMulticlass cm{m_labels};

    int const n = test_dataset.getNbrSamples();

    for (int i = 0; i < n ; i++) {
    int l = Estimate(test_dataset.getInstance(i));
    cm.AddPrediction(test_dataset.getLabel(i), l);
  }
   return cm;
}
