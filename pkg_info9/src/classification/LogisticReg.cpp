#include "LogisticReg.hpp"
#include <cmath> // exp

#include <vector>

/*
 * Formulas taken from:
 * http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex5/ex5.html
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
  s += m_lambda / (2. * m_dim) * beta_1.bottomRows(m_dim).squaredNorm();
  return s;
}

double LogisticReg::sigmoid(double t){
    return 1.0/(1.0 + std::exp(-t));

}


void LogisticReg::fit_gd(double epsilon, double alpha){
    std::cout << "Before:\n";
    std::cout << "\tbeta_1(0) = " << beta_1(0) << std::endl;
    std::cout << "\tJ(beta1) = " << J() << std::endl;
    std::cout << "beginning Gradient Descent with alpha = " << alpha << "." << std::endl;
double difference = 2 * epsilon; //Used to determine when to get out of the loop
    do{
        //Compute gradien
        Eigen::VectorXd gradient = Eigen::VectorXd::Zero(m_dim+1);
        // loop over all instances
        for(int i = 0; i < m_nb_instances; i++){
            // extend observation x into [1 x] = X
            Eigen::VectorXd X(1 + m_dim);
            X << 1, m_dataset->getInstance(i);
            // y takes its values in {0,1}
            int y = m_dataset->getLabel(i);

            double delta_1 = sigmoid(X.transpose() * beta_1);
            gradient += 1. / m_dim * (delta_1 - y) * X;
        }
        {
            Eigen::VectorXd vert(1 + m_dim);
            vert << 0, beta_1.bottomRows(m_dim);
            gradient += m_lambda / m_dim * vert;
        }

        
        // next step
        Eigen::VectorXd d = - alpha * gradient;
        beta_1 += d;
        // Compute norm of change
        difference = d.norm();
        std::cout << "After one iteration:\n";
        std::cout << "\tdifference = " << difference << std::endl;
        std::cout << "\tbeta_1(0) = " << beta_1(0) << std::endl;
        std::cout << "\tJ(beta1) = " << J() << std::endl;

        //repeat Newton-Raphson if we still did not converge.
    } while(difference > epsilon); 
}



void LogisticReg::fit_newton(double epsilon){
    /*Uses Newton-Raphson's method to compute an estimator of the parameter vector Beta
    This vector will then be used to classify*/
    
    
    // ****Find loss function l(beta_1) that we want to maximize****
    /*Change of variable Z = 1(Y=1) => Already done, we take labels 0 OR 1
    l(beta) = sum1_n(zi[1 xi(t)]beta - log(1+exp([1 xi(t)]beta)) )
    We don't actually need to compute it, we only need gradient and Hessian*/

    std::cout << "Before:\n";
    std::cout << "\tbeta_1(0) = " << beta_1(0) << std::endl;
    std::cout << "\tJ(beta1) = " << J() << std::endl;
    std::cout << "beginning Newton-Raphson's Method" << std::endl;


    double difference = 2 * epsilon; //Used to determine when to get out of the loop
    do{
    
        //Compute gradient and Hessian
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
        {
            Eigen::VectorXd vert(1 + m_dim);
            vert << 0, beta_1.bottomRows(m_dim);
            gradient += m_lambda / m_dim * vert;
        }
        {
            Eigen::MatrixXd matr = Eigen::MatrixXd::Identity(1 + m_dim, 1 + m_dim);
            matr(0,0) = 0;
            hessian += m_lambda / m_dim * matr;
        }

        std::cout << "gradient and hessian computed" << std::endl;

        
        //Newton-Raphson's method
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

int LogisticReg::Estimate(Eigen::Ref<Eigen::VectorXd const> const & x){
    // Compute probabilty of label = 1
    Eigen::VectorXd X = Eigen::VectorXd(m_dim+1);
    X << 1. , x;
    double prob = sigmoid(X.transpose() * beta_1);

    // predict label to be 1 if probability is higher than threshold
    return (prob > m_decision_threshold) ? 1 : 0; 
}

ConfusionMatrix LogisticReg::EstimateAll(Dataset const & test_dataset) {
  ConfusionMatrix cm;

  int const n = test_dataset.getNbrSamples();

  for (int i = 0; i < n ; i++) {
    int l = Estimate(test_dataset.getInstance(i));
    cm.AddPrediction(test_dataset.getLabel(i), l);
  }
   return cm;
}