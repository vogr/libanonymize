#include "RandomProjection.hpp"
#include <Eigen/Dense> // for MatrixXd
#include <Eigen/Core> // for MatrixXd
#include <iostream> // for cout
#include <random> // for random number generators
#include <chrono> 

using namespace std;

RMatrixXd RandomProjection::RandomGaussianMatrix(int d, int projection_dim) {
    // Random number generator initialization
    default_random_engine generator;
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    // Distribution declaration
    normal_distribution<double> distribution(0,1.0/std::sqrt(projection_dim));
    // The projection matrix as a d x projection_dim Eigen::MatrixXd 
    // (could probably made more efficient since it does not have to be dynamically sized)
    
    RMatrixXd projection_matrix(d, projection_dim);
    projection_matrix.setZero();
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < projection_dim; ++j) {
          projection_matrix(i,j) = distribution(generator);
        }
    }
    return projection_matrix;
}

RMatrixXi RandomProjection::RandomRademacherMatrix(int d, int projection_dim) {
    // Random number generator initialization
    default_random_engine generator_sign;
    generator_sign.seed(std::chrono::system_clock::now().time_since_epoch().count());
    // Distribution declaration
    std::bernoulli_distribution distribution_sign(0.5);
    // Same for bit
    default_random_engine generator_bit;
    generator_bit.seed(std::chrono::system_clock::now().time_since_epoch().count());
    std::bernoulli_distribution distribution_bit(1.0/3.0);
    // The projection matrix as a d x projection_dim Eigen::SparseMatrix<bool> 
    RMatrixXi projection_matrix(d, projection_dim);
    projection_matrix.setZero();
    for (int i=0; i<d; ++i) {
        for (int j=0; j<projection_dim; ++j) {
            // Random number generation and matrix filling
            bool sign = distribution_sign(generator_sign);
            bool bit = distribution_bit(generator_bit);
            // To fill an entry of a SparseMatrix, we need the coeffRef method
            if (sign & bit) {
                projection_matrix(i,j) = 1;
            } else if (bit) {
                projection_matrix(i,j) = -1;
            }
        }
    }
    return projection_matrix;
}

RandomProjection::RandomProjection(int original_dimension, int projection_dim, std::string type_sample) :
m_original_dimension {original_dimension},
m_projection_dim {projection_dim},
m_type_sample {std::move(type_sample)}
{

    // Initialize projection matrix
    if (m_type_sample == "Gaussian") {
      m_projection = RandomGaussianMatrix(original_dimension, projection_dim);
    }
    else {
      m_projection = std::sqrt(3. / projection_dim) * RandomRademacherMatrix(original_dimension, projection_dim).cast<double>();
    }
}

void RandomProjection::ProjectionQuality(Eigen::Ref<RMatrixXd const> const & datapoints) const {
    RMatrixXd projected_datapoints = Project(datapoints);

	std::cerr << "Original dataset size for evaluation: " << datapoints.rows() << ", " << datapoints.cols() << std::endl;
	std::cerr << "Projected dataset size for evaluation: " << projected_datapoints.rows() << ", " << projected_datapoints.cols() << std::endl;

    std::cerr << "Calculating mean pairwise distance in the original dataset (this may take time):" << std::endl;

    // The cumulative norm between all pairs of points
    double sum_norm = 0.0;

    // TODO - Optional Exercise 2 : A costly loop over all pairs of points
    int n = datapoints.rows();

    for (int i = 0; i < n; i++) {
	  if( i % (n/20) == 0) { std::cerr << "."; }
      for(int j = i+1; j < n; j++) {
        sum_norm += (datapoints.row(i) - datapoints.row(j)).norm();
      }
    }

    // Number of pairs of points
    sum_norm /= n*(n-1)/2;

    std::cerr << "\nResult: " << sum_norm << std::endl;

    // Same for projected data
    std::cerr << "Calculating mean pairwise distance in the projected dataset:" << std::endl;

    double sum_norm_projected = 0.0;

    for (int i = 0; i < n; i++) {
	  if( i % (n/20) == 0) { std::cerr << "."; }
      for(int j = i+1; j < n; j++) {
        sum_norm_projected += (projected_datapoints.row(i) - projected_datapoints.row(j)).norm();
      }
    }

    sum_norm_projected/=n*(n-1)/2;

    std::cerr << "\nResult: " << sum_norm_projected << std::endl;
}

RMatrixXd RandomProjection::Project(Eigen::Ref<RMatrixXd const> const & datapoints) const {
    if (datapoints.cols() < m_projection_dim) {
        std::cerr << "Impossible to project on higher dimensions!" << std::endl;
    }

    /***************************************************
    **** Sometimes weird results, adding .eval() seems 
    **** to be more stable ?
    ****************************************************/

    return (datapoints * m_projection).eval();
}

int RandomProjection::getOriginalDimension() const {
    return m_original_dimension;
}


int RandomProjection::getProjectionDim() const {
    return m_projection_dim;
}

std::string RandomProjection::getTypeSample() const {
    return m_type_sample;
}


Eigen::Ref<RMatrixXd const> RandomProjection::getProjection() const {
    return m_projection;
}

