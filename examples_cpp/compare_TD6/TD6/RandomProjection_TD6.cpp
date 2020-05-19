#include "RandomProjection_TD6.hpp"
#include <Eigen/Dense> // for MatrixXd
#include <Eigen/Core> // for MatrixXd
#include <iostream> // for cout
#include <random> // for random number generators
#include <chrono> 

using namespace std;

Eigen::MatrixXd RandomProjection_TD6::RandomGaussianMatrix(int d, int projection_dim) {
    // Random number generator initialization
    default_random_engine generator;
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    // Distribution declaration
    normal_distribution<double> distribution(0,1.0/std::sqrt(projection_dim));
    // The projection matrix as a d x projection_dim Eigen::MatrixXd 
    // (could probably made more efficient since it does not have to be dynamically sized)
    
    Eigen::MatrixXd projection_matrix(d, projection_dim);
    projection_matrix.setZero();
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < projection_dim; ++j) {
          projection_matrix(i,j) = distribution(generator);
        }
    }
    return projection_matrix;
}

Eigen::MatrixXi RandomProjection_TD6::RandomRademacherMatrix(int d, int projection_dim) {
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
    Eigen::MatrixXi projection_matrix(d, projection_dim);
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

RandomProjection_TD6::RandomProjection_TD6(int original_dimension, int col_class, int projection_dim, std::string type_sample) {
    // Initialize private attributes
    m_original_dimension = original_dimension;
    m_col_class = col_class;
    m_projection_dim = projection_dim;
    m_type_sample = type_sample;
    
    if (type_sample == "Gaussian") {
      m_projection = RandomGaussianMatrix(original_dimension, projection_dim);
    }
    else {
      m_projection = std::sqrt(3. / projection_dim) * RandomRademacherMatrix(original_dimension, projection_dim).cast<double>();
    }
}

void RandomProjection_TD6::ProjectionQuality(Dataset_TD6 *dataset) {
    Dataset_TD6 projected_dataset = Project(dataset);

    std::cout << "Calculating mean pairwise distance in the original dataset (this may take time):" << std::endl;

    // The cumulative norm between all pairs of points
    double sum_norm = 0.0;

    // TODO - Optional Exercise 2 : A costly loop over all pairs of points
    int n = dataset->getNbrSamples();
    int d = dataset->getDim();

    for (int i = 0; i < n; i++) {
      auto const p = dataset->getInstance(i);
      for(int j = i+1; j < n; j++) {
        auto const q = dataset->getInstance(j);
        double norm2 = 0.;
        for(int k = 0; k < m_original_dimension; k++) {
          norm2 += (p[k] - q[k]) * (p[k] - q[k]);
        }
        sum_norm += std::sqrt(norm2);
      }
    }

    // Number of pairs of points
    sum_norm /= n*(n-1)/2;

    std::cout << sum_norm << std::endl;

    // Same for projected data
    std::cout << "Calculating mean pairwise distance in the projected dataset:" << std::endl;

    double sum_norm_projected = 0.0;

    for (int i = 0; i < n; i++) {
      auto const p = projected_dataset.getInstance(i);
      for(int j = i+1; j < n; j++) {
        auto const q = projected_dataset.getInstance(j);
        double norm2 = 0.;
        for(int k = 0; k < m_projection_dim; k++) {
          norm2 += (p[k] - q[k]) * (p[k] - q[k]);
        }
        sum_norm_projected += std::sqrt(norm2);
      }
    }

    sum_norm_projected/=n*(n-1)/2;

    std::cout << sum_norm_projected << std::endl;
}

Dataset_TD6 RandomProjection_TD6::Project(Dataset_TD6 *dataset) {
    if (dataset->getDim()-1 < m_projection_dim) {
        std::cerr << "Impossible to project on higher dimensions!" << std::endl;
    }

    // Gathering all columns in a Eigen::Matrix
    Eigen::MatrixXd data(dataset->getNbrSamples(), dataset->getDim());
    for (int i=0; i<dataset->getNbrSamples(); i++) {
        std::vector<double> sample = dataset->getInstance(i);
        for (int j=1, j2=0; j<dataset->getDim() && j2<dataset->getDim(); j++, j2++) {
            if (j==(m_col_class+1) && j2==m_col_class) {
                j--;
                continue;
            }
            data(i,j) = sample[j2];
        }
        // The col_class goes first
        data(i,0) = sample[m_col_class];
    }

    // Matrix multiplication except col_class
    Eigen::MatrixXd projected_data = data.block(0,1,dataset->getNbrSamples(),dataset->getDim()-1) * m_projection;
    /*
    // PB : ajoute une colonne de 0 à droite mais écrase la 1 colonne (qui contient des valeurs!)
    projected_data.conservativeResize(projected_data.rows(), projected_data.cols()+1);
    projected_data.col(0) = data.col(0);
    */
    // Attribute m_dataset of class Dataset_TD6 is a std::vector< std::vector<double> >
    std::vector< std::vector<double> > _dataset(projected_data.rows(), std::vector<double>(projected_data.cols()+1, 0));
    // Copy each element
	for(size_t i = 0; i < projected_data.rows(); i++) {
		for(size_t j = 0; j < m_col_class; j++)
			_dataset[i][j] = projected_data(i,j);

        _dataset[i][m_col_class] = data(i,0);

        for(size_t j = m_col_class +1 ; j < projected_data.cols()+1; j++)
            _dataset[i][j] = projected_data(i,j-1);
    }
    // Call to constructor
    Dataset_TD6 dataset_class(_dataset);

    return dataset_class;
}

int RandomProjection_TD6::getOriginalDimension() {
    return m_original_dimension;
}

int RandomProjection_TD6::getColClass() {
    return m_col_class;
}

int RandomProjection_TD6::getProjectionDim(){
    return m_projection_dim;
}

std::string RandomProjection_TD6::getTypeSample(){
    return m_type_sample;
}

Eigen::MatrixXd RandomProjection_TD6::getProjection(){
    return m_projection;
}
