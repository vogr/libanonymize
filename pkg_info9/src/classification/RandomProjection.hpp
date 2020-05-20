#pragma once


#include "../helpers.h"

#include "Dataset.hpp"
#include <Eigen/Dense>
#include <Eigen/Core>

/**
  The RandomProjection class .
*/
class RandomProjection {
    private:
        int m_original_dimension {0};
        int m_projection_dim {0};
        std::string m_type_sample;
        RMatrixXd m_projection;
    public:
        /**
          A random Gaussian matrix (0, 1/projection_dim) of size (n, projection_dim).
          @param d original dimension
          @param projection_dim projection dimension (l in the TD)
        */
        static RMatrixXd RandomGaussianMatrix(int d, int projection_dim);
        /**
          A random Gaussian matrix (0, 1/projection_dim) of size (n, projection_dim).
          @param d original dimension
          @param projection_dim projection dimension (l in the TD)
        */
        static RMatrixXi RandomRademacherMatrix(int d, int projection_dim);
        /**
         * Initialize empty RandomProjection
         */
        //RandomProjection() = default;
        //RandomProjection(RandomProjection & other) = default;
        /**
          The constructor.
          @param col_class the classification column
          @param projection_dim projection dimension (l in the TD)
          @param type_sample either "Gaussian" and anything else which would lead to a Rademacher random projection
        */
        RandomProjection() = default;
        RandomProjection(int original_dimension, int projection_dim, std::string type_sample);
        /**
          Verify the quality of the projection as the mean distance between points in the original and projected data
        */
        void ProjectionQuality(Eigen::Ref<RMatrixXd const> const & datapoints) const;
        /**
          Project dataset
          @param dataset
        */
        RMatrixXd Project(Eigen::Ref<RMatrixXd const> const & datapoints) const;   
        /**
          Original dimension getter
        */
        int getOriginalDimension() const;
        /**
          Projection dimension getter
        */
        int getProjectionDim() const;
        /**
          Type of sampling getter
        */
        std::string getTypeSample() const;
        /**
          Projection matrix getter
        */
        Eigen::Ref<RMatrixXd const> getProjection() const;
};
