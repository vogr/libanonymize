
#include <iostream>
#include <memory>
#include <vector>
#include <string>

#include <Eigen/Dense>
#include <ANN/ANN.h>

// Load after Eigen
#include <highfive/H5Easy.hpp>

#include "helpers.h"
#include "classification/KnnClassification.hpp"
#include "classification/Dataset.hpp"
#include "classification/ConfusionMatrix.hpp"
#include "classification/RandomProjection.hpp"

#include "TD6/KnnClassification_TD6.hpp"
#include "TD6/Dataset_TD6.hpp"
#include "TD6/ConfusionMatrix_TD6.hpp"
#include "TD6/RandomProjection_TD6.hpp"


//int const K_NN = 10;
//int const PROJECTION_DIM = 10;
double const threshold = 0.2;
//int const MAX_ITER = 500;
int const COL_CLASS = 0;

std::string const training_dset_fname_hdf5 {"data/mail_train.hdf5"};
std::string const testing_dset_fname_hdf5 {"data/mail_test.hdf5"};
std::string const training_dset_fname_csv {"data/mail_train.csv"};
std::string const testing_dset_fname_csv {"data/mail_test.csv"};


int main(int argc, char **argv) {
  // Disable sync between iostreams and C-style IO.
  std::ios::sync_with_stdio(false);
  if(argc < 4) {
    return 1;
  }
  int const K_NN = std::atoi(argv[1]);
  int const PROJECTION_DIM = std::atoi(argv[2]);
  double const threshold = std::atof(argv[3]);
  std::string const SAMPLE_TYPE = (argc >= 5) ? argv[4] : "Rademacher";

  {
    std::cerr << "*****\tTD6 IMPLEMENTATION\t*****\n\n";
    std::cerr << "Loading datasets\n";
    clock_t t_datasets = clock();
    Dataset_TD6 train_dataset(training_dset_fname_csv.data());
    Dataset_TD6 class_dataset(testing_dset_fname_csv.data());
    t_datasets = clock() - t_datasets;
    std::cerr << "Execution time : " << (t_datasets*1000)/CLOCKS_PER_SEC <<"ms\n\n";
    
   // Performing Knn on original data
    std::cerr << "Performing Knn on projected data" << std::endl;
    clock_t t_knn_train_orig = clock();
    KnnClassification_TD6 knn_class_orig(K_NN, &train_dataset, 0);
    t_knn_train_orig = clock() - t_knn_train_orig;
    std::cerr <<"Execution time: "
         <<(t_knn_train_orig*1000)/CLOCKS_PER_SEC
         <<"ms\n\n";

    {
        ANNkdStats stats;
        knn_class_orig.getKdTree()->getStats(stats);
        std::cout << stats.dim << " : dimension of space (e.g. 1899 for mail_train)" << std::endl;
        std::cout << stats.n_pts << " : no. of points (e.g. 4000 for mail_train)" << std::endl;
        std::cout << stats.bkt_size << " : bucket size" << std::endl;
        std::cout << stats.n_lf << " : no. of leaves (including trivial)" << std::endl;
        std::cout << stats.n_tl << " : no. of trivial leaves (no points)" << std::endl;
        std::cout << stats.n_spl << " : no. of splitting nodes" << std::endl;
        std::cout << stats.n_shr << " : no. of shrinking nodes (for bd-trees)" << std::endl;
        std::cout << stats.depth << " : depth of tree" << std::endl;
        std::cout << stats.sum_ar << " : sum of leaf aspect ratios" << std::endl;
        std::cout << stats.avg_ar << " : average leaf aspect ratio\n" << std::endl;
    }


  // Knn test on projected data
    std::cerr << "Predicting Knn on original data" << std::endl;
    ConfusionMatrix_TD6 confusion_matrix_orig;
    clock_t t_knn_test_orig = clock();
    for (int i=0; i < class_dataset.getNbrSamples(); i++) {
        std::vector<double> sample = class_dataset.getInstance(i);
        Eigen::VectorXd query(class_dataset.getDim()-1);
        int true_label;
        for (int j=0, j2=0; j<class_dataset.getDim()-1 && j2<class_dataset.getDim(); j++, j2++) {
            if (j==COL_CLASS && j2==COL_CLASS) {
                true_label = sample[COL_CLASS];
                j--;
                continue;
            }
            query[j] = sample[j2];
        }
        int predicted_label = knn_class_orig.Estimate(query, threshold);
        //std::cout << true_label << " -> " << predicted_label << std::endl;
        confusion_matrix_orig.AddPrediction(true_label, predicted_label);
    }
    t_knn_test_orig = clock() - t_knn_test_orig;
    std::cerr <<"Execution time: "
         <<(t_knn_test_orig*1000)/CLOCKS_PER_SEC
         <<"ms\n\n";
    confusion_matrix_orig.PrintEvaluation();


    // Random projection
    std::cerr << "Generating Random Projection" << std::endl;
    clock_t t_random_projection = clock();
    RandomProjection_TD6 projection(train_dataset.getDim()-1, COL_CLASS, PROJECTION_DIM, SAMPLE_TYPE);
    t_random_projection = clock() - t_random_projection;
    std::cerr <<"Execution time: "
         <<(t_random_projection*1000)/CLOCKS_PER_SEC
         <<"ms\n\n";

/*
    std::cerr << "Evaluation of the projection quality:" << std::endl;
    clock_t t_projection_quality = clock();
    projection.ProjectionQuality(&train_dataset);
    t_projection_quality = clock() - t_projection_quality;
    std::cerr << "Projection quality execution time : " <<  (t_projection_quality*1000)/CLOCKS_PER_SEC << "ms\n" << std::endl;
*/

    std::cerr << "Projecting datasets" << std::endl;
    clock_t t_project = clock();
    Dataset_TD6 projection_dataset = projection.Project(&train_dataset);
    Dataset_TD6 projection_test_dataset = projection.Project(&class_dataset);
    t_project = clock() - t_project;
    std::cerr << "Projection execution time : " <<  (t_project*1000)/CLOCKS_PER_SEC << "ms\n" << std::endl;
    
    int n = projection_dataset.getNbrSamples();
    int d = projection_dataset.getDim();
    std::cerr<<"Dataset with " << n <<" samples, and "<< d <<" dimensions."<<std::endl;
    /*
    for (int i=0; i< 10; i++) {
        auto v = projection_dataset.getInstance(i);
        for (int j=0; j < d; j++) {
            std::cout << v[j] <<" ";
        }
        std::cout<<std::endl;
    }
    */

    // Performing Knn on projected data
    std::cerr << "Performing Knn on projected data" << std::endl;
    clock_t t_knn_train_projected = clock();
    KnnClassification_TD6 knn_class_projected(K_NN, &projection_dataset, 0);
    t_knn_train_projected = clock() - t_knn_train_projected;
    std::cerr <<"Execution time: "
         <<(t_knn_train_projected*1000)/CLOCKS_PER_SEC
         <<"ms\n\n";

    {
        ANNkdStats stats;
        knn_class_projected.getKdTree()->getStats(stats);
        std::cout << stats.dim << " : dimension of space (e.g. 1899 for mail_train)" << std::endl;
        std::cout << stats.n_pts << " : no. of points (e.g. 4000 for mail_train)" << std::endl;
        std::cout << stats.bkt_size << " : bucket size" << std::endl;
        std::cout << stats.n_lf << " : no. of leaves (including trivial)" << std::endl;
        std::cout << stats.n_tl << " : no. of trivial leaves (no points)" << std::endl;
        std::cout << stats.n_spl << " : no. of splitting nodes" << std::endl;
        std::cout << stats.n_shr << " : no. of shrinking nodes (for bd-trees)" << std::endl;
        std::cout << stats.depth << " : depth of tree" << std::endl;
        std::cout << stats.sum_ar << " : sum of leaf aspect ratios" << std::endl;
        std::cout << stats.avg_ar << " : average leaf aspect ratio\n" << std::endl;
    }

    // Knn test on projected data
    std::cerr << "Predicting Knn on projected data" << std::endl;
    ConfusionMatrix_TD6 confusion_matrix_projected;
    clock_t t_knn_test_projected = clock();
    for (int i=0; i < projection_test_dataset.getNbrSamples(); i++) {
        std::vector<double> sample = projection_test_dataset.getInstance(i);
        Eigen::VectorXd query(projection_test_dataset.getDim()-1);
        int true_label;
        for (int j=0, j2=0; j<projection_test_dataset.getDim()-1 && j2<projection_test_dataset.getDim(); j++, j2++) {
            if (j==COL_CLASS && j2==COL_CLASS) {
                true_label = sample[COL_CLASS];
                j--;
                continue;
            }
            query[j] = sample[j2];
        }
        int predicted_label = knn_class_projected.Estimate(query, threshold);
        //std::cout << true_label << " -> " << predicted_label << std::endl;
        confusion_matrix_projected.AddPrediction(true_label, predicted_label);
    }
    t_knn_test_projected = clock() - t_knn_test_projected;
    std::cerr <<"Execution time: "
         <<(t_knn_test_projected*1000)/CLOCKS_PER_SEC
         <<"ms\n\n";
    confusion_matrix_projected.PrintEvaluation();
  }




  {
    std::cerr << "*****\tOUR IMPLEMENTATION\t*****\n\n";
    std::cerr << "Loading datasets\n";
    clock_t t_datasets = clock();
    H5Easy::File hf_train(training_dset_fname_hdf5, H5Easy::File::ReadOnly);
    auto datapoints_train = H5Easy::load<RMatrixXd>(hf_train, "representation");
    auto true_labels_train =  H5Easy::load<Eigen::VectorXi>(hf_train, "true_labels");
    
    H5Easy::File hf_test(testing_dset_fname_hdf5, H5Easy::File::ReadOnly);
    auto datapoints_test = H5Easy::load<RMatrixXd>(hf_test, "representation");
    auto true_labels_test =  H5Easy::load<Eigen::VectorXi>(hf_test, "true_labels");
    t_datasets = clock() - t_datasets;
    std::cerr << "Execution time : " << (t_datasets*1000)/CLOCKS_PER_SEC <<"ms\n\n";
    
    // Original dataset
    auto training_dataset_orig = std::make_shared<Dataset>(datapoints_train, true_labels_train);

    // Training Knn-classifier
    std::cerr << "Training knn classifier on original data (k = " << K_NN << ")" << std::endl;
    
    clock_t t_knn_train_orig = clock();
    KnnClassification classifier_orig{K_NN, training_dataset_orig};
    t_knn_train_orig = clock() - t_knn_train_orig;
    std::cerr <<"Execution time: "
         <<(t_knn_train_orig*1000)/CLOCKS_PER_SEC
         <<"ms\n\n";

    {
        ANNkdStats stats;
        classifier_orig.get_kd_stats(stats);
        std::cout << stats.dim << " : dimension of space (e.g. 1899 for mail_train)" << std::endl;
        std::cout << stats.n_pts << " : no. of points (e.g. 4000 for mail_train)" << std::endl;
        std::cout << stats.bkt_size << " : bucket size" << std::endl;
        std::cout << stats.n_lf << " : no. of leaves (including trivial)" << std::endl;
        std::cout << stats.n_tl << " : no. of trivial leaves (no points)" << std::endl;
        std::cout << stats.n_spl << " : no. of splitting nodes" << std::endl;
        std::cout << stats.n_shr << " : no. of shrinking nodes (for bd-trees)" << std::endl;
        std::cout << stats.depth << " : depth of tree" << std::endl;
        std::cout << stats.sum_ar << " : sum of leaf aspect ratios" << std::endl;
        std::cout << stats.avg_ar << " : average leaf aspect ratio\n" << std::endl;
    }

    {
        int const n = datapoints_test.rows();
        std::cerr << "Predicting Knn on original data" << std::endl;
        ConfusionMatrix confusion_matrix;
        clock_t t_knn_test_orig = clock();
        for (int i = 0; i < n ; i++) {
          int l = classifier_orig.EstimateBinary(datapoints_test.row(i), threshold);
          //std::cout << true_labels_test.coeff(i) << " -> " << l << std::endl;
          confusion_matrix.AddPrediction(true_labels_test(i), l);
        }
        t_knn_test_orig = clock() - t_knn_test_orig;
        std::cerr <<"Execution time: "
             <<(t_knn_test_orig*1000)/CLOCKS_PER_SEC
             <<"ms\n\n";
        confusion_matrix.PrintEvaluation();
    }

    std::cerr << "We will project datapoints from space of dimension d = " << static_cast<int>(datapoints_train.cols()) << " to space of dimension l = " << PROJECTION_DIM << "." << std::endl;
    std::cerr << "Generating Random Projection" << std::endl;
    clock_t t_random_projection = clock();
    RandomProjection projecter {static_cast<int>(datapoints_train.cols()), PROJECTION_DIM, SAMPLE_TYPE};
    t_random_projection = clock() - t_random_projection;
    std::cerr <<"Execution time: "
         <<(t_random_projection*1000)/CLOCKS_PER_SEC
         <<"ms\n\n";
    

    /*
    std::cerr << "Evaluation of the projection quality:" << std::endl;
    clock_t t_projection_quality = clock();
    projecter.ProjectionQuality(datapoints_train);
    t_projection_quality = clock() - t_projection_quality;
    std::cerr << "Projection quality execution time : " <<  (t_projection_quality*1000)/CLOCKS_PER_SEC << "ms\n" << std::endl;
    */
    
    std::cerr << "Projecting datasets" << std::endl;
    clock_t t_project = clock();

    auto training_dataset_projected = std::make_shared<Dataset>(projecter.Project(datapoints_train), true_labels_train);
    RMatrixXd testing_datapoints_projected = projecter.Project(datapoints_test);
    t_project = clock() - t_project;
    std::cerr << "Projection execution time : " <<  (t_project*1000)/CLOCKS_PER_SEC << "ms\n" << std::endl;

  
    int nt = training_dataset_projected->getNbrSamples();
    int d = training_dataset_projected->getDim();
    std::cerr<<"Dataset with " << nt <<" samples, and "<< d <<" dimensions."<<std::endl;
    /*
    for (int i=0; i< 10; i++) {
        auto v = training_dataset_projected->getInstance(i);
        std::cout << true_labels_train(i) << " ";
        for (int j=0; j < d; j++) {
            std::cout << v(j)<<" ";
        }
        std::cout<<std::endl;
    }
    */
    
    
    // Training Knn-classifier
    std::cerr << "Training knn classifier on projected data (k = " << K_NN << ")" << std::endl;
    
    clock_t t_knn_train_projected = clock();
    KnnClassification classifier_proj{K_NN, training_dataset_projected};
    t_knn_train_projected = clock() - t_knn_train_projected;
    std::cerr <<"Execution time: "
         <<(t_knn_train_projected*1000)/CLOCKS_PER_SEC
         <<"ms\n\n";

    {
        ANNkdStats stats;
        classifier_proj.get_kd_stats(stats);
        std::cout << stats.dim << " : dimension of space (e.g. 1899 for mail_train)" << std::endl;
        std::cout << stats.n_pts << " : no. of points (e.g. 4000 for mail_train)" << std::endl;
        std::cout << stats.bkt_size << " : bucket size" << std::endl;
        std::cout << stats.n_lf << " : no. of leaves (including trivial)" << std::endl;
        std::cout << stats.n_tl << " : no. of trivial leaves (no points)" << std::endl;
        std::cout << stats.n_spl << " : no. of splitting nodes" << std::endl;
        std::cout << stats.n_shr << " : no. of shrinking nodes (for bd-trees)" << std::endl;
        std::cout << stats.depth << " : depth of tree" << std::endl;
        std::cout << stats.sum_ar << " : sum of leaf aspect ratios" << std::endl;
        std::cout << stats.avg_ar << " : average leaf aspect ratio\n" << std::endl;
    }

    {
        int const n = testing_datapoints_projected.rows();
        std::cerr << "Predicting Knn on projected data" << std::endl;
        ConfusionMatrix confusion_matrix;
        clock_t t_knn_test_projected = clock();
        for (int i = 0; i < n ; i++) {
          int l = classifier_proj.EstimateBinary(testing_datapoints_projected.row(i), threshold);
          //std::cout << true_labels_test.coeff(i) << " -> " << l << std::endl;
          confusion_matrix.AddPrediction(true_labels_test(i), l);
        }
        t_knn_test_projected = clock() - t_knn_test_projected;
        std::cerr <<"Execution time: "
             <<(t_knn_test_projected*1000)/CLOCKS_PER_SEC
             <<"ms\n\n";
        confusion_matrix.PrintEvaluation();
    }
  }
}
