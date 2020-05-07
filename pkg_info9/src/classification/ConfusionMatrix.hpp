
#pragma once

#include <vector>

/**
  The ConfusionMatrix class .
*/
class ConfusionMatrix {

    public:
        /**
          The standard constructor.
        */
        ConfusionMatrix();
        
        /**
          Adding a single observation to the (unnormalized) confusion matrix:
        @param true_label (int) the true label of the new point to add to the confusion matrix.
        @param predicted_label (double) the threshold above which class label '1' will be predicted.
        */
        void AddPrediction(int true_label, int predicted_label, std::vector<int> positive_labels);

        /**
          Prints the confusion matrix and the evaluation metrics.
        */
        void PrintEvaluation() const;
        
        /**
        The number of true positive (amounts to an accessor to an accessor of one of the cells of the confusion matrix array).
        */
        int GetTP() const;
    
        /**
          The number of true negative (amounts to an accessor to an accessor of one of the cells of the confusion matrix array).
        */
        int GetTN() const;
    
        /**
          The number of false positive (amounts to an accessor to an accessor of one of the cells of the confusion matrix array).
        */
        int GetFP() const;
    
        /**
          The number of false negative (amounts to an accessor to an accessor of one of the cells of the confusion matrix array).
        */
        int GetFN() const;
    
        /**
          The F-score: 2⋅Precision⋅Recal / (lPrecision+Recall).
        */
        double f_score() const;
    
        /**
          The precision: TP/(TP+FP).
        */
        double precision() const;
    
        /**
          The error rate: number of entries in the diagonal over total number of entries in the array.
        */
        double error_rate() const;
    
        /**
          The detection rate: TP/(TP+FN).
        */
        double detection_rate() const;
    
        /**
          The false alarm rate: FP/(FP+TN).
        */
        double false_alarm_rate() const;

    private:
        /**
        The actual confusion matrix as a 2 by 2 array.
        */
        int m_confusion_matrix[2][2];
};
