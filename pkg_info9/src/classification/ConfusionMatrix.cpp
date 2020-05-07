#include "ConfusionMatrix.hpp"
#include <iostream>

using namespace std;

ConfusionMatrix::ConfusionMatrix() {
  // Populate 2x2 matrix with 0s
  for(int i = 0; i < 2; i++) {
    for(int j = 0; j < 2 ; j++) {
      m_confusion_matrix[i][j] = 0;
    }
  }
}

void ConfusionMatrix::AddPrediction(int true_label, int predicted_label, std::vector<int> positive_labels) {
  int col = 0, row = 0;
  for(auto l : positive_labels) {
	  if(true_label == l) {
		  col = 1; // true label is a positive occurence
	  }
	  if (predicted_label == l) {
		  row = 1; // predicted label is a positive occurence
	  }
  }
  m_confusion_matrix[col][row] += 1;
}

void ConfusionMatrix::PrintEvaluation() const{
    // Prints the confusion matrix
    cout <<"\t\tPredicted\n";
    cout <<"\t\t0\t1\n";
    cout <<"Actual\t0\t"
        <<GetTN() <<"\t"
        <<GetFP() <<endl;
    cout <<"\t1\t"
        <<GetFN() <<"\t"
        <<GetTP() <<endl <<endl;
    // Prints the estimators
    cout <<"Error rate\t\t"
        <<error_rate() <<endl;
    cout <<"False alarm rate\t"
        <<false_alarm_rate() <<endl;
    cout <<"Detection rate\t\t"
        <<detection_rate() <<endl;
    cout <<"F-score\t\t\t"
        <<f_score() <<endl;
    cout <<"Precision\t\t"
        <<precision() <<endl;
}

int ConfusionMatrix::GetTP() const {
    return m_confusion_matrix[1][1];
}

int ConfusionMatrix::GetTN() const {
   return m_confusion_matrix[0][0];
}

int ConfusionMatrix::GetFP() const {
    return m_confusion_matrix[0][1];
}

int ConfusionMatrix::GetFN() const {
  return m_confusion_matrix[1][0];
}

double ConfusionMatrix::f_score() const {
  double const p = precision();
  double const d = detection_rate();
  return 2 * p * d / (p + d);
}

double ConfusionMatrix::precision() const {
  return static_cast<double>(GetTP()) / (GetTP() + GetFP());
}

double ConfusionMatrix::error_rate() const {
  //The error rate: number of entries in the diagonal over
  // total number of entries in the array.
  int s = 0.;
  for(int i = 0; i < 2; i++) {
    for(int j = 0; j < 2 ; j++) {
      s += m_confusion_matrix[i][j];
    }
  }
  double d = m_confusion_matrix[0][1] + m_confusion_matrix[1][0];
  return d / s;
}

double ConfusionMatrix::detection_rate() const {
  return static_cast<double>(GetTP()) / (GetTP() + GetFN());
}

double ConfusionMatrix::false_alarm_rate() const {
  return static_cast<double>(GetFP()) / (GetFP() + GetTN());
}
