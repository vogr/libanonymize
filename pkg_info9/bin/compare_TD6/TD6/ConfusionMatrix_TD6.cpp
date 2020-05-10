#include "ConfusionMatrix_TD6.hpp"
#include <iostream>

using namespace std;

ConfusionMatrix_TD6::ConfusionMatrix_TD6() {
  // Populate 2x2 matrix with 0s
  for(int i = 0; i < 2; i++) {
    for(int j = 0; j < 2 ; j++) {
      m_confusion_matrix[i][j] = 0;
    }
  }
}

ConfusionMatrix_TD6::~ConfusionMatrix_TD6() {
    // Attribute m_confusion_matrix is deleted automatically
}

void ConfusionMatrix_TD6::AddPrediction(int true_label, int predicted_label) {
  m_confusion_matrix[true_label][predicted_label] += 1;
}

void ConfusionMatrix_TD6::PrintEvaluation() const{
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

int ConfusionMatrix_TD6::GetTP() const {
    return m_confusion_matrix[1][1];
}

int ConfusionMatrix_TD6::GetTN() const {
   return m_confusion_matrix[0][0];
}

int ConfusionMatrix_TD6::GetFP() const {
    return m_confusion_matrix[0][1];
}

int ConfusionMatrix_TD6::GetFN() const {
  return m_confusion_matrix[1][0];
}

double ConfusionMatrix_TD6::f_score() const {
  double const p = precision();
  double const d = detection_rate();
  return 2 * p * d / (p + d);
}

double ConfusionMatrix_TD6::precision() const {
  return static_cast<double>(GetTP()) / (GetTP() + GetFP());
}

double ConfusionMatrix_TD6::error_rate() const {
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

double ConfusionMatrix_TD6::detection_rate() const {
  return static_cast<double>(GetTP()) / (GetTP() + GetFN());
}

double ConfusionMatrix_TD6::false_alarm_rate() const {
  return static_cast<double>(GetFP()) / (GetFP() + GetTN());
}
