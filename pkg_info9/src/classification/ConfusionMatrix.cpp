#include "ConfusionMatrix.hpp"
#include <iostream>
#include <sstream>

ConfusionMatrix::ConfusionMatrix() {
  // Populate 2x2 matrix with 0s
  for(int i = 0; i < 2; i++) {
    for(int j = 0; j < 2 ; j++) {
      m_confusion_matrix[i][j] = 0;
    }
  }
}

void ConfusionMatrix::AddPrediction(int true_label, int predicted_label) {
  // Accepts only binary labels (1 or 0) !
  m_confusion_matrix[true_label][predicted_label] += 1;
}

std::string ConfusionMatrix::PrintEvaluation() const{
  std::stringstream ss;
    // Prints the confusion matrix
    ss <<"\t\tPredicted\n";
    ss <<"\t\t0\t1\n";
    ss <<"Actual\t0\t"
        <<GetTN() <<"\t"
        <<GetFP() << std::endl;
    ss <<"\t1\t"
        <<GetFN() <<"\t"
        <<GetTP() <<std::endl <<std::endl;
    // Prints the estimators
    ss <<"Error rate\t\t"
        <<error_rate() <<std::endl;
    ss <<"False alarm rate\t"
        <<false_alarm_rate() <<std::endl;
    ss <<"Detection rate\t\t"
        <<detection_rate() <<std::endl;
    ss <<"F-score\t\t\t"
        <<f_score() <<std::endl;
    ss <<"Precision\t\t"
        <<precision() <<std::endl;
    return ss.str();
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
