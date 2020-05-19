#include "ConfusionMatrixMulticlass.hpp"
#include <sstream>

using namespace std;


void ConfusionMatrixMulticlass::AddPrediction(int true_label, int predicted_label) {
  m_confusion_matrix[true_label][predicted_label] += 1;
}

ConfusionMatrix ConfusionMatrixMulticlass::OneVsAllConfusionMatrix(int label_one) {
  std::array<std::array<int, 2>, 2> v;
  for(int i = 0; i < 2 ; i ++) {
    for(int j=0; j<2; j++) {
      v[i][j] = 0;
    }
  }

  for (size_t i = 0; i < m_confusion_matrix.size(); i++) {
    for (size_t j = 0; j < m_confusion_matrix.size(); j++) {
      v[(static_cast<int>(i) == label_one) ? 1 : 0][(static_cast<int>(j) == label_one) ? 1 : 0] += m_confusion_matrix[i][j];
    }
  }
  return ConfusionMatrix{v};
}

std::string ConfusionMatrixMulticlass::PrintMatrix(){
  std::stringstream ss;
  ss << "\t";
  for (auto & l : m_labels) {
    ss << l << "\t";
  }
  ss << "\n";
  for(size_t i = 0 ; i < m_confusion_matrix.size(); i++) {
    auto & v = m_confusion_matrix[i];
    ss << m_labels[i] << "\t";
    for (auto c : v) {
      ss << c << "\t";
    }
    ss << std::endl;
  }
  return ss.str();
}


std::string ConfusionMatrixMulticlass::PrintEvaluation(){
 std::stringstream ss;
 ss << PrintMatrix();
  for(size_t l = 0; l < m_confusion_matrix.size() ; l++) {
    ss << "\nOne vs all confusion matrix for label " << m_labels[l] << ":" << std::endl;
    ConfusionMatrix cm = OneVsAllConfusionMatrix(static_cast<int>(l));
    ss << cm.PrintEvaluation();
  }
  return ss.str();
}