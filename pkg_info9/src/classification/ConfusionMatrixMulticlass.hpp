
#pragma once
#include <iostream>
#include <vector>
#include "ConfusionMatrix.hpp"
/**
  The ConfusionMatrix class .
*/
class ConfusionMatrixMulticlass {
public:
  ConfusionMatrixMulticlass(std::vector<std::string> labels)
  : m_labels{std::move(labels)}
  {
    int const n_labels = m_labels.size();
    m_confusion_matrix.resize(n_labels, std::vector<int>(n_labels, 0));
  };
  void AddPrediction(int true_label, int predicted_label);

  std::string PrintMatrix();
  std::string PrintEvaluation();

  ConfusionMatrix OneVsAllConfusionMatrix(int label_one);
private:
  std::vector<std::string> m_labels;
  std::vector<std::vector<int> > m_confusion_matrix;
};
