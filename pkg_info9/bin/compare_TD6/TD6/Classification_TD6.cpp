#include "Classification_TD6.hpp"
#include "Dataset_TD6.hpp"

Classification_TD6::Classification_TD6(Dataset_TD6* dataset, int col_class) {
    m_dataset = dataset;
    m_col_class = col_class;
}

Dataset_TD6* Classification_TD6::getDataset_TD6(){
    return m_dataset;
}

int Classification_TD6::getColClass(){
    return m_col_class;
}
