
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include "Dataset_TD6.hpp"


int Dataset_TD6::getNbrSamples() {
	return m_nsamples;
}

int Dataset_TD6::getDim() {
	return m_dim;
}

Dataset_TD6::~Dataset_TD6() {
	// All attributes have destructors
}

void Dataset_TD6::Show(bool verbose) {
	std::cout<<"Dataset_TD6 with "<<m_nsamples<<" samples, and "<<m_dim<<" dimensions."<<std::endl;
	if (verbose) {
		for (int i=0; i<m_nsamples; i++) {
			for (int j=0; j<m_dim; j++) {
				std::cout<<m_instances[i][j]<<" ";
			}
			std::cout<<std::endl;		
		}
	}
}

Dataset_TD6::Dataset_TD6(const char* file) {
	m_nsamples = 0;
	m_dim = -1;

	std::ifstream fin(file);
	
	if (fin.fail()) {
		std::cout<<"Cannot read from file "<<file<<" !"<<std::endl;
		exit(1);
	}
	
	std::vector<double> row; 
    std::string line, word, temp; 

	while (getline(fin, line)) {
		row.clear();
        std::stringstream s(line);
        int ncols = 0;
        while (getline(s, word, ',')) { 
            // add all the column data 
            // of a row to a vector 
            double val = std::atof(word.c_str());
            row.push_back(val);
            ncols++;
        }
        m_instances.push_back(row);
        if (m_dim==-1) m_dim = ncols;
        else if (m_dim!=ncols) {
        	std::cerr << "ERROR, inconsistent dataset" << std::endl;
        	exit(-1);
        }
		m_nsamples ++;
	}
	fin.close();
}

Dataset_TD6::Dataset_TD6(const std::vector<std::vector<double> > &vector_of_vector) {
	m_instances = vector_of_vector;
	m_dim = vector_of_vector[0].size();
	m_nsamples = vector_of_vector.size();
}

std::vector<double> Dataset_TD6::getInstance(int i) {
	return m_instances[i];
}
