#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <ANN/ANN.h>

#include "helpers.h"
#include "classification/KnnClassification.hpp"
#include "classification/Dataset.hpp"

namespace py = pybind11;

PYBIND11_MODULE(info9, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: info9

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");
    m.def("addm", &addm);
    m.def("add_in_first", &add_in_first);
    m.def("read_hdf5_dataset", &read_hdf5_dataset);
    
    py::class_<Dataset, std::shared_ptr<Dataset> >(m, "Dataset")
	  .def(py::init<const std::string&>(), py::arg("csv"))
	  .def("show", &Dataset::Show, py::arg("show_data"));

    py::class_<KnnClassification>(m, "KnnClassification")
	  .def(py::init<int, std::shared_ptr<Dataset>, int>(), py::arg("k"), py::arg("dataset"), py::arg("label_col"))
	  .def("estimate", &KnnClassification::Estimate, py::arg("x"), py::arg("threshold"))
	  .def("print_kd_stats", &KnnClassification::print_kd_stats);
	  
    m.def("annClose", &annClose);
}
