#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "helpers.h"
#include "classification/ConfusionMatrix.hpp"
#include "classification/ConfusionMatrixMulticlass.hpp"
#include "classification/KnnClassificationMulticlass.hpp"
#include "classification/KnnClassificationBinary.hpp"
#include "classification/KnnClassificationMulticlass.hpp"
#include "classification/RandomProjection.hpp"
#include "classification/Dataset.hpp"

namespace py = pybind11;

PYBIND11_MODULE(info9, m) {
    m.doc() = R"pbdoc(
        Project 9 - GDPR anonymization
        -----------------------

        .. currentmodule:: info9

        .. autosummary::
           :toctree: _generate
    )pbdoc";

    m.def("read_hdf5_dataset", &read_hdf5_dataset);
    
    py::class_<Dataset, std::shared_ptr<Dataset> >(m, "Dataset")
	  .def(py::init<RMatrixXd, Eigen::VectorXi>(), py::arg("datapoints"), py::arg("true_labels"))
	  .def("show", &Dataset::Show, py::arg("show_data"));

    py::class_<KnnClassificationBinary>(m, "KnnClassificationBinary")
	  .def(py::init<int, std::shared_ptr<Dataset>, double>(), py::arg("k"), py::arg("dataset"), py::arg("threshold"))
	  .def("estimate", &KnnClassificationBinary::Estimate)
      .def("estimate_all", &KnnClassificationBinary::EstimateAll, py::arg("dataset"))
	  .def("print_kd_stats", &KnnClassificationBinary::print_kd_stats);

    py::class_<KnnClassificationMulticlass>(m, "KnnClassificationMulticlass")
      .def(py::init<int, std::shared_ptr<Dataset>, std::vector<std::string> >(), py::arg("k"), py::arg("dataset"), py::arg("labels"))
      .def("estimate", &KnnClassificationMulticlass::Estimate)
      .def("estimate_all", &KnnClassificationMulticlass::EstimateAll, py::arg("dataset"))
      .def("print_kd_stats", &KnnClassificationMulticlass::print_kd_stats);

    py::class_<ConfusionMatrix>(m, "ConfusionMatrix")
        .def(py::init<>())
        .def("AddPrediction", &ConfusionMatrix::AddPrediction, py::arg("true_label"), py::arg("predicted_label"))
        .def("PrintEvaluation", &ConfusionMatrix::PrintEvaluation);


    py::class_<ConfusionMatrixMulticlass>(m, "ConfusionMatrixMulticlass")
        .def(py::init<std::vector<std::string> >())
        .def("AddPrediction", &ConfusionMatrixMulticlass::AddPrediction, py::arg("true_label"), py::arg("predicted_label"))
        .def("PrintEvaluation", &ConfusionMatrixMulticlass::PrintEvaluation)
        .def("PrintMatrix", &ConfusionMatrixMulticlass::PrintMatrix)
        .def("OneVsAllConfusionMatrix", &ConfusionMatrixMulticlass::OneVsAllConfusionMatrix, py::arg("label_id"));


    py::class_<RandomProjection>(m, "RandomProjection")
        .def(py::init<int, int, std::string>(), py::arg("original_dimension"), py::arg("projection_dim"), py::arg("type_sample"))
        .def("project", &RandomProjection::Project, py::arg("datapoints"));
        .def("")

    m.def("annClose", &annClose);

}
