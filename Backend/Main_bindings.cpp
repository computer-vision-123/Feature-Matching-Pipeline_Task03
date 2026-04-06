#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "image_utils.hpp"  // decode_image / encode_image
#include "Extraction.h"     // ExtractionOutput + runExtraction declaration
                            // (implementation lives in Extraction.cpp — compiled separately)

namespace py = pybind11;

// ─────────────────────────────────────────────────────────────────────────────
//  Small DTO exposed to Python
// ─────────────────────────────────────────────────────────────────────────────
struct PyExtractionResult {
    int       harrisCount;
    int       lambdaCount;
    double    harrisTimeMs;
    double    lambdaTimeMs;
    py::bytes harrisVis;   // PNG bytes
    py::bytes lambdaVis;   // PNG bytes
};

// ─────────────────────────────────────────────────────────────────────────────
//  Wrapper: Python bytes → cv::Mat → runExtraction → Python DTO
// ─────────────────────────────────────────────────────────────────────────────
static PyExtractionResult py_run_extraction(
        const py::bytes& imgData,
        double k         = 0.04,
        int    blockSize = 5,
        double sigma     = 1.0,
        double threshold = 0.01,
        int    nmsRadius = 5)
{
    cv::Mat img = decode_image(imgData);
    ExtractionOutput out = runExtraction(img, k, blockSize, sigma, threshold, nmsRadius);

    return {
        static_cast<int>(out.harrisKps.size()),
        static_cast<int>(out.lambdaKps.size()),
        out.harrisTimeMs,
        out.lambdaTimeMs,
        encode_image(out.harrisVis),
        encode_image(out.lambdaVis)
    };
}

// ─────────────────────────────────────────────────────────────────────────────
//  Module definition
// ─────────────────────────────────────────────────────────────────────────────
PYBIND11_MODULE(cv_backend, m) {
    m.doc() = "Harris / λ- feature extraction backend";

    py::class_<PyExtractionResult>(m, "ExtractionResult")
        .def_readonly("harris_count",   &PyExtractionResult::harrisCount)
        .def_readonly("lambda_count",   &PyExtractionResult::lambdaCount)
        .def_readonly("harris_time_ms", &PyExtractionResult::harrisTimeMs)
        .def_readonly("lambda_time_ms", &PyExtractionResult::lambdaTimeMs)
        .def_readonly("harris_vis",     &PyExtractionResult::harrisVis)
        .def_readonly("lambda_vis",     &PyExtractionResult::lambdaVis);

    m.def("run_extraction",
          &py_run_extraction,
          py::arg("img_data"),
          py::arg("k")          = 0.04,
          py::arg("block_size") = 5,
          py::arg("sigma")      = 1.0,
          py::arg("threshold")  = 0.01,
          py::arg("nms_radius") = 5
          );
}