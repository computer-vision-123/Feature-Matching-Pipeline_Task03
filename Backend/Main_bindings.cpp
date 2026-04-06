#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "image_utils.hpp"   // decode_image / encode_image
#include "Extraction.h"      // ExtractionOutput  + runExtraction()
#include "Description.h"     // DescriptionOutput + runDescription()

namespace py = pybind11;

// =============================================================================
//  DTOs exposed to Python
// =============================================================================

struct PyExtractionResult {
    int       harrisCount;
    int       lambdaCount;
    double    harrisTimeMs;
    double    lambdaTimeMs;
    py::bytes harrisVis;
    py::bytes lambdaVis;
};

struct PyDescriptionResult {
    // Keypoint counts
    int    harrisCount;
    int    lambdaCount;

    // Timing
    double harrisTimeMs;
    double lambdaTimeMs;

    // Visualisation PNGs
    py::bytes harrisVis;
    py::bytes lambdaVis;

    // Raw descriptors as flat byte blobs (row-major float32, N×128)
    // Python side can reconstruct with:
    //   np.frombuffer(result.harris_desc_bytes, dtype=np.float32).reshape(-1, 128)
    py::bytes harrisDescBytes;
    py::bytes lambdaDescBytes;
};

// =============================================================================
//  Wrappers
// =============================================================================

// ── Extraction ───────────────────────────────────────────────────────────────
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

// ── Description (takes same image + re-runs extraction internally) ────────────
// We re-run extraction here so Python only needs to pass image bytes once.
// The extraction params must match whatever the user set in the UI.
static PyDescriptionResult py_run_description(
        const py::bytes& imgData,
        // Extraction params
        double k         = 0.04,
        int    blockSize = 5,
        double sigma     = 1.0,
        double threshold = 0.01,
        int    nmsRadius = 5,
        // Description params
        int    numOctaves = 3,
        double sigmaBase  = 1.6)
{
    cv::Mat img = decode_image(imgData);

    // Step 1: Extract keypoints
    ExtractionOutput ext = runExtraction(img, k, blockSize, sigma, threshold, nmsRadius);

    // Step 2: Compute SIFT descriptors
    DescriptionOutput desc = runDescription(img, ext, numOctaves, sigmaBase);

    // ── Pack descriptor matrices as raw float32 bytes ─────────────────────
    auto matToBytes = [](const cv::Mat& m) -> py::bytes {
        if (m.empty()) return py::bytes("", 0);
        // Ensure the matrix is contiguous CV_32F
        cv::Mat cont;
        if (m.isContinuous()) cont = m;
        else                  m.copyTo(cont);
        return py::bytes(
            reinterpret_cast<const char*>(cont.ptr<float>(0)),
            static_cast<size_t>(cont.total()) * sizeof(float));
    };

    return {
        static_cast<int>(desc.harrisKps.size()),
        static_cast<int>(desc.lambdaKps.size()),
        desc.harrisTimeMs,
        desc.lambdaTimeMs,
        encode_image(desc.harrisVis),
        encode_image(desc.lambdaVis),
        matToBytes(desc.harrisDesc),
        matToBytes(desc.lambdaDesc)
    };
}

// =============================================================================
//  Module
// =============================================================================
PYBIND11_MODULE(cv_backend, m) {
    m.doc() = "Harris / λ- feature extraction + SIFT description backend";

    // ── ExtractionResult ──────────────────────────────────────────────────
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
          py::arg("nms_radius") = 5);

    // ── DescriptionResult ─────────────────────────────────────────────────
    py::class_<PyDescriptionResult>(m, "DescriptionResult")
        .def_readonly("harris_count",      &PyDescriptionResult::harrisCount)
        .def_readonly("lambda_count",      &PyDescriptionResult::lambdaCount)
        .def_readonly("harris_time_ms",    &PyDescriptionResult::harrisTimeMs)
        .def_readonly("lambda_time_ms",    &PyDescriptionResult::lambdaTimeMs)
        .def_readonly("harris_vis",        &PyDescriptionResult::harrisVis)
        .def_readonly("lambda_vis",        &PyDescriptionResult::lambdaVis)
        .def_readonly("harris_desc_bytes", &PyDescriptionResult::harrisDescBytes)
        .def_readonly("lambda_desc_bytes", &PyDescriptionResult::lambdaDescBytes);

    m.def("run_description",
          &py_run_description,
          py::arg("img_data"),
          py::arg("k")           = 0.04,
          py::arg("block_size")  = 5,
          py::arg("sigma")       = 1.0,
          py::arg("threshold")   = 0.01,
          py::arg("nms_radius")  = 5,
          py::arg("num_octaves") = 3,
          py::arg("sigma_base")  = 1.6);
}