#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "image_utils.hpp"
#include "Extraction.h"
#include "Description.h"
#include "Matching.h"

namespace py = pybind11;

// =============================================================================
//  DTOs
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
    int    harrisCount;
    int    lambdaCount;
    double harrisTimeMs;
    double lambdaTimeMs;
    py::bytes harrisVis;
    py::bytes lambdaVis;
    py::bytes harrisDescBytes;
    py::bytes lambdaDescBytes;
    std::vector<std::pair<float, float>> harrisKeypoints;
    std::vector<std::pair<float, float>> lambdaKeypoints;
};

struct PyMatchingOutput {
    // Harris
    py::bytes harrisDescBytesA;
    py::bytes harrisDescBytesB;
    std::vector<std::pair<float, float>> harrisKptsA;
    std::vector<std::pair<float, float>> harrisKptsB;
    int harrisCountA;
    int harrisCountB;
    // Lambda
    py::bytes lambdaDescBytesA;
    py::bytes lambdaDescBytesB;
    std::vector<std::pair<float, float>> lambdaKptsA;
    std::vector<std::pair<float, float>> lambdaKptsB;
    int lambdaCountA;
    int lambdaCountB;
};

// =============================================================================
//  Wrappers
// =============================================================================

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

static PyDescriptionResult py_run_description(
        const py::bytes& imgData,
        double k          = 0.04,
        int    blockSize  = 5,
        double sigma      = 1.0,
        double threshold  = 0.01,
        int    nmsRadius  = 5,
        int    numOctaves = 3,
        double sigmaBase  = 1.6)
{
    cv::Mat img = decode_image(imgData);

    ExtractionOutput ext   = runExtraction(img, k, blockSize, sigma, threshold, nmsRadius);
    DescriptionOutput desc = runDescription(img, ext, numOctaves, sigmaBase);

    auto matToBytes = [](const cv::Mat& m) -> py::bytes {
        if (m.empty()) return py::bytes("", 0);
        cv::Mat cont;
        if (m.isContinuous()) cont = m;
        else                  m.copyTo(cont);
        return py::bytes(
            reinterpret_cast<const char*>(cont.ptr<float>(0)),
            static_cast<size_t>(cont.total()) * sizeof(float));
    };

    std::vector<std::pair<float, float>> harrisKpts, lambdaKpts;
    harrisKpts.reserve(desc.harrisKps.size());
    lambdaKpts.reserve(desc.lambdaKps.size());
    for (const auto& kp : desc.harrisKps) harrisKpts.push_back({ kp.pt.x, kp.pt.y });
    for (const auto& kp : desc.lambdaKps) lambdaKpts.push_back({ kp.pt.x, kp.pt.y });

    return {
        static_cast<int>(desc.harrisKps.size()),
        static_cast<int>(desc.lambdaKps.size()),
        desc.harrisTimeMs,
        desc.lambdaTimeMs,
        encode_image(desc.harrisVis),
        encode_image(desc.lambdaVis),
        matToBytes(desc.harrisDesc),
        matToBytes(desc.lambdaDesc),
        std::move(harrisKpts),
        std::move(lambdaKpts)
    };
}

static PyMatchingOutput py_run_matching(const PyDescriptionResult& a,
                                         const PyDescriptionResult& b)
{
    auto toMat = [](const py::bytes& raw, int count) -> cv::Mat {
        if (count == 0) return cv::Mat();
        std::string s = raw;
        cv::Mat m(count, 128, CV_32F);
        std::memcpy(m.ptr<float>(0), s.data(), s.size());
        return m;
    };

    auto toKpts = [](const std::vector<std::pair<float, float>>& v) {
        std::vector<cv::Point2f> out;
        out.reserve(v.size());
        for (const auto& p : v) out.emplace_back(p.first, p.second);
        return out;
    };

    auto fromKpts = [](const std::vector<cv::Point2f>& v) {
        std::vector<std::pair<float, float>> out;
        out.reserve(v.size());
        for (const auto& p : v) out.push_back({ p.x, p.y });
        return out;
    };

    auto matToBytes = [](const cv::Mat& m) -> py::bytes {
        if (m.empty()) return py::bytes("", 0);
        cv::Mat cont;
        if (m.isContinuous()) cont = m;
        else                  m.copyTo(cont);
        return py::bytes(
            reinterpret_cast<const char*>(cont.ptr<float>(0)),
            static_cast<size_t>(cont.total()) * sizeof(float));
    };

    MatchingInput harris {
        toMat(a.harrisDescBytes, a.harrisCount),
        toMat(b.harrisDescBytes, b.harrisCount),
        toKpts(a.harrisKeypoints),
        toKpts(b.harrisKeypoints)
    };

    MatchingInput lambda {
        toMat(a.lambdaDescBytes, a.lambdaCount),
        toMat(b.lambdaDescBytes, b.lambdaCount),
        toKpts(a.lambdaKeypoints),
        toKpts(b.lambdaKeypoints)
    };

    MatchingOutput out = runMatching(harris, lambda);

    return {
        matToBytes(out.harris.descA), matToBytes(out.harris.descB),
        fromKpts(out.harris.kptsA),   fromKpts(out.harris.kptsB),
        static_cast<int>(out.harris.kptsA.size()),
        static_cast<int>(out.harris.kptsB.size()),
        matToBytes(out.lambda.descA), matToBytes(out.lambda.descB),
        fromKpts(out.lambda.kptsA),   fromKpts(out.lambda.kptsB),
        static_cast<int>(out.lambda.kptsA.size()),
        static_cast<int>(out.lambda.kptsB.size()),
    };
}

// =============================================================================
//  Module
// =============================================================================
PYBIND11_MODULE(cv_backend, m) {
    m.doc() = "Harris / λ- feature extraction + SIFT description + matching backend";

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

    py::class_<PyDescriptionResult>(m, "DescriptionResult")
        .def_readonly("harris_count",      &PyDescriptionResult::harrisCount)
        .def_readonly("lambda_count",      &PyDescriptionResult::lambdaCount)
        .def_readonly("harris_time_ms",    &PyDescriptionResult::harrisTimeMs)
        .def_readonly("lambda_time_ms",    &PyDescriptionResult::lambdaTimeMs)
        .def_readonly("harris_vis",        &PyDescriptionResult::harrisVis)
        .def_readonly("lambda_vis",        &PyDescriptionResult::lambdaVis)
        .def_readonly("harris_desc_bytes", &PyDescriptionResult::harrisDescBytes)
        .def_readonly("lambda_desc_bytes", &PyDescriptionResult::lambdaDescBytes)
        .def_readonly("harris_keypoints",  &PyDescriptionResult::harrisKeypoints)
        .def_readonly("lambda_keypoints",  &PyDescriptionResult::lambdaKeypoints);

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

    py::class_<PyMatchingOutput>(m, "MatchingOutput")
        .def_readonly("harris_desc_bytes_a", &PyMatchingOutput::harrisDescBytesA)
        .def_readonly("harris_desc_bytes_b", &PyMatchingOutput::harrisDescBytesB)
        .def_readonly("harris_kpts_a",       &PyMatchingOutput::harrisKptsA)
        .def_readonly("harris_kpts_b",       &PyMatchingOutput::harrisKptsB)
        .def_readonly("harris_count_a",      &PyMatchingOutput::harrisCountA)
        .def_readonly("harris_count_b",      &PyMatchingOutput::harrisCountB)
        .def_readonly("lambda_desc_bytes_a", &PyMatchingOutput::lambdaDescBytesA)
        .def_readonly("lambda_desc_bytes_b", &PyMatchingOutput::lambdaDescBytesB)
        .def_readonly("lambda_kpts_a",       &PyMatchingOutput::lambdaKptsA)
        .def_readonly("lambda_kpts_b",       &PyMatchingOutput::lambdaKptsB)
        .def_readonly("lambda_count_a",      &PyMatchingOutput::lambdaCountA)
        .def_readonly("lambda_count_b",      &PyMatchingOutput::lambdaCountB);

    m.def("run_matching",
          &py_run_matching,
          py::arg("result_a"),
          py::arg("result_b"));
}