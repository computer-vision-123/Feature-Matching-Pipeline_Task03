#pragma once

#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

// Decode PNG bytes coming from Python into an OpenCV BGR matrix
inline cv::Mat decode_image(const py::bytes &data) {
    std::string raw = data;
    std::vector<uchar> buf(raw.begin(), raw.end());
    cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
    if (img.empty()) throw std::runtime_error("Failed to decode image");
    return img;
}

// Encode an OpenCV matrix back into PNG bytes to send to Python
inline py::bytes encode_image(const cv::Mat &img) {
    std::vector<uchar> buf;
    cv::imencode(".png", img, buf);
    return py::bytes(reinterpret_cast<const char*>(buf.data()), buf.size());
}
