#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
//  Output struct returned by runExtraction()
// ─────────────────────────────────────────────────────────────────────────────
struct ExtractionOutput {
    std::vector<cv::KeyPoint> harrisKps;
    std::vector<cv::KeyPoint> lambdaKps;
    double                    harrisTimeMs;
    double                    lambdaTimeMs;
    cv::Mat                   harrisVis;
    cv::Mat                   lambdaVis;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Public entry point — implemented in Extraction.cpp
// ─────────────────────────────────────────────────────────────────────────────
ExtractionOutput runExtraction(const cv::Mat& imgBGR,
                                double k         = 0.04,
                                int    blockSize  = 5,
                                double sigma      = 1.0,
                                double threshold  = 0.01,
                                int    nmsRadius  = 5);