#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

struct MatchingInput {
    cv::Mat                  descA;    // CV_32F  (N x 128)
    cv::Mat                  descB;    // CV_32F  (M x 128)
    std::vector<cv::Point2f> kptsA;    // pixel (x, y) for each row in descA
    std::vector<cv::Point2f> kptsB;    // pixel (x, y) for each row in descB
};

struct MatchingOutput {
    // SSD-matched keypoints/descriptors (primary)
    MatchingInput harris;
    MatchingInput lambda;

    // NCC-matched keypoints/descriptors
    MatchingInput harrisNcc;
    MatchingInput lambdaNcc;

    // SSD matching timing (ms)
    double harrisSsdTimeMs = 0.0;
    double lambdaSsdTimeMs = 0.0;

    // NCC matching timing (ms)
    double harrisNccTimeMs = 0.0;
    double lambdaNccTimeMs = 0.0;

    // SSD matches (index pairs + distance)
    std::vector<cv::DMatch> harrisSsdMatches;
    std::vector<cv::DMatch> lambdaSsdMatches;

    // NCC matches (index pairs + distance)
    std::vector<cv::DMatch> harrisNccMatches;
    std::vector<cv::DMatch> lambdaNccMatches;
};

MatchingOutput runMatching(const MatchingInput& harris, const MatchingInput& lambda);