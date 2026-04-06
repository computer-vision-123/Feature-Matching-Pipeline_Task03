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
    MatchingInput harris;
    MatchingInput lambda;
};

MatchingOutput runMatching(const MatchingInput& harris, const MatchingInput& lambda);