#pragma once

#include "Extraction.h"

#include <opencv2/opencv.hpp>
#include <vector>

// =============================================================================
//  DescriptionOutput
//  Returned by runDescription() – carries everything the frontend needs.
// =============================================================================
struct DescriptionOutput {
    // ── Keypoints that survived the descriptor stage ──────────────────────
    std::vector<cv::KeyPoint> harrisKps;   // same ordering as descriptors below
    std::vector<cv::KeyPoint> lambdaKps;

    // ── Raw 128-D SIFT descriptors (one row = one descriptor) ─────────────
    cv::Mat harrisDesc;    // CV_32F  N×128
    cv::Mat lambdaDesc;    // CV_32F  M×128

    // ── Annotated visualisation images (PNG-encodable) ────────────────────
    cv::Mat harrisVis;
    cv::Mat lambdaVis;

    // ── Timing ────────────────────────────────────────────────────────────
    double harrisTimeMs;
    double lambdaTimeMs;
};

// =============================================================================
//  Public entry point
//
//  imgBGR        – original colour image
//  extraction    – result of runExtraction() (keypoints already found)
//  numOctaves    – how many octave levels to build for the Gaussian pyramid
//                  (3 is a safe default; more = richer scale coverage)
//  sigmaBase     – base blur for the pyramid (matches SIFT paper: 1.6)
// =============================================================================
DescriptionOutput runDescription(const cv::Mat&      imgBGR,
                                 const ExtractionOutput& extraction,
                                 int    numOctaves = 3,
                                 double sigmaBase  = 1.6);