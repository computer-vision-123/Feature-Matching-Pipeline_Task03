#include "Matching.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

// =============================================================================
//  Helpers
// =============================================================================

// Compute SSD between two descriptor rows (pointers to 128 floats)
static inline float computeSSD(const float* a, const float* b, int len)
{
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

// Compute NCC between two descriptor rows (pointers to 128 floats)
// NCC = dot(a, b) / (||a|| * ||b||)   — ranges from -1 to +1
static inline float computeNCC(const float* a, const float* b, int len)
{
    float dot = 0.0f, normA = 0.0f, normB = 0.0f;
    for (int i = 0; i < len; ++i) {
        dot   += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    float denom = std::sqrt(normA * normB);
    if (denom < 1e-10f) return 0.0f;
    return dot / denom;
}

// =============================================================================
//  SSD matching with Lowe's ratio test
// =============================================================================
static std::vector<cv::DMatch> matchSSD(const cv::Mat& descA,
                                         const cv::Mat& descB,
                                         float ratioThresh = 0.75f)
{
    std::vector<cv::DMatch> matches;
    if (descA.empty() || descB.empty()) return matches;

    const int N   = descA.rows;
    const int M   = descB.rows;
    const int dim = descA.cols;  // 128

    for (int i = 0; i < N; ++i) {
        const float* rowA = descA.ptr<float>(i);

        float best1 = std::numeric_limits<float>::max();
        float best2 = std::numeric_limits<float>::max();
        int   bestIdx = -1;

        for (int j = 0; j < M; ++j) {
            float ssd = computeSSD(rowA, descB.ptr<float>(j), dim);

            if (ssd < best1) {
                best2   = best1;
                best1   = ssd;
                bestIdx = j;
            } else if (ssd < best2) {
                best2 = ssd;
            }
        }

        // Ratio test: reject ambiguous matches
        if (best2 > 1e-10f && (best1 / best2) < ratioThresh) {
            matches.emplace_back(i, bestIdx, best1);
        }
    }
    return matches;
}

// =============================================================================
//  NCC matching with ratio test
// =============================================================================
static std::vector<cv::DMatch> matchNCC(const cv::Mat& descA,
                                         const cv::Mat& descB,
                                         float ratioThresh = 0.75f)
{
    std::vector<cv::DMatch> matches;
    if (descA.empty() || descB.empty()) return matches;

    const int N   = descA.rows;
    const int M   = descB.rows;
    const int dim = descA.cols;  // 128

    for (int i = 0; i < N; ++i) {
        const float* rowA = descA.ptr<float>(i);

        float best1 = -2.0f;   // best NCC  (higher = better)
        float best2 = -2.0f;   // second-best NCC
        int   bestIdx = -1;

        for (int j = 0; j < M; ++j) {
            float ncc = computeNCC(rowA, descB.ptr<float>(j), dim);

            if (ncc > best1) {
                best2   = best1;
                best1   = ncc;
                bestIdx = j;
            } else if (ncc > best2) {
                best2 = ncc;
            }
        }

        // Ratio test on distance-like scores: convert NCC → distance = 1 - NCC
        // keep if (1 - best1) / (1 - best2) < ratio  ⟹  best1 is much better
        float d1 = 1.0f - best1;
        float d2 = 1.0f - best2;
        if (d2 > 1e-10f && (d1 / d2) < ratioThresh) {
            matches.emplace_back(i, bestIdx, d1);
        }
    }
    return matches;
}

// =============================================================================
//  Build a filtered MatchingInput keeping only the matched keypoints/descriptors
// =============================================================================
static MatchingInput buildMatchedInput(const MatchingInput& input,
                                        const std::vector<cv::DMatch>& matches)
{
    MatchingInput out;
    if (matches.empty()) return out;

    const int dim = input.descA.cols;  // 128
    out.descA = cv::Mat(static_cast<int>(matches.size()), dim, CV_32F);
    out.descB = cv::Mat(static_cast<int>(matches.size()), dim, CV_32F);
    out.kptsA.reserve(matches.size());
    out.kptsB.reserve(matches.size());

    for (int i = 0; i < static_cast<int>(matches.size()); ++i) {
        const auto& m = matches[i];
        input.descA.row(m.queryIdx).copyTo(out.descA.row(i));
        input.descB.row(m.trainIdx).copyTo(out.descB.row(i));
        out.kptsA.push_back(input.kptsA[m.queryIdx]);
        out.kptsB.push_back(input.kptsB[m.trainIdx]);
    }
    return out;
}

// =============================================================================
//  Public API
// =============================================================================

MatchingOutput runMatching(const MatchingInput& harris, const MatchingInput& lambda)
{
    using Clock = std::chrono::high_resolution_clock;
    MatchingOutput out;

    // ── Harris SSD ──────────────────────────────────────────────────────────
    auto t0 = Clock::now();
    out.harrisSsdMatches = matchSSD(harris.descA, harris.descB);
    auto t1 = Clock::now();
    out.harrisSsdTimeMs  = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ── Harris NCC ──────────────────────────────────────────────────────────
    t0 = Clock::now();
    out.harrisNccMatches = matchNCC(harris.descA, harris.descB);
    t1 = Clock::now();
    out.harrisNccTimeMs  = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ── Lambda SSD ──────────────────────────────────────────────────────────
    t0 = Clock::now();
    out.lambdaSsdMatches = matchSSD(lambda.descA, lambda.descB);
    t1 = Clock::now();
    out.lambdaSsdTimeMs  = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ── Lambda NCC ──────────────────────────────────────────────────────────
    t0 = Clock::now();
    out.lambdaNccMatches = matchNCC(lambda.descA, lambda.descB);
    t1 = Clock::now();
    out.lambdaNccTimeMs  = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ── Build filtered keypoint/descriptor sets ───────────────────────────
    // SSD-matched
    out.harris = buildMatchedInput(harris, out.harrisSsdMatches);
    out.lambda = buildMatchedInput(lambda, out.lambdaSsdMatches);
    // NCC-matched
    out.harrisNcc = buildMatchedInput(harris, out.harrisNccMatches);
    out.lambdaNcc = buildMatchedInput(lambda, out.lambdaNccMatches);

    // ── Report to stdout ────────────────────────────────────────────────────
    std::cout << "\n=== Feature Matching Results ===\n"
              << "Harris SSD : " << out.harrisSsdMatches.size() << " matches in "
              << out.harrisSsdTimeMs << " ms\n"
              << "Harris NCC : " << out.harrisNccMatches.size() << " matches in "
              << out.harrisNccTimeMs << " ms\n"
              << "Lambda SSD : " << out.lambdaSsdMatches.size() << " matches in "
              << out.lambdaSsdTimeMs << " ms\n"
              << "Lambda NCC : " << out.lambdaNccMatches.size() << " matches in "
              << out.lambdaNccTimeMs << " ms\n"
              << "================================\n" << std::endl;

    return out;
}