#include "Extraction.h"   // <-- single source of truth for ExtractionOutput

#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// =============================================================================
//  FROM-SCRATCH PRIMITIVES
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────────
//  1.  Reflect-101 border index clamping
//      Maps out-of-bounds indices back into [0, len) using mirror padding.
//      e.g. len=5:  -1→1,  -2→2,  5→3,  6→2
// ─────────────────────────────────────────────────────────────────────────────
static inline int reflectIdx(int i, int len)
{
    if (i < 0)    return -i;
    if (i >= len) return 2 * len - 2 - i;
    return i;
}

// ─────────────────────────────────────────────────────────────────────────────
//  2.  Build a normalised 1-D Gaussian kernel
//      G[i] = exp( -(i - half)^2 / (2*sigma^2) )   for i in [0, ksize)
//      then divided by sum so it sums to 1.
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<float> makeGaussianKernel1D(int ksize, double sigma)
{
    if (ksize % 2 == 0) ++ksize;
    const int half = ksize / 2;
    std::vector<float> ker(ksize);
    double sum = 0.0;
    for (int i = 0; i < ksize; ++i) {
        double x = i - half;
        ker[i] = static_cast<float>(
            std::exp(-(x * x) / (2.0 * sigma * sigma)));
        sum += ker[i];
    }
    for (float& v : ker) v /= static_cast<float>(sum);
    return ker;
}

// ─────────────────────────────────────────────────────────────────────────────
//  3.  Separable Gaussian blur — fully from scratch
// ─────────────────────────────────────────────────────────────────────────────
static cv::Mat gaussianBlur(const cv::Mat& src, int ksize, double sigma)
{
    if (src.type() != CV_32F || src.channels() != 1)
        throw std::runtime_error("gaussianBlur: needs CV_32F 1-ch input");

    if (ksize % 2 == 0) ++ksize;
    const int rows = src.rows;
    const int cols = src.cols;
    const int half = ksize / 2;
    const std::vector<float> ker = makeGaussianKernel1D(ksize, sigma);

    // Pass 1: horizontal
    cv::Mat tmp(rows, cols, CV_32F);
    for (int y = 0; y < rows; ++y) {
        const float* srow = src.ptr<float>(y);
        float*       trow = tmp.ptr<float>(y);
        for (int x = 0; x < cols; ++x) {
            float acc = 0.f;
            for (int k = 0; k < ksize; ++k)
                acc += ker[k] * srow[reflectIdx(x + k - half, cols)];
            trow[x] = acc;
        }
    }

    // Pass 2: vertical
    cv::Mat dst(rows, cols, CV_32F);
    for (int y = 0; y < rows; ++y) {
        float* drow = dst.ptr<float>(y);
        for (int x = 0; x < cols; ++x) {
            float acc = 0.f;
            for (int k = 0; k < ksize; ++k)
                acc += ker[k] * tmp.ptr<float>(reflectIdx(y + k - half, rows))[x];
            drow[x] = acc;
        }
    }
    return dst;
}

// ─────────────────────────────────────────────────────────────────────────────
//  4.  Sobel 3×3 — fully from scratch
// ─────────────────────────────────────────────────────────────────────────────
static void sobel(const cv::Mat& src, cv::Mat& Ix, cv::Mat& Iy)
{
    if (src.type() != CV_32F || src.channels() != 1)
        throw std::runtime_error("sobel: needs CV_32F 1-ch input");

    const int rows = src.rows;
    const int cols = src.cols;

    static const float sm[3] = {  1.f, 2.f, 1.f };   // smooth
    static const float df[3] = { -1.f, 0.f, 1.f };   // diff

    // 1-D horizontal convolution
    auto hConv = [&](const cv::Mat& in, const float* ker) -> cv::Mat {
        cv::Mat out(rows, cols, CV_32F);
        for (int y = 0; y < rows; ++y) {
            const float* irow = in.ptr<float>(y);
            float*       orow = out.ptr<float>(y);
            for (int x = 0; x < cols; ++x)
                orow[x] = ker[0] * irow[reflectIdx(x - 1, cols)]
                         + ker[1] * irow[x]
                         + ker[2] * irow[reflectIdx(x + 1, cols)];
        }
        return out;
    };

    // 1-D vertical convolution
    auto vConv = [&](const cv::Mat& in, const float* ker) -> cv::Mat {
        cv::Mat out(rows, cols, CV_32F);
        for (int y = 0; y < rows; ++y) {
            const float* r0   = in.ptr<float>(reflectIdx(y - 1, rows));
            const float* r1   = in.ptr<float>(y);
            const float* r2   = in.ptr<float>(reflectIdx(y + 1, rows));
            float*       orow = out.ptr<float>(y);
            for (int x = 0; x < cols; ++x)
                orow[x] = ker[0] * r0[x] + ker[1] * r1[x] + ker[2] * r2[x];
        }
        return out;
    };

    // Ix = diff_x( smooth_y( src ) )
    Ix = hConv(vConv(src, sm), df);

    // Iy = diff_y( smooth_x( src ) )
    Iy = vConv(hConv(src, sm), df);
}


// =============================================================================
//  HARRIS & λ- DETECTORS  (all internal — only runExtraction() is public)
// =============================================================================

struct HarrisResult { cv::Mat response; double computeTimeMs; };
struct LambdaResult { cv::Mat response; double computeTimeMs; };

static HarrisResult harrisResponse(const cv::Mat& grayF,
                                    double k, int blockSize, double sigma)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    cv::Mat Ix, Iy;
    sobel(grayF, Ix, Iy);

    cv::Mat Ixx = Ix.mul(Ix), Iyy = Iy.mul(Iy), Ixy = Ix.mul(Iy);

    int ksize = (blockSize % 2) ? blockSize : blockSize + 1;
    cv::Mat Sxx = gaussianBlur(Ixx, ksize, sigma);
    cv::Mat Syy = gaussianBlur(Iyy, ksize, sigma);
    cv::Mat Sxy = gaussianBlur(Ixy, ksize, sigma);

    cv::Mat det   = Sxx.mul(Syy) - Sxy.mul(Sxy);
    cv::Mat trace = Sxx + Syy;
    cv::Mat R     = det - static_cast<float>(k) * trace.mul(trace);

    auto t1 = std::chrono::high_resolution_clock::now();
    return {R, std::chrono::duration<double, std::milli>(t1 - t0).count()};
}

static LambdaResult lambdaMinResponse(const cv::Mat& grayF,
                                       int blockSize, double sigma)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    cv::Mat Ix, Iy;
    sobel(grayF, Ix, Iy);

    cv::Mat Ixx = Ix.mul(Ix), Iyy = Iy.mul(Iy), Ixy = Ix.mul(Iy);

    int ksize = (blockSize % 2) ? blockSize : blockSize + 1;
    cv::Mat Sxx = gaussianBlur(Ixx, ksize, sigma);
    cv::Mat Syy = gaussianBlur(Iyy, ksize, sigma);
    cv::Mat Sxy = gaussianBlur(Ixy, ksize, sigma);

    cv::Mat diff = Sxx - Syy;
    cv::Mat disc;
    cv::sqrt(diff.mul(diff) + 4.0f * Sxy.mul(Sxy), disc);
    cv::Mat lambdaMin = 0.5f * (Sxx + Syy - disc);

    auto t1 = std::chrono::high_resolution_clock::now();
    return {lambdaMin, std::chrono::duration<double, std::milli>(t1 - t0).count()};
}

static std::vector<cv::KeyPoint> extractKeypoints(const cv::Mat& response,
                                                   float threshold, int nmsRadius)
{
    std::vector<cv::KeyPoint> kps;
    const int rows = response.rows, cols = response.cols;

    cv::Mat dilated;
    cv::dilate(response, dilated,
               cv::getStructuringElement(cv::MORPH_RECT,
                   cv::Size(2 * nmsRadius + 1, 2 * nmsRadius + 1)));

    for (int y = nmsRadius; y < rows - nmsRadius; ++y) {
        const float* Rrow = response.ptr<float>(y);
        const float* Drow = dilated.ptr<float>(y);
        for (int x = nmsRadius; x < cols - nmsRadius; ++x) {
            float v = Rrow[x];
            if (v >= threshold && std::abs(v - Drow[x]) < 1e-6f)
                kps.emplace_back(cv::KeyPoint(
                    static_cast<float>(x), static_cast<float>(y),
                    static_cast<float>(2 * nmsRadius), -1.f, v));
        }
    }
    return kps;
}


// =============================================================================
//  PUBLIC API
// =============================================================================

ExtractionOutput runExtraction(const cv::Mat& imgBGR,
                                double k, int blockSize,
                                double sigma, double threshold, int nmsRadius)
{
    cv::Mat gray, grayF;
    if (imgBGR.channels() == 3)
        cv::cvtColor(imgBGR, gray, cv::COLOR_BGR2GRAY);
    else
        gray = imgBGR.clone();
    gray.convertTo(grayF, CV_32F, 1.0 / 255.0);

    // Harris
    auto   hRes   = harrisResponse(grayF, k, blockSize, sigma);
    double hMax;  cv::minMaxLoc(hRes.response, nullptr, &hMax);
    auto   hKps   = extractKeypoints(hRes.response,
                                      static_cast<float>(threshold * hMax), nmsRadius);

    // λ-
    auto   lRes   = lambdaMinResponse(grayF, blockSize, sigma);
    double lMax;  cv::minMaxLoc(lRes.response, nullptr, &lMax);
    auto   lKps   = extractKeypoints(lRes.response,
                                      static_cast<float>(threshold * lMax), nmsRadius);

    // Visualisation
    cv::Mat harrisVis, lambdaVis;
    cv::drawKeypoints(imgBGR, hKps, harrisVis, cv::Scalar(0, 0, 255),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::drawKeypoints(imgBGR, lKps, lambdaVis, cv::Scalar(0, 255, 0),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    return {hKps, lKps,
            hRes.computeTimeMs, lRes.computeTimeMs,
            harrisVis, lambdaVis};
}