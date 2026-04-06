#include "Description.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <chrono>

// =============================================================================
//  Constants  (mirror the reference SIFT repo)
// =============================================================================
namespace {

constexpr int    SCALES_PER_OCT    = 3;
constexpr int    IMAGES_PER_OCT    = SCALES_PER_OCT + 3;   // s+3 = 6
constexpr double SIGMA_BASE        = 1.6;
constexpr double ASSUMED_BLUR      = 0.5;

// Orientation histogram
constexpr int    ORI_BINS          = 36;
constexpr double ORI_PEAK_RATIO    = 0.80;
constexpr double ORI_SCALE_FACTOR  = 1.5;   // radius = factor * scale
constexpr double ORI_RADIUS_FACTOR = 3.0;

// Descriptor
constexpr int    DESC_WINDOW_WIDTH = 4;      // 4×4 spatial bins
constexpr int    DESC_HIST_BINS    = 8;      // 8 orientation bins
constexpr double DESC_SCALE_MUL   = 3.0;
constexpr double DESC_MAX_VAL      = 0.2;    // clamp threshold
constexpr double EPS               = 1e-7;
constexpr double PI                = 3.14159265358979323846;

// Scale mapping: keypoint pixel diameter → octave
constexpr double BASE_SCALE        = 4.0;   // a kpt.size ≈ BASE_SCALE
                                             // maps to octave 0

} // anonymous namespace

// =============================================================================
//  Helpers
// =============================================================================

static inline double deg2rad(double d) { return d * PI / 180.0; }
static inline double rad2deg(double r) { return r * 180.0 / PI; }

// Safe pixel accessor with reflect-101 border
static inline double safeGet(const cv::Mat& m, int r, int c)
{
    r = std::max(0, std::min(m.rows - 1, r));
    c = std::max(0, std::min(m.cols - 1, c));
    return m.at<double>(r, c);
}

// =============================================================================
//  1.  Gaussian pyramid
//      gauss[oct][img]  — each entry is CV_64F, single channel
// =============================================================================
static std::vector<std::vector<cv::Mat>>
buildGaussianPyramid(const cv::Mat& grayF64, int numOctaves)
{
    // Per-level incremental sigma (same formula as reference repo)
    const double k = std::pow(2.0, 1.0 / SCALES_PER_OCT);
    std::array<double, IMAGES_PER_OCT> kernelSigmas{};
    kernelSigmas[0] = SIGMA_BASE;
    double prev = SIGMA_BASE;
    for (int i = 1; i < IMAGES_PER_OCT; ++i) {
        double now   = prev * k;
        kernelSigmas[i] = std::sqrt(now * now - prev * prev);
        prev = now;
    }

    // Base image: upsample × 2 then blur to remove assumed camera blur
    cv::Mat interpolated;
    cv::resize(grayF64, interpolated,
               cv::Size(grayF64.cols * 2, grayF64.rows * 2),
               0, 0, cv::INTER_LINEAR);
    double initDiff = std::max(
        std::sqrt(SIGMA_BASE * SIGMA_BASE - 4.0 * ASSUMED_BLUR * ASSUMED_BLUR),
        0.1);
    cv::Mat base;
    cv::GaussianBlur(interpolated, base, cv::Size(0, 0), initDiff, initDiff);

    std::vector<std::vector<cv::Mat>> pyr;
    pyr.reserve(numOctaves);

    cv::Mat octBase = base;
    for (int oct = 0; oct < numOctaves; ++oct) {
        std::vector<cv::Mat> octImgs(IMAGES_PER_OCT);
        octImgs[0] = octBase;
        for (int i = 1; i < IMAGES_PER_OCT; ++i) {
            cv::GaussianBlur(octImgs[i - 1], octImgs[i],
                             cv::Size(0, 0),
                             kernelSigmas[i], kernelSigmas[i]);
        }
        // Next octave base = downsampled third-from-last image (index s = SCALES_PER_OCT)
        cv::resize(octImgs[SCALES_PER_OCT],
                   octBase,
                   cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
        pyr.push_back(std::move(octImgs));
    }
    return pyr;
}

// =============================================================================
//  2.  Map a keypoint's pixel-space size to (octave, layer)
//
//      kpt.size from extractKeypoints() = 2 * nmsRadius  (a pixel diameter).
//      We treat it as a scale hint: larger size → higher octave.
//
//      octave = clamp( round( log2(size / BASE_SCALE) ), 0, numOctaves-1 )
//      layer  = middle layer index (SCALES_PER_OCT / 2)  → stable, well-blurred
// =============================================================================
static void assignPyramidLevel(cv::KeyPoint& kpt, int numOctaves,
                                int& outOct, int& outLayer)
{
    double size  = std::max(static_cast<double>(kpt.size), 1.0);
    int    oct   = static_cast<int>(std::round(std::log2(size / BASE_SCALE)));
    oct          = std::max(0, std::min(numOctaves - 1, oct));
    int    layer = SCALES_PER_OCT / 2;   // e.g. layer 1 out of 0,1,2

    outOct   = oct;
    outLayer = layer;

    // Pack into kpt.octave:  bits 0-7 = oct,  bits 8-15 = layer
    // (matches the unpacking in the descriptor loop below)
    kpt.octave = oct | (layer << 8);
}

// =============================================================================
//  3.  Orientation assignment  (36-bin histogram, Gaussian-weighted)
//      Returns zero or more keypoints, each with a dominant orientation.
// =============================================================================
static std::vector<cv::KeyPoint>
assignOrientations(const cv::KeyPoint&                    kpt,
                   int                                    oct,
                   int                                    layer,
                   const std::vector<std::vector<cv::Mat>>& pyr)
{
    const cv::Mat& img = pyr[oct][layer];
    const cv::Size sz  = img.size();

    // Scale of this keypoint inside the octave coordinate frame
    // kpt.pt and kpt.size are in the ORIGINAL image space (halved once
    // because we upsampled by 2 in the pyramid base, then downsampled oct times).
    // The pyramid base was upsampled ×2, so a point at pixel p in the original
    // is at 2p in octave-0.  Each subsequent octave halves: 2p / 2^oct = p * 2^(1-oct).
    double octScale = std::pow(2.0, static_cast<double>(oct));  // pixels per octave unit
    int    base_x   = static_cast<int>(std::round(kpt.pt.x * 2.0 / octScale));
    int    base_y   = static_cast<int>(std::round(kpt.pt.y * 2.0 / octScale));

    double scale         = ORI_SCALE_FACTOR * kpt.size / (octScale * 2.0);
    int    radius        = static_cast<int>(std::round(scale * ORI_RADIUS_FACTOR));
    double weight_factor = -0.5 / (scale * scale);

    std::vector<double> hist(ORI_BINS, 0.0), smooth(ORI_BINS, 0.0);

    for (int i = -radius; i <= radius; ++i) {
        int ry = base_y + i;
        if (ry <= 0 || ry >= sz.height - 1) continue;
        for (int j = -radius; j <= radius; ++j) {
            int rx = base_x + j;
            if (rx <= 0 || rx >= sz.width - 1) continue;

            double dx = safeGet(img, ry,     rx + 1) - safeGet(img, ry,     rx - 1);
            double dy = safeGet(img, ry - 1, rx    ) - safeGet(img, ry + 1, rx    );
            double mag = std::sqrt(dx * dx + dy * dy);
            double ori = rad2deg(std::atan2(dy, dx));

            int idx = (static_cast<int>(std::round(ori * ORI_BINS / 360.0))
                       % ORI_BINS + ORI_BINS) % ORI_BINS;
            hist[idx] += std::exp(weight_factor * (i * i + j * j)) * mag;
        }
    }

    // Smooth histogram (same 6-4-1 kernel as reference)
    auto circ = [&](int i) { return (i + ORI_BINS) % ORI_BINS; };
    for (int i = 0; i < ORI_BINS; ++i) {
        smooth[i] = (6.0 * hist[i]
                     + 4.0 * (hist[circ(i-1)] + hist[circ(i+1)])
                     +        hist[circ(i-2)]  + hist[circ(i+2)]) / 16.0;
    }

    double maxOri = *std::max_element(smooth.begin(), smooth.end());
    std::vector<cv::KeyPoint> result;

    for (int i = 0; i < ORI_BINS; ++i) {
        double l = smooth[circ(i-1)], r = smooth[circ(i+1)];
        if (smooth[i] > l && smooth[i] > r && smooth[i] >= ORI_PEAK_RATIO * maxOri) {
            double peak = smooth[i];
            double interp = std::fmod(
                i + 0.5 * (l - r) / (l + r - 2.0 * peak),
                static_cast<double>(ORI_BINS));
            double orientation = 360.0 - interp * 360.0 / ORI_BINS;
            if (std::abs(360.0 - orientation) < EPS) orientation = 0.0;

            cv::KeyPoint nk = kpt;
            nk.angle = static_cast<float>(orientation);
            result.push_back(nk);
        }
    }

    // Fallback: if histogram was flat / empty, pick the global max bin
    if (result.empty()) {
        int best = static_cast<int>(
            std::max_element(smooth.begin(), smooth.end()) - smooth.begin());
        double orientation = 360.0 - best * 360.0 / ORI_BINS;
        cv::KeyPoint nk = kpt;
        nk.angle = static_cast<float>(orientation);
        result.push_back(nk);
    }

    return result;
}

// =============================================================================
//  4.  128-D SIFT descriptor for a single keypoint
//      Returns false if the keypoint is too close to the image border.
// =============================================================================
static bool computeDescriptor(const cv::KeyPoint&                    kpt,
                               const std::vector<std::vector<cv::Mat>>& pyr,
                               std::vector<float>&                    desc)
{
    // Unpack octave / layer from kpt.octave
    int oct   = kpt.octave & 255;
    int layer = (kpt.octave >> 8) & 255;
    if (oct >= static_cast<int>(pyr.size()))           return false;
    if (layer >= static_cast<int>(pyr[oct].size()))    return false;

    const cv::Mat& image = pyr[oct][layer];
    const cv::Size sz    = image.size();

    // Map keypoint coordinates into this octave's coordinate frame
    // Original image space → octave space: multiply by 2 (pyramid base ×2)
    //                                       then divide by 2^oct
    double octScale = std::pow(2.0, static_cast<double>(oct));
    int    pt_x     = static_cast<int>(std::round(kpt.pt.x * 2.0 / octScale));
    int    pt_y     = static_cast<int>(std::round(kpt.pt.y * 2.0 / octScale));

    double angle    = 360.0 - static_cast<double>(kpt.angle);
    double cosA     = std::cos(deg2rad(angle));
    double sinA     = std::sin(deg2rad(angle));

    // hist_width: half-width of the sampling window in pixels
    double hist_width = DESC_SCALE_MUL * 0.5 * (kpt.size * 2.0 / octScale);
    int    half_width = static_cast<int>(std::min(
        std::round(hist_width * (DESC_WINDOW_WIDTH + 1) / std::sqrt(2.0)),
        std::sqrt(static_cast<double>(sz.height * sz.height + sz.width * sz.width))));

    double weight_mul = -0.5 / std::pow(0.5 * DESC_WINDOW_WIDTH, 2.0);
    constexpr double bins_per_deg = DESC_HIST_BINS / 360.0;

    // Accumulators
    std::vector<double> rows, cols, mags, oris;
    rows.reserve(4 * half_width * half_width);
    cols.reserve(rows.capacity());
    mags.reserve(rows.capacity());
    oris.reserve(rows.capacity());

    for (int i = -half_width; i <= half_width; ++i) {
        for (int j = -half_width; j <= half_width; ++j) {
            double row_rot = sinA * j + cosA * i;
            double col_rot = sinA * i - cosA * j;

            double bin_row = row_rot / hist_width + 0.5 * (DESC_WINDOW_WIDTH - 1);
            double bin_col = col_rot / hist_width + 0.5 * (DESC_WINDOW_WIDTH - 1);

            if (bin_row <= -1.0 || bin_col <= -1.0 ||
                bin_row >= DESC_WINDOW_WIDTH || bin_col >= DESC_WINDOW_WIDTH)
                continue;

            int wr = pt_y + i, wc = pt_x + j;
            if (wr <= 0 || wc <= 0 || wr >= sz.height - 1 || wc >= sz.width - 1)
                continue;

            double dx  = safeGet(image, wr, wc + 1) - safeGet(image, wr, wc - 1);
            double dy  = safeGet(image, wr - 1, wc) - safeGet(image, wr + 1, wc);
            double mag = std::sqrt(dx * dx + dy * dy);
            double ori = std::fmod(rad2deg(std::atan2(dy, dx)), 360.0);

            double exponent = (row_rot * row_rot + col_rot * col_rot)
                              / (hist_width * hist_width);
            double weight = std::exp(weight_mul * exponent);

            rows.push_back(bin_row);
            cols.push_back(bin_col);
            mags.push_back(mag * weight);
            oris.push_back((ori - angle) * bins_per_deg);
        }
    }

    // Build 3-D histogram tensor  (WINDOW_WIDTH+2) × (WINDOW_WIDTH+2) × HIST_BINS
    // The +2 guard-ring around each axis absorbs trilinear spill-over.
    const int   W  = DESC_WINDOW_WIDTH + 2;
    const int   B  = DESC_HIST_BINS;
    std::vector<double> tensor(static_cast<size_t>(W * W * B), 0.0);

    auto tidx = [&](int r, int c, int b) {
        return (r * W + c) * B + b;
    };

    for (size_t l = 0; l < rows.size(); ++l) {
        int    rb  = static_cast<int>(std::floor(rows[l]));
        int    cb  = static_cast<int>(std::floor(cols[l]));
        int    ob  = static_cast<int>(std::floor(oris[l]));

        double rp  = rows[l] - rb;
        double cp  = cols[l] - cb;
        double op  = oris[l] - ob;

        if (ob < 0)   ob += B;
        if (ob >= B)  ob -= B;

        for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
        for (int k = 0; k < 2; ++k) {
            int rr = rb + 1 + i;   // +1 for the guard-ring offset
            int cc = cb + 1 + j;
            int oo = ((ob + k) % B + B) % B;

            if (rr < 0 || rr >= W || cc < 0 || cc >= W) continue;

            double c_val = mags[l];
            c_val *= (i == 0) ? (1.0 - rp) : rp;
            c_val *= (j == 0) ? (1.0 - cp) : cp;
            c_val *= (k == 0) ? (1.0 - op) : op;
            tensor[tidx(rr, cc, oo)] += c_val;
        }
    }

    // Extract the inner WINDOW_WIDTH × WINDOW_WIDTH region → 128-D
    desc.clear();
    desc.reserve(DESC_WINDOW_WIDTH * DESC_WINDOW_WIDTH * DESC_HIST_BINS);
    for (int i = 1; i <= DESC_WINDOW_WIDTH; ++i)
    for (int j = 1; j <= DESC_WINDOW_WIDTH; ++j)
    for (int k = 0; k < DESC_HIST_BINS; ++k)
        desc.push_back(static_cast<float>(tensor[tidx(i, j, k)]));

    // Normalise → clamp at DESC_MAX_VAL → re-normalise → quantise to [0,255]
    double norm = 0.0;
    for (float v : desc) norm += static_cast<double>(v) * v;
    norm = std::sqrt(norm);

    double thresh = norm * DESC_MAX_VAL;
    norm = 0.0;
    for (float& v : desc) {
        v = static_cast<float>(std::min(static_cast<double>(v), thresh));
        norm += static_cast<double>(v) * v;
    }
    norm = std::sqrt(norm);

    for (float& v : desc) {
        v /= static_cast<float>(std::max(norm, EPS));
        v  = std::round(v * 512.0f);
        v  = std::max(0.0f, std::min(255.0f, v));
    }

    return true;
}

// =============================================================================
//  5.  Process one keypoint set → oriented keypoints + descriptors
// =============================================================================
static void processKeypoints(
        const std::vector<cv::KeyPoint>&         inKps,
        const std::vector<std::vector<cv::Mat>>& pyr,
        int                                      numOctaves,
        std::vector<cv::KeyPoint>&               outKps,
        cv::Mat&                                 outDesc)
{
    outKps.clear();
    std::vector<std::vector<float>> descRows;

    for (cv::KeyPoint kpt : inKps) {
        // ── 1. Assign pyramid level ─────────────────────────────────────
        int oct, layer;
        assignPyramidLevel(kpt, numOctaves, oct, layer);

        // ── 2. Orientation assignment (may produce multiple kpts) ───────
        std::vector<cv::KeyPoint> oriented = assignOrientations(kpt, oct, layer, pyr);

        // ── 3. Descriptor for each oriented variant ─────────────────────
        for (cv::KeyPoint& ok : oriented) {
            // Re-assign level (assignOrientations copies kpt.octave already)
            std::vector<float> d;
            if (computeDescriptor(ok, pyr, d)) {
                outKps.push_back(ok);
                descRows.push_back(std::move(d));
            }
        }
    }

    // Pack into cv::Mat  (N × 128, CV_32F)
    if (descRows.empty()) {
        outDesc = cv::Mat();
        return;
    }
    outDesc.create(static_cast<int>(descRows.size()), 128, CV_32F);
    for (int i = 0; i < static_cast<int>(descRows.size()); ++i)
        std::copy(descRows[i].begin(), descRows[i].end(),
                  outDesc.ptr<float>(i));
}

// =============================================================================
//  PUBLIC API
// =============================================================================
DescriptionOutput runDescription(const cv::Mat&          imgBGR,
                                 const ExtractionOutput& extraction,
                                 int                     numOctaves,
                                 double                  /*sigmaBase — reserved*/)
{
    // ── Greyscale float image for the pyramid ────────────────────────────────
    cv::Mat gray, grayF64;
    if (imgBGR.channels() == 3)
        cv::cvtColor(imgBGR, gray, cv::COLOR_BGR2GRAY);
    else
        gray = imgBGR.clone();
    gray.convertTo(grayF64, CV_64F, 1.0 / 255.0);

    // ── Build shared Gaussian pyramid ────────────────────────────────────────
    auto pyr = buildGaussianPyramid(grayF64, numOctaves);

    // ── Harris descriptors ───────────────────────────────────────────────────
    DescriptionOutput out;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        processKeypoints(extraction.harrisKps, pyr, numOctaves,
                         out.harrisKps, out.harrisDesc);
        auto t1 = std::chrono::high_resolution_clock::now();
        out.harrisTimeMs =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // ── Lambda- descriptors ──────────────────────────────────────────────────
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        processKeypoints(extraction.lambdaKps, pyr, numOctaves,
                         out.lambdaKps, out.lambdaDesc);
        auto t1 = std::chrono::high_resolution_clock::now();
        out.lambdaTimeMs =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    // ── Visualisation ────────────────────────────────────────────────────────
    cv::drawKeypoints(imgBGR, out.harrisKps, out.harrisVis,
                      cv::Scalar(0, 0, 255),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::drawKeypoints(imgBGR, out.lambdaKps, out.lambdaVis,
                      cv::Scalar(0, 255, 0),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    return out;
}