#include "Matching.h"

MatchingOutput runMatching(const MatchingInput& harris, const MatchingInput& lambda)
{
    // harris.descA  — Image A Harris descriptors  (N x 128)
    // harris.descB  — Image B Harris descriptors  (M x 128)
    // harris.kptsA  — Image A Harris keypoints    (N)
    // harris.kptsB  — Image B Harris keypoints    (M)

    // lambda.descA  — Image A λ- descriptors      (N x 128)
    // lambda.descB  — Image B λ- descriptors      (M x 128)
    // lambda.kptsA  — Image A λ- keypoints        (N)
    // lambda.kptsB  — Image B λ- keypoints        (M)

    // your matching logic here

    return { harris, lambda };
}