#include "roi.hpp"

#include <opencv2/imgproc.hpp>

std::vector<cv::Point> toPixels(const std::vector<cv::Point2f>& normalized, int width, int height)
{
    std::vector<cv::Point> out;
    out.reserve(normalized.size());
    for (const auto& p : normalized)
    {
        out.emplace_back(static_cast<int>(p.x * width), static_cast<int>(p.y * height));
    }
    return out;
}

bool pointInPoly(const cv::Point& p, const std::vector<cv::Point>& poly)
{
    if (poly.size() < 3)
    {
        return false;
    }
    return cv::pointPolygonTest(poly, p, false) >= 0.0;
}
