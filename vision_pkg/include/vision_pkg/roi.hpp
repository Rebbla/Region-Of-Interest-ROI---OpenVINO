#pragma once

#include <opencv2/core.hpp>
#include <vector>

std::vector<cv::Point> toPixels(const std::vector<cv::Point2f>& normalized, int width, int height);

bool pointInPoly(const cv::Point& p, const std::vector<cv::Point>& poly);
