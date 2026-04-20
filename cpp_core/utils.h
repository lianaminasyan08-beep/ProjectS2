#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <opencv2/opencv.hpp>

std::vector<cv::Point2f> sampleContour(std::vector<cv::Point> contour, int n);

#endif