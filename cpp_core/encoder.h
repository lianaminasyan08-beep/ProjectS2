#ifndef ENCODER_H
#define ENCODER_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

struct Point2D {
    float x;
    float y;
};

class Encoder {
public:
    std::vector<std::vector<Point2D>> encodeImage(const cv::Mat& img);

private:
    std::vector<cv::Point> orderContour(const std::vector<cv::Point>& pts);
    std::vector<cv::Point2f> resample(const std::vector<cv::Point>& pts, int n = 40);
};

#endif