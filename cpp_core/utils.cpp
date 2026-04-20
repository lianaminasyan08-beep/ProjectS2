#include "utils.h"

std::vector<cv::Point2f> sampleContour(std::vector<cv::Point> contour, int n)
{
    std::vector<cv::Point2f> pts;

    int step = contour.size() / n;
    if (step == 0) step = 1;

    for (int i = 0; i < contour.size(); i += step)
    {
        pts.push_back(contour[i]);
        if (pts.size() >= n) break;
    }

    return pts;
}