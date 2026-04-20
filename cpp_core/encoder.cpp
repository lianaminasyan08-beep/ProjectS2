#include "encoder.h"
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace cv;
using namespace std;

void encodeImage(string imgPath, string outFile)
{
    Mat img = imread(imgPath, 0);

    if (img.empty())
        return;

    threshold(img, img, 128, 255, THRESH_BINARY_INV);

    vector<vector<Point>> contours;
    findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    ofstream f(outFile);
    f << "[";

    for (size_t i = 0; i < contours.size(); i++)
    {
        f << "{\"points\":[";

        auto &c = contours[i];

        for (size_t j = 0; j < c.size(); j += max(1, (int)c.size()/30))
        {
            f << "[" << c[j].x << "," << c[j].y << "]";
            if (j + 1 < c.size()) f << ",";
        }

        f << "]}";

        if (i + 1 < contours.size()) f << ",";
    }

    f << "]";
}