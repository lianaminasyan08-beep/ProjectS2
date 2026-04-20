#include "decoder.h"
#include "lagrange.h"

#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <sstream>

using namespace cv;
using namespace std;

// ---------------- SIMPLE PARSER ----------------
static vector<vector<Point>> parseKey(string data)
{
    vector<vector<Point>> chars;
    vector<Point> current;

    int x = 0, y = 0;
    bool reading = false;

    for (size_t i = 0; i < data.size(); i++)
    {
        if (sscanf(data.c_str() + i, "[%d,%d]", &x, &y) == 2)
        {
            current.push_back(Point(x, y));
            reading = true;
        }

        if (data[i] == '}' && reading)
        {
            if (!current.empty())
            {
                chars.push_back(current);
                current.clear();
            }
            reading = false;
        }
    }

    if (!current.empty())
        chars.push_back(current);

    return chars;
}

// ---------------- DECODER ----------------
void decodeKey(string file, string outImg, string outTxt)
{
    ifstream f(file);
    string data((istreambuf_iterator<char>(f)), {});

    vector<vector<Point>> characters = parseKey(data);

    Mat canvas = Mat::zeros(300, 600, CV_8UC1);

    string result = "";

    int charIndex = 0;

    for (auto &ch : characters)
    {
        vector<double> X, Y;

        for (auto &p : ch)
        {
            X.push_back(p.x);
            Y.push_back(p.y);
        }

        // reconstruct curve using Lagrange
        for (double xi = 0; xi < 600; xi++)
        {
            double yi = lagrange(X, Y, xi);

            if (yi >= 0 && yi < 300)
                canvas.at<uchar>((int)yi, (int)xi) = 255;
        }

        // simple placeholder recognition (you can upgrade later)
        result += "#";
        charIndex++;
    }

    imwrite(outImg, canvas);

    ofstream out(outTxt);
    out << result;
}