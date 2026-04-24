#include "encoder.h"
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace std;

// =========================
// ORDER POINTS (ANGLE SORT)
// =========================
vector<Point> Encoder::orderContour(const vector<Point>& pts)
{
    Point2f center(0, 0);

    for (auto& p : pts)
        center += Point2f(p.x, p.y);

    center *= (1.0f / pts.size());

    vector<pair<float, Point>> angPts;

    for (auto& p : pts)
    {
        float angle = atan2(p.y - center.y, p.x - center.x);
        angPts.push_back({angle, p});
    }

    sort(angPts.begin(), angPts.end(),
        [](auto& a, auto& b) {
            return a.first < b.first;
        });

    vector<Point> ordered;
    for (auto& ap : angPts)
        ordered.push_back(ap.second);

    return ordered;
}

// =========================
// RESAMPLE TO FIXED SIZE
// =========================
vector<Point2f> Encoder::resample(const vector<Point>& pts, int n)
{
    vector<Point2f> res;

    if (pts.size() < 2)
        return res;

    vector<float> dist(pts.size(), 0);

    for (size_t i = 1; i < pts.size(); i++)
    {
        float dx = pts[i].x - pts[i - 1].x;
        float dy = pts[i].y - pts[i - 1].y;
        dist[i] = dist[i - 1] + sqrt(dx * dx + dy * dy);
    }

    float total = dist.back();

    for (int i = 0; i < n; i++)
    {
        float t = (float)i / (n - 1) * total;

        for (size_t j = 1; j < dist.size(); j++)
        {
            if (dist[j] >= t)
            {
                float ratio = (t - dist[j - 1]) / (dist[j] - dist[j - 1] + 1e-6f);

                float x = pts[j - 1].x + ratio * (pts[j].x - pts[j - 1].x);
                float y = pts[j - 1].y + ratio * (pts[j].y - pts[j - 1].y);

                res.push_back(Point2f(x, y));
                break;
            }
        }
    }

    return res;
}

// =========================
// MAIN ENCODER FUNCTION
// =========================
vector<vector<Point2D>> Encoder::encodeImage(const Mat& img)
{
    vector<vector<Point2D>> output;

    Mat gray, bin;
    if (img.channels() == 3)
        cvtColor(img, gray, COLOR_BGR2GRAY);
    else
        gray = img.clone();

    threshold(gray, bin, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

    vector<vector<Point>> contours;
    findContours(bin, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

    for (auto& c : contours)
    {
        if (c.size() < 10) continue;

        auto ordered = orderContour(c);
        auto sampled = resample(ordered, 40);

        vector<Point2D> pts;

        for (auto& p : sampled)
        {
            pts.push_back({p.x, p.y});
        }

        output.push_back(pts);
    }

    return output;
}