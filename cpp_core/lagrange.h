#ifndef LAGRANGE_H
#define LAGRANGE_H

#include <vector>

double lagrange(const std::vector<double>& x,
                const std::vector<double>& y,
                double xi);

#endif