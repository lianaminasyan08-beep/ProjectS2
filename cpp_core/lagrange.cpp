#include "lagrange.h"

double lagrange(const std::vector<double>& x,
                const std::vector<double>& y,
                double xi)
{
    double sum = 0;

    for (size_t i = 0; i < x.size(); i++)
    {
        double term = y[i];

        for (size_t j = 0; j < x.size(); j++)
        {
            if (i != j)
                term *= (xi - x[j]) / (x[i] - x[j]);
        }

        sum += term;
    }

    return sum;
}