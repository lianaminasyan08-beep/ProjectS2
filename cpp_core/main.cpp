#include <iostream>
#include "encoder.h"

using namespace std;

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        cout << "Usage: app image.jpg\n";
        return 0;
    }

    encodeImage(argv[1]);
    cout << "Encoded -> key.json created\n";

    return 0;
}