#include <iostream>
#include "encoder.h"
#include "decoder.h"

using namespace std;

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        cout << "Usage: encode/decode file\n";
        return 1;
    }

    string mode = argv[1];
    string file = argv[2];

    if (mode == "encode")
        encodeImage(file, "key.json");

    else if (mode == "decode")
        decodeKey(file, "result.png", "result.txt");

    else
        cout << "Invalid mode";

    return 0;
}