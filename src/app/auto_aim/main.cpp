#include <iostream>

#include "behavior.hpp"
#include "vision.hpp"

int main(int argc, char const *argv[])
{
    std::cout << "Auto aimer started." << std::endl;
    RunTestTree();
    TestOpenCV();
    TestVideoWrite();
    TestOpenCVGraph();
    return 0;
}
