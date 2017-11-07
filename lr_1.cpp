/*************************************************************************
	> File Name: lr_1.cpp
	> Author: 
	> Mail: 
	> Created Time: Tue 07 Nov 2017 10:04:06 AM CST
 ************************************************************************/

#include <stdio.h>
#include <ctime>

#include <iostream>

double gradient(double x) {
    return 2 * x -4;
}

int main() {

    clock_t start, end;
    start = clock();

    double x = 0;
    int iteration_number = 150000;
    double alpha = 0.0001;
    while (iteration_number--) {
        double g = gradient(x);
        x -= alpha * g;
        //std::cout << "x=" << x << " gradient=" << g << std::endl;
    }
    printf("x= %lf\n", x);
    end = clock();
    std::cout << end - start << "/" << CLOCKS_PER_SEC << "(s)" << std::endl;
    return 0;
}
