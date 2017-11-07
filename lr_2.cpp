/*************************************************************************
	> File Name: lr_2.cpp
	> Author: 
	> Mail: 
	> Created Time: 2017年11月07日 星期二 23时23分16秒
 ************************************************************************/

#include<stdio.h>
#include<iostream>
using namespace std;

double compute_gradient_0(double x[], double y[], double w0, double w1, int sample_number) {
    double sum = 0;
    for (int i=0; i < sample_number; i++) {
        sum += w0 + w1 * x[i] -y[i];
    }
    return sum / sample_number;
}

double compute_gradient_1(double x[], double y[], double w0, double w1, int sample_number) {
    double sum = 0;
    for (int i=0; i < sample_number; i++) {
        sum += (w0 + w1 * x[i] - y[i]) * x[i];
    }
    return sum / sample_number;
}

void gradient_descent(double x[], double y[], int sample_number, double alpha, int iterate_number, double &w0, double &w1) {
    w0 = 0;
    w1 = 1;
    for (int i=0; i < sample_number; i++) {
        double temp0 = w0 - alpha * compute_gradient_0(x, y, w0, w1, sample_number);
        double temp1 = w1 - alpha * compute_gradient_1(x, y, w0, w1, sample_number);
        w0 = temp0;
        w1 = temp1;
    }
}

double compute_cost(double x[], double y[], double w0, double w1, int sample_number) {
    double sum = 0;
    for (int i=0; i < sample_number; i++) {
        sum += (w0 + w1 * x[i] - y[i]) * (w0 + w1 * x[i] - y[i]);
    }
    return sum / (2 *sample_number);
}

double predict(double w0, double w1, double x) {
    return w0 + w1 * x;
}

int main() {
    double x[6] = {96.79, 110.39, 70.25, 99.96, 118.15, 115.08};
    double y[6] = {287, 343, 199, 298, 340, 350};
    int sample_number = 6;

    double alpha = 0.0001;
    int iterate_number = 1500;
    double w0 = 0;
    double w1 = 0;
    gradient_descent(x, y, sample_number, alpha, iterate_number, w0, w1);
    double cost = compute_cost(x, y, w0, w1, sample_number);
    printf("After %d iterates, the cost Error(w0, w1) is %lf\n", iterate_number, cost);
    //printf("w0 = [%.3lf], w1 = [%.3lf]\n", w0, w1);
    //printf("predict(112) = %.3lf\n", predict(w0, w1, 112));
    //printf("predict(110) = %.3lf\n", predict(w0, w1, 110));
    printf("w0 = [%lf], w1 = [%lf]\n", w0, w1);
    printf("predict(112) = %lf\n", predict(w0, w1, 112));
    printf("predict(110) = %lf\n", predict(w0, w1, 110));
    return 0;
}
