#include<iostream>
#include<stdio.h>
#include<math.h>
using namespace std;

const int MAX_FEATURE_DIMENSION = 3;
const int MAX_SAMPLE_NUMBER = 12;
const int MAX_ITERATE_NUMBER = 10;

double sigmoid(double z) {
    return 1 / (1 + exp(-z));
}

double hfun(double x[], double th[], int feature_number) {
    double s = 0.0;
    for (int i=0; i < feature_number; i++) {
        s += th[i] * x[i]; 
    }
    return sigmoid(s);
}

double compute_gradient(double X[][MAX_FEATURE_DIMENSION], int y[], double th[], int feature_number, int feature_pos, int sample_num) {
    double sum = 0.0;
    for (int i=0; i < sample_num; i++) {
        double hth = hfun(X[i], th, feature_number);
        sum += (hth - y[i]) * X[i][feature_pos];
    }
    return sum / sample_num;
}

double compute_cost(double X[][MAX_FEATURE_DIMENSION], int y[], double th[], int feature_number, int sample_num) {
    double sum = 0;
    for (int i=0; i < sample_num; i++) {
        double hth = hfun(X[i], th, feature_number);
        double res = -y[i] * log(hth) - (1 - y[i]) * log(1 - hth);
        sum += res;
    }
    return sum / sample_num;
}

void gradient_descent(double X[][MAX_FEATURE_DIMENSION], int y[], double th[], int feature_number, int sample_num, double alpha, int iterate_number, double J[]) {
    for (int i=0; i < iterate_number; i++) {
        double temp[MAX_FEATURE_DIMENSION] = {0};
        for (int j=0; j < feature_number; j++) {
            temp[j] = th[j] - alpha * compute_gradient(X, y, th, feature_number, j, sample_num);
        }
        for (int j=0; j < feature_number; j++) {
            th[j] = temp[j];
        }
        J[i] = compute_cost(X, y, th, feature_number, sample_num);
    }
}


int main() {
    double X[MAX_SAMPLE_NUMBER][MAX_FEATURE_DIMENSION] = {
        {1, 34.6, 78.0},
        {1, 30.2, 43.8},
        {1, 35.8, 72.9},
        {1, 60.1, 86.3},
        {1, 79.0, 75.3},
        {1, 45.0, 56.3},
        {1, 61.1, 96.5},
        {1, 75.0, 46.5},
        {1, 76.0, 87.4},
        {1, 84.4, 43.5},
        {1, 95.8, 38.2},
        {1, 75.0, 30.6}
    };
    double J[MAX_ITERATE_NUMBER];
    int y[MAX_SAMPLE_NUMBER] = {0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0};
    double th[MAX_FEATURE_DIMENSION] = {0};
    double alpha = 0.001;
    gradient_descent(X, y, th, MAX_FEATURE_DIMENSION, MAX_SAMPLE_NUMBER, alpha, MAX_ITERATE_NUMBER, J);

    for (int i = 0; i < MAX_ITERATE_NUMBER; i++) {
        printf("%lf\n", J[i]);
    }
    printf("\n");
    for (int i = 0; i < MAX_FEATURE_DIMENSION; i++) {
        printf("%lf\n", th[i]);
    }

    printf("\n");
    return 0;
}


//0.693542
//0.692222
//0.692593
//0.691437
//0.691784
//0.690761
//0.691085
//0.690169
//0.690472
//0.689647
//
//-0.000744
//0.000890
//-0.000123

