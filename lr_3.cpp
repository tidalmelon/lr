#include<stdio.h>
#include<iostream>
using namespace std;

const int MAX_FEATURE_DIMENSION = 5;


double compute_gradient(double X[][MAX_FEATURE_DIMENSION], int y[], double w[], int feature_number, int feature_pos, int sample_num) {
    double sum = 0;
    for (int i=0; i < sample_num; i++) {
        double res = 0;
        for (int j=0; j < feature_number; j++) {
            res += w[j] * X[i][j];
        }
        sum += (res - y[i]) * X[i][feature_pos];
    }
    return sum / sample_num;
}

void gradient_descent(double X[][MAX_FEATURE_DIMENSION], int y[], double w[], int feature_number, int sample_num, double alpha, int iterate_number) {
    for (int i=0; i < iterate_number; i++) {
        double temp[MAX_FEATURE_DIMENSION] = {0};
        for (int j=0; j < feature_number; j++) {
            temp[j] = w[j] - alpha * compute_gradient(X, y, w, feature_number, j, sample_num);
        }
        for (int j=0; j < feature_number; j++) {
            w[j] = temp[j];
        }
    }
}

double compute_cost(double X[][MAX_FEATURE_DIMENSION], int y[], double w[], int feature_number, int sample_num) {
    double sum = 0;
    for (int i=0; i < sample_num; i++) {
        double res = 0;
        for (int j=0; j < feature_number; j++) {
            res += w[j] * X[i][j];
        }
        sum += (res - y[i]) * (res - y[i]);
    }
    return sum / (2*sample_num);
}

double predict(double w[], double x[], int feature_number) {
    double sum = 0;
    for (int i=0; i < feature_number; i++) {
        sum += w[i] * x[i];
    }
    return sum;    
}


int main() {
    double X[][MAX_FEATURE_DIMENSION] = {
        {1, 96.79, 2, 1, 2},
        {1, 110.39, 3, 1, 0},
        {1, 70.25, 1, 0, 2},
        {1, 99.96, 2, 1, 1},
        {1, 118.15, 3, 1, 0},
        {1, 115.08, 3, 1, 2}
    };

    int y[6] = {287, 343, 199, 298, 340, 350};
    int sample_num = 6;
    double alpha = 0.0001;
    int iterate_number = 1500;
    int feature_number = 5;
    double w[5] = {0};
    gradient_descent(X, y, w, feature_number, sample_num, alpha, iterate_number);
    double cost = compute_cost(X, y, w, feature_number, sample_num);
    double testx1[] = {1, 112, 3, 1, 0};
    double testx2[] = {1, 110, 3, 1, 1};
    
    printf("After %d iterates, the cost Error(w0, w1) is %lf\n", iterate_number, cost);
    for (int i=0; i<feature_number; i++) {
        printf("w%d=%lf\n", i, w[i]);
    }
    printf("predict(112) = %lf\n", predict(w, testx1, 112));
    printf("predict(110) = %lf\n", predict(w, testx2, 110));
    return 0;

}
