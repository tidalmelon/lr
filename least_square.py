# -*- coding: utf-8 -*-

def Mean(t):
    return float(sum(t)) / len(t)

def Var(t, mu=None):
    if mu is None:
        mu = Mean(t)
    dev2 = [(x - mu) ** 2 for x in t]
    var = Mean(dev2)
    return var

def MeanVar(t):
    mu = Mean(t)
    var = Var(t, mu)
    return mu, var

def Cov(xs, ys, mux=None, muy=None):
    if mux is None:
        mux = Mean(xs)
    if muy is None:
        muy = Mean(ys)

    total = 0.0
    for x, y in zip(xs, ys):
        total += (x-mux)*(y-muy)
    return total / len(xs)


def LeastSquare(xs, ys):
    xbar, varx = MeanVar(xs)
    ybar, vary = MeanVar(ys)
    # 斜率=XY的协方差/X的方差
    slope = Cov(xs, ys, xbar, ybar) / varx

    inter = ybar - slope * xbar
    return inter, slope

def compute_cost(x, y, w0, w1, sample_number):
    total = 0.0
    for i in range(sample_number):
        total += (w1 * x[i] + w0 - y[i]) ** 2
    return total / (2 * sample_number)


def Predict(inter, slope, x):
    return inter + slope * x


if __name__ == '__main__':
    x = [96.79, 110.39, 70.25, 99.96, 118.15, 115.08]
    y = [287, 343, 199, 298, 340, 350]

    inter, slope = LeastSquare(x, y)
    cost = compute_cost(x, y, inter, slope, 6)
    print 'least square error cost(%s, %s) is %s' % (inter, slope, cost)
    print 'predict(112) = ', Predict(inter, slope, 112) 
    print 'predict(110) = ', Predict(inter, slope, 110) 
