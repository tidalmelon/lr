# linear regression

## 一元二次方程求极值
梯度下降法:  lr_1.cpp lr_1.py  

```math
\theta_i = \theta_i - \alpha\frac\partial{\partial\theta_i}J(\theta)
```

```
伪代码
while ( max_iterate_num or other ) {
    x = x - alpha * 导数(f(x))
}
```

##一元线性回归

```
lr_2.cpp
After 1500 iterates, the cost Error(w0, w1) is 41.844789
w0 = [0.018653], w1 = [2.981086]
predict(112) = 333.900241
predict(110) = 327.938070

```
```
最小二乘拟合
least_square.py

least square error cost(-23.5512952764, 3.20708095323) is 35.1218735092
predict(112) =  335.641771485
predict(110) =  329.227609578

```
## 一元线性回归 与 最小二乘拟合对比
![](https://github.com/tidalmelon/lr/blob/master/img/lr_leastsquare.png)

```
多元线性回归
lr_3.cpp
After 1500 iterates, the cost Error(w0, w1) is 40.559510
w0=-0.053241
w1=2.973633
w2=0.364542
w3=0.196225
w4=-0.198871
predict(112) = 325.488320
predict(110) = 325.977873
```

## 逻辑回归
sigmoid 函数： 

![](https://github.com/tidalmelon/lr/blob/master/img/sigmoid.png)
