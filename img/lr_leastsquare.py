# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

k1 = 2.981086
b1 = 0.018653
    
k2 = 3.20708095323
b2 = -23.5512952764

x = [96.79, 110.39, 70.25, 99.96, 118.15, 115.08]
y = [287, 343, 199, 298, 340, 350]

xx = np.linspace(65, 120, 500)
y1 = k1 * xx + b1
y2 = k2 * xx + b2


plt.figure(figsize=(10, 5))
plt.plot(xx, y1, label='line regression')
plt.scatter(x, y, marker='+')

plt.plot(xx, y2, label='least square')
plt.legend()
plt.show()


