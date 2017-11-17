# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-11, 15, 500)
y = x ** 2 - 4 * x - 5


plt.figure(figsize=(10, 5))
plt.xticks()


plt.plot(x, y, label='y=x^2-4x-5')

y1 = 2 * x - 4
plt.plot(x, y1, label='y=2*x-4')

plt.axvline(2, label='x=2', color='red')
plt.axhline(-9, label='y=-9', color='blue')
plt.legend()
plt.show()
