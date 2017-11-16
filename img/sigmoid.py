# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 10000)
y = 1.0 / (1 + np.exp(-x))

plt.subplot(211)

plt.ylim(0, 1)
plt.xlim(-5, 5)
plt.plot(x, y, label='sigmoid-1')
plt.legend()

x = np.linspace(-100, 100, 10000)
y = 1.0 / (1 + np.exp(-x))
plt.subplot(212)
plt.ylim(0, 1)
plt.xlim(-100, 100)

plt.plot(x, y, label='sigmoid-2')
plt.legend()
plt.show()
