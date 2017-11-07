# -*- coding: utf-8 -*-
import time

def gradient(x):
    return 2 * x - 4


start = time.clock()

x = 0.0
alpha = 0.0001
iteration_number = 150000
while iteration_number:
    g = gradient(x)
    x -= alpha * g
    #print 'x=', x, ' gradient=', g
    iteration_number -= 1

print 'x=', x

end = time.clock()
print (end - start) * 1000000, "(s)" 


