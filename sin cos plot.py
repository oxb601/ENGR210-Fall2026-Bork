# plotting
import numpy as np
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt

x = np.arange(0, 24, 0.1)
y1=np.sin(x)*2
y2=np.cos(x)*2

plt.plot(x,y1,label='sin(x)', ls='--', color='green')
plt.plot(x,y2,label='cos(x)', ls='-.', color='red')


plt.xlabel('t')
plt.ylabel('f(t)')
plt.legend(loc='upper right')
plt.show()