import numpy as np
import matplotlib.pyplot as plt

lr = np.loadtxt('log/lr.txt')
plt.title('learning scheduler')
plt.xlabel('batch index')
plt.ylabel('learning rate')
plt.plot(lr)
plt.show()