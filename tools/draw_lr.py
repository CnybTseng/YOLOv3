import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--workspace', type=str, default='workspace', help='workspace path')
args = parser.parse_args()

lr = np.loadtxt(f'{args.workspace}/log/lr.txt')
plt.title('learning scheduler')
plt.xlabel('batch index')
plt.ylabel('learning rate')
plt.plot(lr)
plt.show()