import argparse
import numpy as np
import matplotlib.pyplot as plt

def median_filter1d(data, kernel_size=3):
    radius = kernel_size // 2
    size = len(data)
    filtered_data = [0] * size
    filtered_data[0:radius] = data[0:radius]
    filtered_data[size-radius:size] = data[size-radius:size]
    for i in range(radius, size-radius):
        filtered_data[i] = np.sort(data[i-radius:i+radius+1])[radius]
    return filtered_data

def replace_zero(data, eps=0.000001):
    if data[0] < eps:
        for i in range(len(data)):
            if data[i] > eps:
                data[0] = data[i]
                break
    
    for i in range(1, len(data)):
        if data[i] < eps:
            data[i] = data[i-1]
    
    return data

parser = argparse.ArgumentParser()
parser.add_argument('--interval', '-i', type=int, default=50, help='data drawing interval')
args = parser.parse_args()

obj = list()
bkg = list()

with open('../YOLOv3/log/verbose.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        if line:
            segments = line.split()
            for segment in segments:
                if 'Obj' in segment:
                    v = float(segment.replace('Obj:', ''))
                    obj.append(v)
                elif 'Bkg' in segment:
                    v = float(segment.replace('Bkg:', ''))
                    bkg.append(v)

yolo1_obj = replace_zero(obj[0::3])
yolo2_obj = replace_zero(obj[1::3])
yolo3_obj = replace_zero(obj[2::3])

yolo1_bkg = bkg[0::3]
yolo2_bkg = bkg[1::3]
yolo3_bkg = bkg[2::3]

plt.xlabel('batches')
plt.ylabel('objectness')

interval = args.interval
plt.plot(yolo1_obj[0::interval], 'r-')
plt.plot(yolo2_obj[0::interval], 'y-')
plt.plot(yolo3_obj[0::interval], 'b-')

plt.plot(yolo1_bkg[0::interval], 'r--')
plt.plot(yolo2_bkg[0::interval], 'y--')
plt.plot(yolo3_bkg[0::interval], 'b--')

plt.show()