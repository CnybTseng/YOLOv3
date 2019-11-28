import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', type=str, help='verbose path')
    parser.add_argument('--metric', '-m', type=str, help='metric name[CACC,CONF,BKGC,PREC,RC50,RC75,AIOU,ACAT,LBOX,LOBJ,LCLS,LBKG]')
    parser.add_argument('--interval', '-i', type=int, default=50, help='data drawing interval[50]')
    args = parser.parse_args()
    
    values = list()
    with open(args.verbose, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            segments = line.split()
            for segment in segments:
                if args.metric in segment:
                    value = float(segment.replace(f'{args.metric}:', ''))
                    values.append(value)
    
    if not values:
        print(f'not find metric:{args.metric}')
        sys.exit(0)
    
    num = len(values)
    if num % 3 != 0:
        print(f'incomplete metrics:{num}')
    
    step = 3 * args.interval
    plt.xlabel('batch index')
    plt.ylabel('{args.metric}')
    for i in range(3):
        plt.plot(values[i::step])
    
    plt.show()