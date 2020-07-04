import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import log2

def calculate_entropy(pos):
    entropy = np.array([])
    for posterior in pos:
        ent = -sum([x*log2(x) if x != 0 else 0 for x in posterior])
        entropy = np.append(entropy,ent)
        
    return entropy

def save_distribution(label, method, output, savename, hist=True):
    normal = method[np.where(label!=2)[0]]
    garbage = method[np.where(label==2)[0]]
    plt.figure()
    sns.distplot(normal, hist=hist, label='normal')
    sns.distplot(garbage, hist=hist, label='garbage')
    plt.title(savename + '_distribution')
    plt.legend()
    plt.savefig(os.path.join(output, (savename + '_distribution.png')))
    
    
        
def main(args):
    bald = np.load(os.path.join(args.input, 'BALD_0.3/100_drops_bald.npy'))
    label = np.load(os.path.join(args.input, 'DATA/y_true.npy'))
    pred = np.load(os.path.join(args.input, 'DATA/y_pred.npy'))
    label_in_garbage = np.load(os.path.join(args.input, 'DATA/label_include_garbage.npy'))# 2:Garbage
    pos = np.load(os.path.join(args.input, 'POSTERIOR/posterior_scaled.npy'))
    entropy = calculate_entropy(pos)
    
    # Garbageとの分離
    output = args.output
    print(bald.shape)
    print(pos[:,1].shape)
    print(entropy.shape)
    print(label_in_garbage.shape)
    save_distribution(label_in_garbage, bald, output, savename='bald')
    save_distribution(label_in_garbage, pos[:,1], output, savename='posterior')
    save_distribution(label_in_garbage, entropy, output, savename='entropy')
    
    #cost関数の計算
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comparison method')
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    
    main(args)