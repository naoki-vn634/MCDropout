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
def pre_result(cost_pre_bald, cost_pre_pos, cost_pre_ent, pre_bald,pre_pos,pre_ent,output):
    plt.figure()
    plt.plot(cost_pre_bald,pre_bald,label='bald')
    plt.plot(cost_pre_pos,pre_pos,label='posterior')
    plt.plot(cost_pre_ent,pre_ent,label='entropy')
    plt.legend()
    plt.savefig(os.path.join(output, 'precision_new.png'))

def acc_result(cost_acc_bald,cost_acc_pos,cost_acc_ent,acc_bald,acc_pos,acc_ent,output):
    plt.figure()
    plt.plot(cost_acc_bald,acc_bald,label='bald')
    plt.plot(cost_acc_pos,acc_pos,label='posterior')
    plt.plot(cost_acc_ent,acc_ent,label='entropy')
    plt.legend()
    plt.savefig(os.path.join(output, 'accuracy_new.png'))
    
def save_distribution(label, method, output, savename, hist=True):
    normal = method[np.where(label!=2)[0]]
    garbage = method[np.where(label==2)[0]]
    plt.figure()
    sns.distplot(normal, hist=hist, label='normal')
    sns.distplot(garbage, hist=hist, label='garbage')
    plt.title(savename + '_distribution')
    plt.legend()
    plt.savefig(os.path.join(output, (savename + '_distribution_new.png')))
    
def posterior_transform(pos):
    for i, posterior in enumerate(pos):
        if posterior>0.5:
            pos[i] = 1 - posterior
    return pos
    
def calculate_acc(label, pred, method):
    acc = []
    cost = []
    max = np.max(method)
    for thu in np.arange(max, 0, -0.01):
        query = np.where(method>=thu)[0]
        cost.append(len(query))
        acc_child = len(np.where(label[np.where(pred==1)[0]]==1)[0])
        acc_mother = len(np.where(pred==1)[0])  
        for i in query:
            if pred[i] != label[i]:
                if pred[i] == 1:
                    acc_child += 1

        
        acc.append(float(acc_child/acc_mother))
    return np.array(cost), np.array(acc)

    
def calculate_pre(label, pred, method):
    pre = []
    cost = []
    max = np.max(method)
    for thu in np.arange(max, 0, -0.01):
        query = np.where(method>=thu)[0]
        cost.append(len(query))
        pre_child = len(np.where(pred[np.where(label==1)[0]]==1)[0])
        pre_mother = len(np.where(label==1)[0])  
        for i in query:
            if pred[i] != label[i]:
                if pred[i] == 0:
                    pre_child += 1
        
        pre.append(float(pre_child/pre_mother))
    return np.array(cost), np.array(pre)
    

    
        
def main(args):
    bald = np.load(os.path.join(args.input, 'BALD_0.3/10_drops_bald_new.npy'))
    label = np.load(os.path.join(args.input, 'DATA/y_true.npy'))
    pred = np.load(os.path.join(args.input, 'DATA/y_pred.npy'))
    label_in_garbage = np.load(os.path.join(args.input, 'DATA/label_include_garbage.npy'))# 2:Garbage
    pos = np.load(os.path.join(args.input, 'POSTERIOR/posterior_scaled.npy'))
    entropy = calculate_entropy(pos)
    
    # Garbageとの分離
    output = args.output
    save_distribution(label_in_garbage, bald, output, savename='bald')
    save_distribution(label_in_garbage, entropy, output, savename='entropy')
    
    #cost関数
    pos_transformed =  posterior_transform(pos[:,1])
    
    cost_acc_bald,acc_bald = calculate_acc(label, pred, bald)
    cost_pre_bald,pre_bald = calculate_pre(label, pred, bald)
    cost_acc_pos,acc_pos = calculate_acc(label, pred, pos_transformed)
    cost_pre_pos,pre_pos = calculate_pre(label, pred, pos_transformed)
    cost_acc_ent,acc_ent = calculate_acc(label, pred, entropy)
    cost_pre_ent,pre_ent = calculate_pre(label, pred, entropy)
    
    pre_result(cost_pre_bald, cost_pre_pos, cost_pre_ent, pre_bald,pre_pos,pre_ent,output)
    acc_result(cost_acc_bald,cost_acc_pos,cost_acc_ent,acc_bald,acc_pos,acc_ent,output)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comparison method')
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    
    main(args)