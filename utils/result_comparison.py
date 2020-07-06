import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import log2
from glob import glob

def calculate_entropy(poses):
    entropy = np.array([])
    for pos in poses:
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
    plt.savefig(os.path.join(output, 'precision.png'))

def acc_result(cost_acc_bald,cost_acc_pos,cost_acc_ent,acc_bald,acc_pos,acc_ent,output):
    plt.figure()
    plt.plot(cost_acc_bald,acc_bald,label='bald')
    plt.plot(cost_acc_pos,acc_pos,label='posterior')
    plt.plot(cost_acc_ent,acc_ent,label='entropy')
    plt.legend()
    plt.savefig(os.path.join(output, 'accuracy.png'))
    
def save_distribution(label, method, output, savename, hist=True):
    normal = method[np.where(label!=2)[0]]
    garbage = method[np.where(label==2)[0]]
    plt.figure()
    sns.distplot(normal, hist=hist, label='normal')
    sns.distplot(garbage, hist=hist, label='garbage')
    plt.title(savename + '_distribution')
    plt.legend()
    plt.savefig(os.path.join(output, (savename + '_distribution.png')))
    
def posterior_transform(pos):
    for i, posterior in enumerate(pos):
        if posterior>0.5:
            pos[i] = 1 - posterior
    return pos
    
def calculate_acc(label, pred, method):
    acc = []
    cost = []
    max = np.max(method)
    min = np.min(method)
    for thu in np.arange(max, min-0.5, -0.01):
        query = np.where(method>=thu)[0]
        cost.append(len(query))
        acc_child = len(np.where(pred[np.where(pred==label)[0]]==1)[0])
        acc_mother = len(np.where(pred==1)[0])  
        for i in query:
            if pred[i] != label[i]:
                if pred[i] == 1:
                    acc_child += 1

        
        acc.append(float(acc_child/acc_mother))
    print(len(cost))
    print(len(acc))
    return np.array(cost), np.array(acc)

    
def calculate_pre(label, pred, method):
    pre = []
    cost = []
    max = np.max(method)
    min = np.min(method)
    
    for thu in np.arange(max, min-0.5, -0.01):
        query = np.where(method>=thu)[0]
        cost.append(len(query))
        pre_child = len(np.where(label[np.where(pred==label)[0]]==1)[0])
        pre_mother = len(np.where(label==1)[0])
        for i in query:
            if pred[i] != label[i]:
                if pred[i] == 0:
                    pre_child += 1
        
        pre.append(float(pre_child/pre_mother))
    print(len(cost))
    print(len(pre))
    return np.array(cost), np.array(pre)
    

    
        
def main(args):
    
    cfg={
        'dr_rate': args.dr_rate,
        'n_drop': args.n_drop
    }
    
    
    label = np.load(os.path.join(args.input, 'DATA/y_true.npy'))
    label_in_garbage = np.load(os.path.join(args.input, 'DATA/label_include_garbage.npy'))# 2:Garbage
    
    #BALD
    bald_dirs = glob(os.path.join(args.input, 'BALD/*/'))
    for i, bald_dir in enumerate(bald_dirs):
        if i == 0:
            balds = np.expand_dims(np.load(os.path.join(bald_dir, '{}_drops_bald.npy'.format(cfg['n_drop']))),axis=0)
        else:
            bald = np.expand_dims(np.load(os.path.join(bald_dir, '{}_drops_bald.npy'.format(cfg['n_drop']))),axis=0)
            balds = np.concatenate((balds, bald),axis=0)
        print(balds.shape)
             
    pos_dirs = glob(os.path.join(args.input, 'POSTERIOR/*/'))
    for i, pos_dir in enumerate(pos_dirs):
        if i == 0:
            poses = np.expand_dims(np.load(os.path.join(bald_dir, '{}_posterior_vgg.npy'.format(cfg['n_drop']))),axis=0)
        else:
            pos = np.expand_dims(np.load(os.path.join(bald_dir, '{}_posterior_vgg.npy'.format(cfg['n_drop']))),axis=0)
            poses = np.concatenate((poses, pos),axis=0)
        print(poses.shape)
    entropy = calculate_entropy(poses)
    print(entropy.shape)
    
    pred = np.load(os.path.join(args.input, 'DATA/y_pred_dense.npy'))
    
    
    
    # Garbageとの分離
    output = args.output
    save_distribution(label_in_garbage, balds, output, savename='bald')
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
    parser.add_argument('--dr_rate', type=float)
    parser.add_argument('--n_drop', type=int, default=10)
    args = parser.parse_args()
    
    main(args)