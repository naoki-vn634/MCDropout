import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from math import log2
from glob import glob

def calculate_entropy(pos):
    entropy = np.array([])
    
    for posterior in pos:
        ent = -sum([x*log2(x) if x != 0 else 0 for x in posterior])
        entropy = np.append(entropy,ent)    
    return entropy

def pre_result(cost_pre_bald, cost_pre_ent, pre_bald, pre_ent, output, rates):
    plt.figure()
    for i, rate in enumerate(rates):
        plt.plot(cost_pre_bald[i], pre_bald[i], label=f'bald_{rate}', alpha=0.5)
        plt.plot(cost_pre_ent[i], pre_ent[i], label=f'entropy_{rate}',alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(output, 'precision.png'))

def acc_result(cost_acc_bald, cost_acc_ent, acc_bald, acc_ent, output, rates):
    plt.figure()
    for i, rate in enumerate(rates):
        
        plt.plot(cost_acc_bald[i], acc_bald[i], label=f'bald_{rate}', alpha=0.5)
        plt.plot(cost_acc_ent[i], acc_ent[i], label=f'entropy_{rate}', alpha=0.5)
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
    
def posterior_transform(poses):
    for pos in poses:
        for i, posterior in enumerate(pos):
            if posterior>0.5:
                pos[i] = 1 - posterior
    return pos
    
def calculate_acc(label, preds, methods):
    acc = []
    cost = []
    acc_ = []
    cost_ = []
    for i, (pred, method) in enumerate(zip(preds, methods)):
        max = np.nanmax(method)
        min = np.nanmin(method)
        for thu in np.arange(max, min-0.02, -0.01):
            query = np.where(method>=thu)[0]
            cost.append(len(query))
            acc_child = len(np.where(pred[np.where(pred==label)[0]]==1)[0])
            acc_mother = len(np.where(pred==1)[0])  
            for j in query:
                if pred[j] != label[i]:
                    if pred[j] == 1:
                        acc_child += 1
                        
            acc.append(float(acc_child/acc_mother))

        cost_.append(np.array(cost))
        acc_.append(np.array(acc))

    return  cost_ ,acc_

def calculate_pre(label, preds, methods):
    pre = []
    cost = []
    pre_ = []
    cost_ = []
    for i, (pred, method) in enumerate(zip(preds, methods)):
        max = np.nanmax(method)
        min = np.nanmin(method)
        
        for thu in np.arange(max, min-0.5, -0.01):
            query = np.where(method>=thu)[0]
            cost.append(len(query))
            pre_child = len(np.where(label[np.where(pred==label)[0]]==1)[0])
            pre_mother = len(np.where(label==1)[0])
            for j in query:
                if pred[j] != label[j]:
                    if pred[j] == 0:
                        pre_child += 1
            
            pre.append(float(pre_child/pre_mother))
            
        cost_.append(np.array(cost))
        pre_.append(np.array(pre))

    return cost_, pre_


def main(args):
    
    cfg={
        'dr_rate': args.dr_rate,
        'n_drop': args.n_drop
    }
    
    
    label = np.load(os.path.join(args.input, 'y_true.npy'))
    rates = []
    
    rate_dirs = glob(os.path.join(args.input, '*/'))
    for i, rate_dir in enumerate(rate_dirs):
        rate = os.path.basename(os.path.dirname(rate_dir))
        rates.append(rate)
        print(rates)
        if i == 0:
            label_in_garbage = np.load(os.path.join(rate_dir, 'result/label_include_garbage.npy'))# 2:Garbage
            balds = np.expand_dims(np.load(os.path.join(rate_dir, 'result/{}_drops_bald.npy'.format(cfg['n_drop']))),axis=0)
            poses = np.expand_dims(np.load(os.path.join(rate_dir, 'result/{}_posterior.npy'.format(cfg['n_drop']))),axis=0)
            preds = np.expand_dims(np.load(os.path.join(rate_dir, 'result/{}_pred.npy'.format(cfg['n_drop']))),axis=0)
            entropies = np.expand_dims(calculate_entropy(np.load(os.path.join(rate_dir, 'result/{}_posterior.npy'.format(cfg['n_drop'])))),axis=0)
        else:
            pos = np.expand_dims(np.load(os.path.join(rate_dir, 'result/{}_posterior.npy'.format(cfg['n_drop']))),axis=0)
            poses = np.concatenate((poses, pos), axis=0)
            bald = np.expand_dims(np.load(os.path.join(rate_dir, 'result/{}_drops_bald.npy'.format(cfg['n_drop']))),axis=0)
            balds = np.concatenate((balds, bald), axis=0)
            pred = np.expand_dims(np.load(os.path.join(rate_dir, 'result/{}_pred.npy'.format(cfg['n_drop']))),axis=0)
            preds = np.concatenate((preds, pred), axis=0)
            entropy = np.expand_dims(calculate_entropy(np.load(os.path.join(rate_dir, 'result/{}_posterior.npy'.format(cfg['n_drop'])))),axis=0)
            entropies = np.concatenate((entropies, entropy), axis=0)
    
    print("##BALD: ", balds.shape)
    print('##POSTERIOR: ', poses.shape)
    print('##PREDS: ',preds.shape)
    print('##ENTROPIES', entropies.shape)    

    # Garbageとの分離
    output = args.output
    for i, rate in enumerate(rates):
        print(rate)
        save_distribution(label_in_garbage, balds[i], output, savename= f'bald_{rate}')
        save_distribution(label_in_garbage, entropies[i], output, savename= f'entropy_{rate}')
    
    #cost関数
    
    cost_acc_bald,acc_bald = calculate_acc(label, preds, balds)
    cost_pre_bald,pre_bald = calculate_pre(label, preds, balds)
    cost_acc_ent,acc_ent = calculate_acc(label, preds, entropies)
    cost_pre_ent,pre_ent = calculate_pre(label, preds, entropies)
        
    pre_result(cost_pre_bald, cost_pre_ent, pre_bald, pre_ent, output, rates)
    acc_result(cost_acc_bald, cost_acc_ent, acc_bald, acc_ent, output, rates)
    
    #Confusion Matrix
    for i, rate in enumerate(rates):
        conf_mat = confusion_matrix(label, preds[i])
        plt.figure()
        sns.heatmap(conf_mat, annot=True, cmap='Blues')
        plt.savefig(os.path.join(args.output, '{}_confusion_matrix.png'.format(rate)))
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comparison method')
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--dr_rate', type=float)
    parser.add_argument('--n_drop', type=int, default=10)
    args = parser.parse_args()
    
    main(args)