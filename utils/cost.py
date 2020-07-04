import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def save_dist(BALD,args):
    plt.figure()
    sns.distplot(BALD)
    plt.xlabel('distributuion')
    plt.title('bald distribution')
    plt.savefig(os.path.join(args.output, f'{args.n_drop}_bald_distributuion.png'))

def save_cost(BALD, mode, args, pred, label):
    acc = []
    pre = []
    cost = []
    bald_max = np.max(BALD)
    for thu in np.arange(bald_max, 0, -0.01):
        query = np.where(BALD>=thu)[0]
        cost.append(len(query))
        acc_child = len(np.where(label[np.where(pred==1)[0]]==1)[0])
        acc_mother = len(np.where(pred==1)[0])  
        pre_child = len(np.where(pred[np.where(label==1)[0]]==1)[0])
        pre_mother = len(np.where(label==1)[0]) 
        for i in query:
            if pred[i] != label[i]:
                if pred[i] == 1:
                    acc_child += 1
                elif pred[i] == 0:
                    pre_child += 1
        
        acc.append(float(acc_child/acc_mother))
        pre.append(float(pre_child/pre_mother))

    
    plt.figure()
    plt.plot(np.array(cost),np.array(acc),label='accuracy')
    plt.plot(np.array(cost),np.array(pre),label='precision')
    plt.ylabel('acc / pre')
    plt.xlabel('cost')
    plt.title('BALD Efficiency')
    plt.savefig(os.path.join(args.output, f'{args.n_drop}_result.png'))
    
    
def main(args):
    mode = np.load(os.path.join(args.input, 'mode.npy'))
    BALD = np.load(os.path.join(args.input, f'{args.n_drop}_drops_bald.npy'))
    
    eval_pred = np.load(os.path.join(args.eval, 'y_pred.npy'))
    eval_label = np.load(os.path.join(args.eval, 'y_true.npy'))
    save_dist(BALD,args)
    save_cost(BALD, mode, args, eval_pred, eval_label)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='saving cost efficiency')
    parser.add_argument('--input', type=str)
    parser.add_argument('--eval', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--n_drop', type=int)
    args = parser.parse_args()
    
    main(args)