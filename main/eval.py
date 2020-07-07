import os
import sys
import torch
import argparse
import pickle
import torch.nn as nn
import numpy as np

from glob import glob
from sklearn.model_selection import train_test_split
from distutils.util import strtobool


sys.path.append('../preprocess/')
from img_preprocess import ImageTransform, MonteCarloDataset

sys.path.append('../model/')
from model import CustomMonteCarloVGG, CustomMonteCarloDensenet

sys.path.append('../utils/')
from util import BALD


def main(args):
    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    print('#device: ',device)
    img_path = []
    label = []
    
    mean = (0.485,0.456,0.406)
    std = (0.229,0.224,0.225)
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    transforms = ImageTransform(mean,std)
    
    for dir in glob(os.path.join(args.input,"*")):
        paths = sorted(glob(os.path.join(dir,'*.jpg')))
        img_path.extend(paths)
        for img in paths:
            if 'yes' in img:
                label.append(1)
            elif 'no' in img:
                label.append(0)
            else:
                label.append(2)
    np.save(os.path.join(args.output, 'label_include_garbage.npy'),np.array(label))
    
    
    with open('/mnt/aoni02/matsunaga/ae/inputs/extracted_wrong.txt', 'rb') as f:
        wrong_img_path = pickle.load(f)
    mode = []
    for _ in img_path:
        if _ in wrong_img_path:
            mode.append(1)#誤分類
        else:
            mode.append(0)#正解
    mode_np = np.array(mode)

    print("##Image_Length: ",len(img_path))
    print("|-- right: ",mode.count(0))
    print("|-- wrong: ",mode.count(1))
    print("##Label_length: ",len(label))
    if args.model ==0: # VGG16
        cfg ={'A': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}
        net = CustomMonteCarloVGG(config=cfg['A'],rate=args.dr_rate)

    elif args.model == 1: #Densenet161
        net = CustomMonteCarloDensenet(pretrained=False,dr_rate=args.dr_rate)

    net.load_state_dict(torch.load(args.weight))
    net.to(device)

    if args.multi_gpu:
        net = torch.nn.DataParallel(net)
        torch.backends.cudnn.benchmark = True
    

    test_dataset = MonteCarloDataset(img_path, label, transform=transforms, phase='test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, num_workers=0, shuffle=False)

    
    bald = BALD(test_dataloader, net, device, n_drop=args.n_drop, n_cls=2)
    probs = bald.training()
    posterior = (probs.cpu().data.numpy()).mean(0)
    pred = []
    for pos in posterior:
        label = np.argmax(pos)
        pred.append(label)
 
    np.save(os.path.join(args.output,f'{args.n_drop}_pred.npy'),np.array(pred))
    np.save(os.path.join(args.output,f'{args.n_drop}_posterior.npy'),posterior)
    Bald = bald.evaluating(probs)
    

    Bald_np = Bald.cpu().data.numpy()
    np.save(os.path.join(args.output, 'mode.npy'), mode_np)
    np.save(os.path.join(args.output, f'{args.n_drop}_drops_bald.npy'), Bald_np)
    print(Bald.size())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate BALD(Bayesian ActiveLearning by Disagreement)')
    parser.add_argument('--model', type=int)
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--multi_gpu', type=strtobool, default=False)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--weight', type=str)
    parser.add_argument('--dr_rate',type=float)
    parser.add_argument('--n_drop', type=int, default=10)
    args = parser.parse_args()
    main(args)
    