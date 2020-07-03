import os
import sys
import torch
import argparse
import torch.nn as nn
from glob import glob
from sklearn.model_selection import train_test_split
from distutils.util import strtobool


sys.path.append('../preprocess/')
from img_preprocess import ImageTransform, MonteCarloDataset

sys.path.append('../model/')
from model import CustomMonteCarloVGG

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
            else:
                label.append(0)

    x_train, x_test, y_train, y_test = train_test_split(img_path,label,test_size=0.25)
    
    # VGG16:A, 
    cfg ={'A': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}

    net = CustomMonteCarloVGG(config=cfg['A'],rate=0.5)
    net.load_state_dict(torch.load(args.weight))
    net.to(device)

    if args.multi_gpu:
        net = torch.nn.DataParallel(net)
        torch.backends.cudnn.benchmark = True
    

    test_dataset = MonteCarloDataset(x_test,y_test,transform=transforms,phase='test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, num_workers=0, shuffle=False)

    
    bald = BALD(test_dataloader, net, device, n_drop=10, n_cls=2)
    probs = bald.training()
    
    print(probs.size())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--multi_gpu', type=strtobool, default=False)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--weight', type=str)
    args = parser.parse_args()
    main(args)
    