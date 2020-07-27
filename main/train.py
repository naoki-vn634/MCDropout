import os
import re
import sys
import torch
import argparse
import pickle
import torch.nn as nn
from glob import glob
from sklearn.model_selection import train_test_split
from distutils.util import strtobool


sys.path.append('../preprocess/')
from img_preprocess import ImageTransform, MonteCarloDataset

sys.path.append('../model/')
from model import CustomMonteCarloVGG, DropoutDenseNet, CustomMonteCarloDensenet


def train(net, dataloaders_dict, output, num_epoch, optimizer, criterion, device, tfboard):

    Loss = {'train':[0]*num_epoch, 'test':[0]*num_epoch}
    Acc  = {'train':[0]*num_epoch, 'test':[0]*num_epoch}
    
    if args.tfboard:
        print('Using Tensorboard')
    from torch.utils.tensorboard import SummaryWriter
    save_dir = os.path.join(output, 'tfboard')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    tblogger = SummaryWriter(save_dir)
    
    phases = ['train', 'test']
    for epoch in range(num_epoch):
        for phase in phases:
            epoch_loss = 0
            epoch_correct = 0
            
            if (epoch==0)and(phase=='train'):
                continue
            print('---------')
            print(f'Epoch:{epoch+1}/{num_epoch}')
            print(f'Phase:{phase}')
            if phase == 'train':
                net.train()
                torch.set_grad_enabled(True)
            else:
                torch.set_grad_enabled(False)
                
            
            for inputs, labels in dataloaders_dict[phase]:
                optimizer.zero_grad()
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                out = net(inputs)
                _,preds = torch.max(out, 1) 
                loss = criterion(out,labels)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                epoch_loss += float(loss.item())*inputs.size(0)
                epoch_correct += torch.sum(preds == labels.data)
            
            epoch_loss = epoch_loss/len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_correct.double()/len(dataloaders_dict[phase].dataset)

            Loss[phase][epoch] = epoch_loss
            Acc[phase][epoch] = epoch_acc
            print('{} Loss:{:.4f} Acc:{:.4f}'.format(phase,epoch_loss,epoch_acc))
            
            if tfboard:
                tblogger.add_scalar(f'{phase}/loss',epoch_loss,epoch)

        if epoch == 0:
                continue
        elif epoch == 1:
            best_acc_epoch = epoch
            best_loss_epoch = epoch
            best_acc = epoch_acc
            best_loss = epoch_loss
            torch.save(net.state_dict(),os.path.join(output,'best_acc.pth'))
            torch.save(net.state_dict(),os.path.join(output,'best_loss.pth'))
        elif best_acc < epoch_acc:
            best_acc_epoch = epoch
            best_acc = epoch_acc
            torch.save(net.state_dict(),os.path.join(output,'best_acc.pth'))
            
        elif best_loss > epoch_loss:
            best_loss_epoch = epoch
            best_loss = epoch_loss
            torch.save(net.state_dict(),os.path.join(output,'best_loss.pth'))
    
    
    best_acc_weight = os.path.join(output,'best_acc.pth')
    best_loss_weight = os.path.join(output,'best_loss.pth')
    
    rename_acc_weight = os.path.join(output,'epoch_{}_loss_{:.3f}_acc_{:.3f}_best_acc.pth'.format(best_acc_epoch,Loss['test'][best_acc_epoch],Acc['test'][best_acc_epoch]))
    rename_loss_weight = os.path.join(output,'epoch_{}_loss_{:.3f}_acc_{:.3f}_best_loss.pth'.format(best_acc_epoch,Loss['test'][best_loss_epoch],Acc['test'][best_loss_epoch]))
              
    os.rename(best_acc_weight,rename_acc_weight)
    os.rename(best_loss_weight,rename_loss_weight)
    
    tblogger.close()
    
    
def main(args):
    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    print('#device: ',device)
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    
    img_path = []
    label = []
    
    mean = (0.485,0.456,0.406)
    std = (0.229,0.224,0.225)
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    transforms = ImageTransform(mean,std)
    
    if args.txt:     
        with open(args.txt, 'rb') as f:
            img_path = pickle.load(f)
    else:
        for dir in glob(os.path.join(args.input,"*")):
            paths = sorted(glob(os.path.join(dir,'*.jpg')))
            img_path.extend(paths)
        
    for img in img_path:
        if 'yes' in img:
            label.append(1)
        elif 'no' in img:
            label.append(0)
        elif 'garbage' in img:
            label.append(2)
            
    x_train, x_test, y_train, y_test = train_test_split(img_path,label,test_size=0.25)
    if args.model ==0: # VGG16
        cfg ={'A': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}
        
        print(type(args.n_cls))
        print(args.n_cls)
        net = CustomMonteCarloVGG(config=cfg['A'],rate=args.dr_rate, all_layer=False, num_classes=args.n_cls)
 
    
    elif args.model == 1: #Densenet161
        net = CustomMonteCarloDensenet(pretrained=True, dr_rate=args.dr_rate, num_classes=args.n_cls)
    
    net.to(device)  

    if args.multi_gpu:
        net = torch.nn.DataParallel(net)
        torch.backends.cudnn.benchmark = True
    
    for name, param in net.named_parameters():
        if args.model == 0:
            param.require_grad = True
        elif args.model == 1:
            if 'fc' in name:
                param.require_grad = True
            else:
                param.require_grad = False
                
    train_dataset = MonteCarloDataset(x_train, y_train, transform=transforms, phase='train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batchsize,num_workers=0, shuffle=True)
    
    test_dataset = MonteCarloDataset(x_test,y_test,transform=transforms,phase='test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, num_workers=0, shuffle=False)

    print("Train_Length: ",len(train_dataloader.dataset))
    print("Test_Length: ",len(test_dataloader.dataset))
    
    dataloaders_dict = {
        'train':train_dataloader,
        'test':test_dataloader
    }
    
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    train(net, dataloaders_dict, output=args.output, num_epoch=args.epoch, optimizer=optimizer, criterion=criterion, device=device, tfboard=args.tfboard)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt', type=str)
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--multi_gpu', type=strtobool, default=False)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--tfboard', type=strtobool, default=False)
    parser.add_argument('--dr_rate', type=float)
    parser.add_argument('--model', type=int)
    parser.add_argument('--n_cls', type=int)
    
    args = parser.parse_args()
    main(args)
    