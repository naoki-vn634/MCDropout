import cv2
import torch
import torchvision
import torch.nn as nn 
import torchvision.transforms as transforms


class ImageTransform(object):
    def __init__(self,mean,std):
        self.transform = {
            'train':transforms.Compose([
                # transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean,std),
            ]),
            'test':transforms.Compose([
                # transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean,std),
            ]),
        }
        
    def __call__(self,image,phase):
        img_transformed =  self.transform[phase](image)
        return img_transformed


class MonteCarloDataset(object):
    
    def __init__(self,file_list,label_list,transform,phase):
        self.file_list = file_list
        self.label_list = label_list
        self.transform = transform
        self.phase = phase
                
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self,index):
        image = self.file_list[index]
        img = cv2.imread(image)
        img_transformed = self.transform(img,self.phase)
        
        return img_transformed, int(self.label_list[index])
