import sys
import torch

sys.path.append('../model/')
from model import CustomMonteCarloVAE

# VGG16:A, 

cfg ={'A': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}

net = CustomMonteCarloVAE(config=cfg['A'])


inputs = torch.ones(16,3,224,224)
print(inputs.size())
out = net(inputs)
print(out.size())