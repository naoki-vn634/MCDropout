import torch
import torch.nn.functional as F

class BALD(object):
    def __init__(self, dataloader, net, device, n_drop=10, n_cls=2):
        self.dataloader = dataloader
        self.net = net
        self.device = device
        self.n_drop = n_drop
        self.n_cls = n_cls
    
    def training(self):
        self.net.train()
        probability = torch.zeros([self.n_drop, len(self.dataloader.dataset), self.n_cls])
        
        for i in range(self.n_drop):
            print(f'Dropout:{i+1}/{self.n_drop}')
            with torch.no_grad():
                idxs = 0
                for inputs, labels in self.dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    out = self.net(inputs)
                    probability[i][idxs:idxs+out.size()[0]] += F.softmax(out, dim=1).cpu()
                    idxs += out.size()[0]
        
        return probability
    
    def evaluating(self, probs):
        pb = probs.mean(0)
        entropy1 = (-pb*torch.log(pb)).sum(1)
        entropy2 = (probs*torch.log(probs)).sum(2).mean(0)
        U = entropy2 + entropy1
        return U
        