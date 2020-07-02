import torch
import torch.nn.functional as F

class BALD(Module):
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
                for inputs, labels, idxs in self.dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(device)
                    
                    out = self.net(inputs, labels)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()
                    
        return probs

            