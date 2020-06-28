import torch
import torch.nn as nn

# class CustomMonteCarloNet(nn.Module):
#     '''
#     Image_size : (3,224,224)
#     p : Probability of Dropout apply
#     '''
#     def __init__(self, num_classes=2, channel=3, hidden_size=16, mid_dim=1024, rate=0.5):
#         super(CustomMonteCarloNet, self).__init__()
#         self.conv1 = nn.Sequential(nn.Conv2d(channel, hidden_size,5,padding=1),
#                                    nn.Dropout(p=rate),
#                                    nn.MaxPool2d(2,2))#110
#         self.conv2 = nn.Sequential(nn.Conv2d(hidden_size, hidden_size*2,5,padding=1),
#                                    nn.Dropout(p=rate),
#                                    nn.MaxPool2d(2,2))#54
#         self.conv3 = nn.Sequential(nn.Conv2d(hidden_size*2,hidden_size*4,4,padding=1),
#                                    nn.Dropout(p=rate))#53
#         self.conv4 = nn.Sequential(nn.Conv2d(hidden_size*4,hidden_size*4,4,padding=1),
#                                    nn.Dropout(p=rate),
#                                    nn.MaxPool2d(2,2))#26
#         self.conv5 = nn.Sequential(nn.Conv2d(hidden_size*4,hidden_size*8,3,padding=1),
#                                    nn.Dropout(p=rate),
#                                    nn.MaxPool2d(2,2)))#13
#         self.conv6 = nn.Sequential(nn.Conv2d(hidden_size*8,hidden_size*8,4,padding=1),
#                                    nn.Dropout(p=rate),
#                                    nn.MaxPool2d(2,2))#6
#         self.fc1 = nn.Linear(6*6*hidden_size*8,mid_dim)
#         self.fc2 = nn.Linear(mid_dim, mid_dim/4)
#         self.fc3 = nn.Linear(mid_dim/4, num_classes)
        
#         self.dr = nn.Dropout(p=rate)
        

class CustomMonteCarloVAE(nn.Module):
    '''
    Image_size : (3,224,224)
    p : Probability of Dropout apply
    '''
    def __init__(self, config, num_classes=2, channel=3, rate=0.5, bn=False, init_weight=True):
        super(CustomMonteCarloVAE, self).__init__()
        
        self.features = self.make_layers(config, batch_norm=bn, in_channels=channel, p=rate)
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout2d(p=rate),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Dropout2d(p=rate),
            nn.Linear(4096,num_classes)
        )
        if init_weight:
            self.initialize_weights()

    def forward(self, x):
        h = self.features(x)
        h = self.avgpool(h)
        h = h.view(-1,512*7*7)
        y = self.classifier(h)

        return y
            


    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def make_layers(self,config, batch_norm, in_channels, p):
        layers = []
        for v in config:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.Dropout2d(p), nn.ReLU(inplace=True)]
                elif p=None:
                    layers = [conv2d, nn.ReLU(inplace=True)]                
                else:
                    layers += [conv2d, nn.Dropout2d(p), nn.ReLU(inplace=True)]
                
                in_channels = v
        return nn.Sequential(*layers)
    
