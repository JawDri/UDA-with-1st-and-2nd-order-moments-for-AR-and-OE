from torchvision.models import alexnet
import torch.nn as nn
import torch




class ResNet50Fc(nn.Module):
    def __init__(self):
        super(ResNet50Fc, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm1d(512)
        self.conv5 = nn.Conv1d(512, 1024, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU(inplace=True)
        self.__in_features = 9216########9216/32768

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #print('HEYEEEEEEEEEEEE',out.size())
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        #print('HEYEEEEEEEEEEEE',out.size())
        out = self.conv3(out)
        out = self.bn3(out)
        #print('HEYEEEEEEEEEEEE',out.size())
        out = self.conv4(out)
        out = self.bn4(out)
        #print('HEYEEEEEEEEEEEE',out.size())
        out = self.conv5(out)
        out = self.bn5(out)
        #print('HEYEEEEEEEEEEEE',out.size())
        
        out = self.relu(out)
        x = out.view(out.size(0), -1)
        return x
'''
    def output_num(self):
        return self.__in_features

'''



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # check https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
        self.features = nn.Sequential(nn.Conv1d(1, 64, kernel_size=1, bias=False),
        nn.BatchNorm1d(64),nn.Conv1d(64, 128, kernel_size=1, bias=False),
        nn.BatchNorm1d(128),nn.Conv1d(128, 256, kernel_size=1, bias=False),
        nn.BatchNorm1d(256),nn.Conv1d(256, 512, kernel_size=1, bias=False),
        nn.BatchNorm1d(512),nn.Conv1d(512, 1024, kernel_size=1, bias=False),
        nn.BatchNorm1d(1024))
        self.classifier = nn.Sequential(
            nn.Linear(9216, 512),#########9216/32768
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,10)
            )

        # if we want to feed 448x448 images
        # self.model.avgpool = nn.AdaptiveAvgPool2d(1)

        # In case we want to apply the loss to any other layer than the last
        # we need a forward hook on that layer
        # def save_features_layer_x(module, input, output):
        #     self.layer_x = output

        # This is a forward hook. Is executed each time forward is executed
        # self.model.layer4.register_forward_hook(save_features_layer_x)

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out  # , self.layer_x