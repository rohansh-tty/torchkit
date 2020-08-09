from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F



class BasicBlock(nn.Module):
    expansion = 1


    def __init__(self, in_planes, planes, stride = 1):
        """
        Arguments:
        in_planes: Input Channels
        planes: Outgoing Channels or Number of Kernels
        """
        super(BasicBlock, self).__init__()


        # Residual Block
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                                stride = stride, padding = 1, bias = False)
        self.bn1 = nn.Sequential(nn.BatchNorm2d(planes))
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, 
                                stride = stride, padding = 1, bias=False)
        self.bn2 = nn.Sequential(nn.BatchNorm2d(planes))

      
    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1))
        out = out2 + x
        out = F.relu(out)
        return out

# customNet Class
class customNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        """
        Arguments:-
        Block: Type of ResNet Block i.e Basic Block and Bottleneck 
        num_blocks: Number of blocks
        num_classes: Number of Classes in the Dataset
        """
        super(customNet,self).__init__()
        self.in_planes = 128
        
        # Prep Layer
        self.conv1 = self.create_conv2d(3,64, MaxPool=False)

        self.conv2 = self.create_conv2d(64, 128, MaxPool=True)

        # ResBlock 1
        self.res1 = self._make_layer(block, 128, num_blocks[0], stride = 1) 

        # Layer 2 
        self.conv3 = self.create_conv2d(128, 256, MaxPool=True)

        # Layer 3 
        self.conv4 = self.create_conv2d(256, 512, MaxPool=True)

        # chnage the in_planes from 128 to 512
        self.in_planes = 512
        # ResBlock 2
        self.res2 = self._make_layer(block, 512, num_blocks[1], stride = 1)
        
        self.pool4 = nn.MaxPool2d((4,4)) # MaxPool-4
        self.linear = nn.Linear(512, num_classes) # FCN


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes*block.expansion
        return nn.Sequential(*layers)

    def create_conv2d(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, MaxPool=False):
        
        if MaxPool:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                  nn.MaxPool2d((2,2)),
                                  nn.Sequential(nn.BatchNorm2d(out_channels), nn.ReLU()))
        else:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                    nn.Sequential(nn.BatchNorm2d(out_channels), nn.ReLU()))
        return self.conv

    
    def forward(self, x):
        """
        Function Variables:
        
        """
        # Prep Layer
        conv1x = self.conv1(x)

        # Layer 1
        conv2x = self.conv2(conv1x)
        
        # Res Block 1
        res1 = self.res1(conv2x)
        res1X = res1 + conv2x
        
        # Layer 2
        conv3x = self.conv3(res1X)
      
        # Layer 3 I guess
        conv4x = self.conv4(conv3x)
        
        # Res Block 2
        res2 = self.res2(conv4x)
        res2X = res2 + conv4x

        outX = self.pool4(res2X)
        outX = outX.view(outX.size(0), -1)
        outX = self.linear(outX)
        

        return F.log_softmax(outX)


def main11():
  return customNet(BasicBlock, [1,1])



