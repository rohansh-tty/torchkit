import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
  def __init__(self, config):
    super(Net, self).__init__()
    self.config = config
    self.ch_norm = self.config.channel_norm

  def convblock(self, in_ch=1, mid_ch=8, out_ch=16, kernel_=(3,3), padding_=[0,0], bias=False):#, ch_norm = self.ch_norm):
    if self.ch_norm == 'BatchNorm2d':
      _block = nn.Sequential(
                                    nn.Conv2d(in_channels=in_ch, out_channels=mid_ch, kernel_size=kernel_, padding=padding_[0], bias=False),
                                    nn.ReLU(), 
                                    nn.BatchNorm2d(mid_ch), 
                                    nn.Dropout(self.config.dropout_value),
                                    
                                    nn.Conv2d(in_channels=mid_ch, out_channels=out_ch, kernel_size=kernel_, padding=padding_[1],bias=False),
                                    nn.ReLU(), 
                                    nn.BatchNorm2d(out_ch),
                                    nn.Dropout(self.config.dropout_value)
      )
    if self.ch_norm == 'GroupNorm':
      _block = nn.Sequential(
                                  nn.Conv2d(in_channels=in_ch, out_channels=mid_ch, kernel_size=kernel_, padding=padding_[0], bias=False),
                                  nn.ReLU(), 
                                  nn.GroupNorm(2, mid_ch), 
                                  nn.Dropout(self.config.dropout_value),
                                   
                                  nn.Conv2d(in_channels=mid_ch, out_channels=out_ch, kernel_size=kernel_, padding=padding_[1],bias=False),
                                  nn.ReLU(), 
                                  nn.GroupNorm(2, out_ch),
                                  nn.Dropout(self.config.dropout_value)
    )
    if self.ch_norm == 'LayerNorm':
      _block = nn.Sequential(
                                  nn.Conv2d(in_channels=in_ch, out_channels=mid_ch, kernel_size=kernel_, padding=padding_[0], bias=False),
                                  nn.ReLU(), 
                                  nn.GroupNorm(1, mid_ch),
                                  nn.Dropout(self.config.dropout_value),
                                   
                                  nn.Conv2d(in_channels=mid_ch, out_channels=out_ch, kernel_size=kernel_, padding=padding_[1],bias=False),
                                  nn.ReLU(), 
                                  nn.GroupNorm(1, out_ch),
                                  nn.Dropout(self.config.dropout_value)
    )
    return _block
                                 
  


# 4 block model
class LinearNet(nn.Module):
    def __init__(self, config):
        super(LinearNet, self).__init__()
        self.config = config
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(config.dropout_value)                        
        )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(config.dropout_value)                       
        )
        self.transitionblock1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), bias=True),
            nn.ReLU()                     
        )

        # Maxpooling
        self.pool1 = nn.MaxPool2d(2, 2) 


        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(config.dropout_value)                        
        )
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=13, kernel_size=(3, 3), bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(13),
            nn.Dropout(config.dropout_value)                        
        )

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=13, out_channels=10, kernel_size=(3, 3), bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(config.dropout_value)                        
        )

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(config.dropout_value)                        
        )

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        )

        self.translinear = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=1, bias=True),
            )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.transitionblock1(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.gap(x)
        x = self.translinear(x)
        x = x.view(-1, 10)
        if self.config.loss_function == 'CrossEntropyLoss':
            return x
        elif self.config.loss_function == 'NLLoss':
            return F.log_softmax(x, dim=-1)
            
        
    
class Skeleton(nn.Module):
    def __init__(self, config):
        super(Skeleton, self).__init__()
        self.config = config
        self.conv1 = nn.Sequential(
                                  nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3)),   
                                #input - 28 OUtput - 26 RF - 3
                                   
                                  nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3)),   
                                    #input - 26 OUtput - 24 RF - 5
                                   
                                  nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3)),   
                                #input - 24 OUtput - 22 RF - 7
        )

       
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(
                                  nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(1,1)),   
                                    #input - 11 OUtput - 11 RF - 12
                                   
                                  nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3)),   
                                    #input - 11 OUtput - 9 RF - 16
                                   
                                  nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3)),   
                                 #input - 9 OUtput - 7 RF - 20
        )
        
        self.conv3 = nn.Conv2d(32, 64, 1) # input - 7 - 20
        
        self.fc1 = nn.Linear(64,32)
        self.fc2 = nn.Linear(32,10)

        self.gap = nn.AvgPool2d(7)

    def forward(self, x):
        conv1_op = self.conv1(x)
        pool1_op = self.pool1(conv1_op)
        conv2_op = self.conv2(pool1_op)
        gap_op = self.gap(conv2_op)
        conv3_op = self.conv3(gap_op)
    
        conv3_op = conv3_op.view(conv3_op.shape[0],-1)
        
        fc1_op = self.fc1(conv3_op)
        fc2_op = self.fc2(fc1_op)

        final_op = fc2_op.view(-1, 10)
        if self.config.loss_function == 'CrossEntropyLoss':
            return final_op
        elif self.config.loss_function == 'NLLoss':
            return F.log_softmax(final_op, dim=-1)
      
        
    
class BasicMNISTNet(nn.Module):
    def __init__(self, config):
        super(BasicMNISTNet, self).__init__()
        self.config = config
        self.conv1 = nn.Sequential(
                                  nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(8), #input - 28 OUtput - 26 RF - 3
                                   
                                  nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(16), #input - 26 OUtput - 24 RF - 5
                                   
                                  nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(32) #input - 24 OUtput - 22 RF - 7
        )


       
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(
                                  nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(1,1)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(8), #input - 11 OUtput - 11 RF - 12
                                   
                                  nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(16), #input - 11 OUtput - 9 RF - 16
                                   
                                  nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(32) #input - 9 OUtput - 7 RF - 20
        )
        
        self.conv3 = nn.Conv2d(32, 64, 1) # input - 7 - 20
        
        self.fc1 = nn.Linear(64,32)
        self.fc2 = nn.Linear(32,10)

        self.gap = nn.AvgPool2d(7)

    def forward(self, x):
        conv1_op = self.conv1(x)
        conv1_op = F.dropout(conv1_op, p=0.030)

        pool1_op = self.pool1(conv1_op)

        conv2_op = self.conv2(pool1_op)
        conv2_op = F.dropout(conv2_op, p=0.030)

        gap_op = self.gap(conv2_op)
        conv3_op = self.conv3(gap_op)
        
        conv3_op = conv3_op.view(conv3_op.shape[0],-1)
        
        fc1_op = self.fc1(conv3_op)
        fc2_op = self.fc2(fc1_op)

        final_op = fc2_op.view(-1, 10)
        if self.config.loss_function == 'CrossEntropyLoss':
            return final_op
        elif self.config.loss_function == 'NLLoss':
            return F.log_softmax(final_op, dim=-1)
      



class AvgMNISTNet(nn.Module):
    def __init__(self, config):
        super(AvgMNISTNet, self).__init__()
        self.config = config
        self.conv1 = nn.Sequential(
                                  nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(8), #input - 28 OUtput - 26 RF - 3
                                   
                                  nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(16), #input - 26 OUtput - 24 RF - 5
                                   
                                  nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(32) #input - 24 OUtput - 22 RF - 7
        )


       
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(
                                  nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(1,1)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(8), #input - 11 OUtput - 11 RF - 12
                                   
                                  nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(16), #input - 11 OUtput - 9 RF - 16
                                   
                                  nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(32) #input - 9 OUtput - 7 RF - 20
        )
        
        self.conv3 = nn.Conv2d(32, 64, 1) # input - 7 - 20
        
        self.fc1 = nn.Linear(64,32)
        self.fc2 = nn.Linear(32,10)

        self.gap = nn.AvgPool2d(7)

    def forward(self, x):
        conv1_op = self.conv1(x)
        conv1_op = F.dropout(conv1_op, p=0.030)

        pool1_op = self.pool1(conv1_op)

        conv2_op = self.conv2(pool1_op)
        conv2_op = F.dropout(conv2_op, p=0.030)

        gap_op = self.gap(conv2_op)
        conv3_op = self.conv3(gap_op)
        
        conv3_op = conv3_op.view(conv3_op.shape[0],-1)
        
        fc1_op = self.fc1(conv3_op)
        fc2_op = self.fc2(fc1_op)

        final_op = fc2_op.view(-1, 10)
        if self.config.loss_function == 'CrossEntropyLoss':
            return final_op
        elif self.config.loss_function == 'NLLoss':
            return F.log_softmax(final_op, dim=-1)
      



# Reached 99 in 7-8th Epoch
class DilatedNet(nn.Module):
    def __init__(self, config):
        super(DilatedNet, self).__init__()
        self.config = config
        self.conv1 = nn.Sequential(
                                  nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), padding=1),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(8), #input - 28 OUtput - 28 RF - 3
                                  nn.Dropout(config.dropout_value),
                                   
                                  nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3,3)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(12), #input - 28 OUtput - 26 RF - 5
                                  nn.Dropout(config.dropout_value),
                                   
                                  nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3,3), dilation=2),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(16), #input - 26 OUtput - 22 RF - 9
                                  nn.Dropout(config.dropout_value),
                                   
                             
        )


       
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(
                                  nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1,1)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(8), #input - 11 OUtput - 11 RF - 10
                                  nn.Dropout(config.dropout_value),
                                   
                                  nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3,3), dilation=2),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(12), #input - 11 OUtput - 7 RF - 18
                                  nn.Dropout(config.dropout_value),
                                   
                                  nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3,3)),
                                    #input - 7 OUtput - 5 RF - 22
                                   
                                  # nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(3,3)),
                                  # nn.ReLU(), 
                                  # nn.BatchNorm2d(24) #input - 7 OUtput - 5 RF - 22
        )
        
        self.conv3 = nn.Conv2d(16, 48, 1) # input - 1  output - 1 RF - 22s
        
        self.fc1 = nn.Linear(48,10)
        # self.fc2 = nn.Linear(32,10)

        self.gap = nn.AvgPool2d(5)

    def forward(self, x):
        conv1_op = self.conv1(x)
        # conv1_op = F.dropout(conv1_op, p=0.050)

        pool1_op = self.pool1(conv1_op)

        conv2_op = self.conv2(pool1_op)
        # conv2_op = F.dropout(conv2_op, p=0.050)

        gap_op = self.gap(conv2_op)
        conv3_op = self.conv3(gap_op)
        
        conv3_op = conv3_op.view(conv3_op.shape[0],-1)
        
        fc1_op = self.fc1(conv3_op)
        # fc2_op = self.fc2(fc1_op)

        final_op = fc1_op.view(-1, 10)
        if self.config.loss_function == 'CrossEntropyLoss':
            return final_op
        elif self.config.loss_function == 'NLLoss':
            return F.log_softmax(final_op, dim=-1)
      




class DropNet(nn.Module):
    def __init__(self, config):
        super(DropNet, self).__init__()
        self.config = config
        self.conv1 = nn.Sequential(
                                  nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), padding=1),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(8), #input - 28 OUtput - 28 RF - 3
                                   nn.Dropout(config.dropout_value),
                                   
                                  nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3,3)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(12), #input - 28 OUtput - 26 RF - 5
                                   nn.Dropout(config.dropout_value),
                                   
                                  nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3,3)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(16), #input - 26 OUtput - 24 RF - 7
                                   nn.Dropout(config.dropout_value)
        )


       
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(
                                  nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1,1)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(8), #input - 12 OUtput - 12 RF - 8
                                  nn.Dropout(config.dropout_value),
                                   
                                  nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3,3)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(12), #input - 12 OUtput - 10 RF - 12
                                   nn.Dropout(config.dropout_value),
                                   
                                  nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3,3)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(16),  #input - 10 OUtput - 8 RF - 16
                                   
                                   
                                  nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(3,3)),
                                  # nn.ReLU(), 
                                  # nn.BatchNorm2d(32) #input - 8 OUtput - 6 RF - 20
        )
        
        self.conv3 = nn.Conv2d(20, 32, 1) # input - 8 - 20
        
        self.fc1 = nn.Linear(32,10)
        # self.fc2 = nn.Linear(32,10)

        self.gap = nn.AvgPool2d(6)

    def forward(self, x):
        conv1_op = self.conv1(x)
        # conv1_op = F.dropout(conv1_op, p=0.050)

        pool1_op = self.pool1(conv1_op)

        conv2_op = self.conv2(pool1_op)
        # conv2_op = F.dropout(conv2_op, p=0.050)

        gap_op = self.gap(conv2_op)
        conv3_op = self.conv3(gap_op)
        
        conv3_op = conv3_op.view(conv3_op.shape[0],-1)
        
        fc1_op = self.fc1(conv3_op)
        # fc2_op = self.fc2(fc1_op)

        final_op = fc1_op.view(-1, 10)
        if self.config.loss_function == 'CrossEntropyLoss':
            return final_op
        elif self.config.loss_function == 'NLLoss':
            return F.log_softmax(final_op, dim=-1)
      



# increasing model capacity
class NonDilatedNet(nn.Module):
    def __init__(self, config):
        super(NonDilatedNet, self).__init__()
        self.config = config
        self.conv1 = nn.Sequential(
                                  nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), padding=1),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(8), #input - 28 OUtput - 28 RF - 3
                                   nn.Dropout(config.dropout_value),
                                   
                                  nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3,3), padding=1),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(10), #input - 28 OUtput - 28 RF - 5
                                   nn.Dropout(config.dropout_value),
                                   
                              
        )

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(
                                  nn.Conv2d(in_channels=10, out_channels=8, kernel_size=(1,1)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(8), #input - 14 OUtput - 14 RF - 6
                                  nn.Dropout(config.dropout_value),
                                   
                                  nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(3,3), padding=1),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(10), #input - 14 OUtput - 14 RF - 10
                                  nn.Dropout(config.dropout_value),
                                   
                                  nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3,3)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(12),  #input - 14 OUtput - 12 RF - 14
                                   nn.Dropout(config.dropout_value),
                                   
                                  nn.Conv2d(in_channels=12, out_channels=14, kernel_size=(3,3)),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(14), #input - 12 OUtput - 10 RF - 18
                                  nn.Dropout(config.dropout_value),

                                  nn.Conv2d(in_channels=14, out_channels=16, kernel_size=(3,3)),
                                   #input - 10 OUtput - 8 RF - 22
        
                                  
        )
        
        self.conv3 = nn.Conv2d(16, 32, 1) # input - 8 - 20
        
        self.fc1 = nn.Linear(32,10)
        # self.fc2 = nn.Linear(32,10)

        self.gap = nn.AvgPool2d(8)

    def forward(self, x):
        conv1_op = self.conv1(x)
       
        pool1_op = self.pool1(conv1_op)

        conv2_op = self.conv2(pool1_op)
       

        gap_op = self.gap(conv2_op)
        conv3_op = self.conv3(gap_op)
        
        conv3_op = conv3_op.view(conv3_op.shape[0],-1)
        
        fc1_op = self.fc1(conv3_op)
        # fc2_op = self.fc2(fc1_op)

        final_op = fc1_op.view(-1, 10)
        if self.config.loss_function == 'CrossEntropyLoss':
            return final_op
        elif self.config.loss_function == 'NLLoss':
            return F.log_softmax(final_op, dim=-1)
    

class NoFCNet2(Net):
    def __init__(self, config):
        super(NoFCNet2, self).__init__(config)
        self.config = config
        self.conv1 = self.convblock(in_ch=1, mid_ch=10, out_ch=16, kernel_=(3,3), padding_=[1,0])
       
        self.pool1 = nn.MaxPool2d(2, 2)
        self.transition1 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1,1), bias=False)

        self.conv2_1 = self.convblock(in_ch=8, mid_ch=10, out_ch=12, kernel_=(3,3))
        self.conv2_2 = self.convblock(in_ch=12, mid_ch=14, out_ch=16, kernel_=(3,3))

        self.conv3 = nn.Conv2d(16, 24, 1) # input - 8 - 18
        self.conv4 = nn.Conv2d(24, 10, 1) 
        
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        conv1_op = self.conv1(x)
        pool1_op = self.pool1(conv1_op)
        transition1_op = self.transition1(pool1_op)
        conv2_1op = self.conv2_1(transition1_op)
        conv2_2op = self.conv2_2(conv2_1op)


        conv3_op = self.conv3(conv2_2op)
        gap_op = self.gap(conv3_op)
        conv4_op = self.conv4(gap_op)
        final_op = conv4_op.view(conv4_op.shape[0],-1)

        if self.config.loss_function == 'CrossEntropyLoss':
            return final_op
        elif self.config.loss_function == 'NLLoss':
            return F.log_softmax(final_op, dim=-1)
