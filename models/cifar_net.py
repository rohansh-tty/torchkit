import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        self.ch_norm = self.config.channel_norm

    def convblock(self, in_ch=1, mid_ch=8, out_ch=16, kernel_=(3,3), padding_=[0,0], dilation_=[1,1], bias=False):#, ch_norm = self.ch_norm):
        if self.ch_norm == 'BatchNorm2d':
          _block = nn.Sequential(
                                  nn.Conv2d(in_channels=in_ch, out_channels=mid_ch, kernel_size=kernel_, padding=padding_[0], dilation=dilation_[0], bias=False),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(mid_ch), 
                                  nn.Dropout(self.config.dropout_value),
                                  
                                  nn.Conv2d(in_channels=mid_ch, out_channels=out_ch, kernel_size=kernel_, padding=padding_[1],dilation=dilation_[1], bias=False),
                                  nn.ReLU(), 
                                  nn.BatchNorm2d(out_ch),
                                  nn.Dropout(self.config.dropout_value)
            )
        if self.ch_norm == 'GroupNorm':
            _block = nn.Sequential(
                                    nn.Conv2d(in_channels=in_ch, out_channels=mid_ch, kernel_size=kernel_, padding=padding_[0], dilation=dilation_[0], bias=False),
                                    nn.ReLU(), 
                                    nn.GroupNorm(2, mid_ch), 
                                    nn.Dropout(self.config.dropout_value),
                                    
                                    nn.Conv2d(in_channels=mid_ch, out_channels=out_ch, kernel_size=kernel_, padding=padding_[1],dilation=dilation_[1], bias=False),
                                    nn.ReLU(), 
                                    nn.GroupNorm(2, out_ch),
                                    nn.Dropout(self.config.dropout_value)
            )
        if self.ch_norm == 'LayerNorm':
            _block = nn.Sequential(
                                    nn.Conv2d(in_channels=in_ch, out_channels=mid_ch, kernel_size=kernel_, padding=padding_[0], dilation=dilation_[0], bias=False),
                                    nn.ReLU(), 
                                    nn.GroupNorm(1, mid_ch),
                                    nn.Dropout(self.config.dropout_value),
                                    
                                    nn.Conv2d(in_channels=mid_ch, out_channels=out_ch, kernel_size=kernel_, padding=padding_[1],dilation=dilation_[1], bias=False),
                                    nn.ReLU(), 
                                    nn.GroupNorm(1, out_ch),
                                    nn.Dropout(self.config.dropout_value)
            )
        return _block

    def single_convblock(self, in_ch=1, out_ch=16, kernel_=(3,3), padding_=0, bias_=False):
      if self.ch_norm == 'BatchNorm2d':
            _block = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_, padding=padding_, bias=bias_),
                                    nn.ReLU(), 
                                    nn.BatchNorm2d(out_ch), 
                                    nn.Dropout(self.config.dropout_value))
            
      if self.ch_norm == 'GroupNorm':
          _block = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_, padding=padding_, bias=bias_),
                                  nn.ReLU(), 
                                  nn.GroupNorm(2, out_ch), 
                                  nn.Dropout(self.config.dropout_value))
          
      if self.ch_norm == 'LayerNorm':
            _block = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_, padding=padding_, bias=bias_),
                                  nn.ReLU(), 
                                  nn.GroupNorm(1, out_ch),
                                  nn.Dropout(self.config.dropout_value))
      return _block

    
    def transition_block(self, in_ch=0, out_ch=10, kernel_=(2,2), stride_value=2, padding_=[1,1], dilation_=[1,1], bias_=False):
      _block = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=kernel_, padding=padding_[0], dilation=dilation_[0], stride=stride_value, bias=bias_),
                              nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=(1,1), padding=padding_[1], dilation=dilation_[1], bias=False))
      return _block


    def depthwise_conv(self, in_ch=0, out_ch=10, padding_=0, bias_=False):
      _block = nn.Sequential(nn.Conv2d(in_ch, in_ch, 3, padding_, bias_), # Depthwise
                               nn.Conv2d(in_ch, out_ch, 1, padding_, bias_),
                               nn.ReLU(), 
                              nn.BatchNorm2d(out_ch),
                              nn.Dropout(self.config.dropout_value)) # Pointwise    
      return _block


  

  


# 25th Epoch 75%, 1st epoch 49-50%
class CifarNet(Net):
    def __init__(self, config):
        super(CifarNet, self).__init__(config)
        # self.conv0 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3,3), padding=1)
        self.conv1 = self.convblock(in_ch=3, mid_ch=16, out_ch=32, padding_=[2,2], dilation_=[2,2]) # RF 5x5
        self.transition1 = self.transition_block(in_ch=32, out_ch=16, kernel_=(3,3), stride_value=2, dilation_=[2,1], padding_=[2,0]) # RF 
        
        self.conv2 = self.convblock(in_ch=16, mid_ch=32, out_ch=64, padding_=[2,2], dilation_=[2,2]) # RF
        self.transition2 = self.transition_block(in_ch=64, out_ch=16, kernel_=(3,3), stride_value=2, dilation_=[1,1], padding_=[2,0]) # RF

        self.conv3 = self.convblock(in_ch=16, mid_ch=32, out_ch=64, padding_=[2,1], dilation_=[2,1]) # RF
        self.transition3 = self.transition_block(in_ch=64, out_ch=16, kernel_=(3,3), stride_value=2, dilation_=[1,1],padding_=[1,0]) # RF

        # self.conv4 = self.convblock(in_ch=16, mid_ch=32, out_ch=64, padding_=[1,1], dilation_=[1,1]) # RF
        self.conv4 = self.single_convblock(in_ch=16, out_ch=32, kernel_=(1,1))
        # self.transition4 = self.transition_block(in_ch=64, out_ch=16, kernel_=(3,3), stride_value=2) # RF


        # self.conv2 = self.convblock(in_ch=16, mid_ch=32, out_ch=64)

        self.gap = nn.AvgPool2d(5)

        self.last_conv = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1,1))
        self.fc = self.single_convblock(in_ch=64, out_ch=10, kernel_=(1,1))


    def forward(self, ip):
        # conv0 = self.conv0(ip)
        conv1x = self.conv1(ip)
        # print('conv1x.shape', conv1x.shape)

        transition1x = self.transition1(conv1x)
        # print('transition1x.shape', transition1x.shape)

        conv2x = self.conv2(transition1x)
        # print('conv2x.shape', conv2x.shape)

        transition2x = self.transition2(conv2x)
        # print('transition2x.shape', transition2x.shape)

        conv3x = self.conv3(transition2x)
        # print('conv3x.shape', conv3x.shape)

        transition3x = self.transition3(conv3x)
        # print('transition3x.shape', transition3x.shape)
        
        conv4x = self.conv4(transition3x)
        # print('conv4x.shape', conv4x.shape)
        # print('conv2x.shape', conv2x.shape)
        gap = self.gap(conv4x)
        # # print('gap.shape',gap.shape)
        # fc = self.fc(gap)
        last_conv = self.last_conv(gap)
        # # print('fc.shape',fc.shape)

        final_op = last_conv.view(last_conv.shape[0],-1)

        if self.config.loss_function == 'CrossEntropyLoss':
            return final_op
        elif self.config.loss_function == 'NLLoss':
            return F.log_softmax(final_op, dim=-1)




class CifarNet2(Net):
    def __init__(self, config):
        super(CifarNet2, self).__init__(config)

        self.conv1 = self.convblock(in_ch=3, mid_ch=32, out_ch=48, padding_=[2,2], dilation_=[2,1]) # RF 5x5
        self.transition1 = self.transition_block(in_ch=48, out_ch=16, kernel_=(3,3), stride_value=2, dilation_=[2,1], padding_=[2,0]) # RF 
        
        self.conv2 = self.convblock(in_ch=16, mid_ch=32, out_ch=48, padding_=[2,2], dilation_=[2,2]) # RF
        self.transition2 = self.transition_block(in_ch=48, out_ch=16, kernel_=(3,3), stride_value=2, dilation_=[1,1], padding_=[2,0]) # RF

        self.conv3 = self.convblock(in_ch=16, mid_ch=32, out_ch=64, padding_=[1,1], dilation_=[1,1]) # RF
        self.transition3 = self.transition_block(in_ch=64, out_ch=32, kernel_=(3,3), stride_value=2, dilation_=[1,1],padding_=[1,0]) # RF

        self.conv4 = self.single_convblock(in_ch=32, out_ch=32, kernel_=(1,1))

        self.gap = nn.AvgPool2d(5)

        self.last_conv = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1,1))
        self.fc = self.single_convblock(in_ch=64, out_ch=10, kernel_=(1,1))


    def forward(self, ip):
        conv1x = self.conv1(ip)
        # print('conv1x.shape', conv1x.shape)

        transition1x = self.transition1(conv1x)
        # print('transition1x.shape', transition1x.shape)

        conv2x = self.conv2(transition1x)
        # print('conv2x.shape', conv2x.shape)

        transition2x = self.transition2(conv2x)
        # print('transition2x.shape', transition2x.shape)

        conv3x = self.conv3(transition2x)
        # print('conv3x.shape', conv3x.shape)

        transition3x = self.transition3(conv3x)
        # print('transition3x.shape', transition3x.shape)
        
        conv4x = self.conv4(transition3x)
        # print('conv4x.shape', conv4x.shape)
        # print('conv2x.shape', conv2x.shape)
        gap = self.gap(conv4x)
        # # print('gap.shape',gap.shape)
        # fc = self.fc(gap)
        last_conv = self.last_conv(gap)
        # # print('fc.shape',fc.shape)

        final_op = last_conv.view(last_conv.shape[0],-1)

        if self.config.loss_function == 'CrossEntropyLoss':
            return final_op
        elif self.config.loss_function == 'NLLoss':
            return F.log_softmax(final_op, dim=-1)



class SeaFar(Net):
    def __init__(self, config):
        super(SeaFar, self).__init__(config)


        self.conv1_0 = nn.Conv2d(3, 32, 3, padding=2, dilation=2)
        self.conv1_1 = self.depthwise_conv(32, 64, padding_=1)

        self.transition1 = self.transition_block(in_ch=64, out_ch=32, kernel_=(3,3), stride_value=2, dilation_=[1,1], padding_=[1,0]) # RF 
        
        self.conv2_0 = nn.Conv2d(32, 48, 3, padding=2, dilation=2)
        self.conv2_1 = self.depthwise_conv(48, 64, padding_=1)

        self.transition2 = self.transition_block(in_ch=64, out_ch=32, kernel_=(3,3), stride_value=2, dilation_=[1,1], padding_=[1,0]) # RF

        self.conv3 = self.convblock(in_ch=32, mid_ch=32, out_ch=48, padding_=[1,1], dilation_=[1,1]) # RF
        self.transition3 = self.transition_block(in_ch=48, out_ch=24, kernel_=(3,3), stride_value=2, dilation_=[1,1],padding_=[1,0]) # RF

        self.conv4 = self.single_convblock(in_ch=24, out_ch=32, kernel_=(1,1))

        self.gap = nn.AvgPool2d(3)

        self.last_conv = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1,1))
        self.fc = self.single_convblock(in_ch=64, out_ch=10, kernel_=(1,1))


    def forward(self, ip):
        conv1_0x = self.conv1_0(ip)
        conv1_1x = self.conv1_1(conv1_0x)
        transition1x = self.transition1(conv1_1x)

        conv2_0x = self.conv2_0(transition1x)
        conv2_1x = self.conv2_1(conv2_0x)
        transition2x = self.transition2(conv2_1x)

        conv3x = self.conv3(transition2x)
        transition3x = self.transition3(conv3x)
        
        conv4x = self.conv4(transition3x)
        gap = self.gap(conv4x)
        last_conv = self.last_conv(gap)
        final_op = last_conv.view(last_conv.shape[0],-1)

        if self.config.loss_function == 'CrossEntropyLoss':
            return final_op
        elif self.config.loss_function == 'NLLoss':
            return F.log_softmax(final_op, dim=-1)



class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        ## Convolution Block1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias = False),  # Input: 32x32x3 | Output: 32x32x32 | RF: 3x3
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(self.config.dropout_value),

            nn.Conv2d(32, 64, 3, padding=1, bias = False), # Input: 32x32x32 | Output: 32x32x64 | RF: 5x5
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(self.config.dropout_value)
        )
        
        ## Transition Block1
        self.trans1 = nn.Sequential(
            nn.Conv2d(64, 32,3, stride=2), # Input: 32x32x64 | Output: 16x16x32 | RF: 5x5
            nn.ReLU(),
        )

        ## Convolution Block2
        self.conv2 =  nn.Sequential(
            nn.Conv2d(32, 64, 3,  padding=1,  dilation=2, bias = False), # Input: 16x16x32 | Output: 16x16x32 | RF: 5x5
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(self.config.dropout_value),

            ## Depthwise Seperable Convolution1
            nn.Conv2d(64,32, 3,  padding=1,groups=32 ,bias = False),  # Input: 16x16x32 | Output: 16x16x32 | RF: 9x9
            nn.Conv2d(32, 64, 1, padding=1, bias = False),   # Input: 16x16x32 | Output: 18x18x64 | RF: 13x13
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(self.config.dropout_value),
        )
        
        #Transition Block2
        self.trans2 = nn.Sequential(

            nn.Conv2d(64, 32, 3, stride=2), # Input: 18x18x32 | Output: 9x9x64 | RF: 17x17
            nn.ReLU()
        )

        #Convolution Block3
        self.conv3 = nn.Sequential(
            
            ## Dilation Block
            nn.Conv2d(32, 64, 3,  padding=1, bias = False), # Input: 9x9x64 | Output: 7x7x64 | RF: 33x33
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(self.config.dropout_value),

            nn.Conv2d(64, 64, 3,  padding=1, bias = False),  # Input: 7x7x64| Output: 7x7x64 | RF: 41x41
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(self.config.dropout_value),
        )

        #Transition Block3
        self.trans3 = nn.Sequential(

            nn.Conv2d(64, 16, 3, stride=2, padding=1), # Input: 7x7x64| Output: 4x4x16 | RF: 49x49
            nn.ReLU()
        )

        #Convolution Block4        
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias = False), # Input: 4x4x16 | Output: 4x4x32 | RF: 65x65
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(self.config.dropout_value),

            ## Depthwise seperable Convolution2
            nn.Conv2d(32,32, 3,bias = False),# Input: 4x4x16 | Output: 4x4x32 | RF: 97x97
            nn.Conv2d(32, 10, 1,  bias = False),          # Input: 4x4x32| Output: 6x6x10 | RF: 97x97
            nn.ReLU(),
            nn.BatchNorm2d(10),
            # nn.Dropout2d(self.config.dropout_value),
        )

        ## Output Block
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        ) 


    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(x)

        x = self.conv2(x) 
        x = self.trans2(x) 

        x = self.conv3(x) 
        x = self.trans3(x)

        x = self.conv4(x)
        x = self.gap(x)

        final_op = x.view(-1,10)
        if self.config.loss_function == 'CrossEntropyLoss':
            return final_op
        elif self.config.loss_function == 'NLLoss':
            return F.log_softmax(final_op, dim=-1)





class Net2(nn.Module):
    def __init__(self, config):
        super(Net2, self).__init__()
        self.config = config
        ## Convolution Block1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias = False),  # Input: 32x32x3 | Output: 32x32x32 | RF: 3x3
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(self.config.dropout_value),

            nn.Conv2d(32, 48, 3, padding=1, bias = False), # Input: 32x32x32 | Output: 32x32x48 | RF: 5x5
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout2d(self.config.dropout_value),

            ## Depthwise Seperable Convolution1
            nn.Conv2d(48, 48, 3, padding=1, bias = False),  # Input: 32x32x48 | Output: 32x32x48 | RF: 7x7
            nn.Conv2d(48, 64, 1, bias = False),   # Input: 32x32x48 | Output: 32x32x64 | RF: 7x7
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(self.config.dropout_value)
        )
        
        ## Transition Block1
        self.trans1 = nn.Sequential(
            nn.Conv2d(64, 16,3, stride=2), # Input: 32x32x64 | Output: 15x15x32 | RF: 9x9
            nn.ReLU(),
        )

        ## Convolution Block2
        self.conv2 =  nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias = False), # Input: 15x15x32 | Output: 15x15x32 | RF: 13x13
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(self.config.dropout_value),

            nn.Conv2d(32, 64, 3, padding=1 ,bias = False),  # Input: 15x15x32 | Output: 15x15x64 | RF: 17x17
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(self.config.dropout_value),
        )
        
        #Transition Block2
        self.trans2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=2), # Input: 15x15x64 | Output: 7x7x32 | RF: 21x21
            nn.ReLU()
        )

        #Convolution Block3
        self.conv3 = nn.Sequential(
            
            ## Dilation Block
            nn.Conv2d(32, 64, 3, dilation=2, padding=1 ,bias = False), # Input: 7x7x32 | Output: 5x5x64 | RF: 37x37
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(self.config.dropout_value),

            nn.Conv2d(64, 64, 3,  padding=1 ,bias = False),  # Input: 5x5x64| Output: 5x5x64 | RF: 45x45
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(self.config.dropout_value),
        )

        #Transition Block3
        self.trans3 = nn.Sequential(

            nn.Conv2d(64, 16, 3, stride=2, padding=1), # Input: 5x5x64| Output: 5x5x16 | RF: 53x53
            nn.ReLU()
        )

        #Convolution Block4        
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias = False), # Input: 5x5x16 | Output:  5x5x32 | RF: 69x69
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(self.config.dropout_value),

            ## Depthwise seperable Convolution2
            nn.Conv2d(32,32, 3,bias = False),# Input:  5x5x32 | Output: 3x3x32 | RF: 69x69
            nn.Conv2d(32, 10, 1,  bias = False),          # Input: 3x3x32| Output: 3x3x10 | RF: 85x85
            nn.ReLU(),
            nn.BatchNorm2d(10),
            # nn.Dropout2d(self.config.dropout_value),
        )

        ## Output Block
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)
        ) 


    def forward(self, x):
        x = self.conv1(x)
        x = self.trans1(x)

        x = self.conv2(x) 
        x = self.trans2(x) 

        x = self.conv3(x) 
        x = self.trans3(x)

        x = self.conv4(x)
        x = self.gap(x)

        final_op = x.view(-1,10)
        if self.config.loss_function == 'CrossEntropyLoss':
            return final_op
        elif self.config.loss_function == 'NLLoss':
            return F.log_softmax(final_op, dim=-1)
