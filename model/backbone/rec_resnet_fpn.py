# -*- coding: utf-8 -*-
# rec_resnet_fpn.py
# author lm

import torch
import torch.nn as nn
import torch.nn.functional as F


def pair(x):
    if isinstance(x, int):
        return (x, x)
    return x 

class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 act=None):
        super(ConvBNLayer, self).__init__()
        # stride = pair(stride)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size= 2 if stride == (1, 1) else kernel_size,
            dilation= 2 if stride == (1, 1) else 1,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias=False)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act 
        if self.act is not None:
            self.act = nn.ReLU(inplace=True)
            
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x 
  

class ShortCut(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 is_first=False):
        super(ShortCut, self).__init__()  
        self.conv = None
        if in_channels != out_channels or stride != 1 or is_first is True:
            if stride == (1, 1):
                self.conv = ConvBNLayer(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride = 1)
            else:
                self.conv = ConvBNLayer(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride = stride)

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        return x 
        
    
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 is_first=False):
        super(BasicBlock, self).__init__()
        self.conv0 = ConvBNLayer(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 3,
            stride = stride,
            act = 'relu')
        self.conv1 = ConvBNLayer(
            in_channels = out_channels,
            out_channels = out_channels,
            kernel_size = 3,
            stride = 1,
            act=None)
        self.short = ShortCut(
            in_channels = in_channels,
            out_channels = out_channels,
            stride = stride,
            is_first = is_first)
        self.out_channels = out_channels 
    
    def forward(self, x):
        y = self.conv0(x)
        y = self.conv1(y)
        y += self.short(x)
        y = F.relu(y)
        return y 
        
class BottleneckBlock(nn.Module):
    expansion = 4
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 **kwargs):
        super(BottleneckBlock, self).__init__()
        # squeeze
        self.conv0 = ConvBNLayer(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 1,
            act = 'relu')
        # conv
        self.conv1 = ConvBNLayer(
            in_channels = out_channels,
            out_channels = out_channels,
            kernel_size = 3,
            stride = stride,
            act = 'relu')
        # expansion
        self.conv2 = ConvBNLayer(
            in_channels = out_channels,
            out_channels = out_channels * self.expansion,
            kernel_size = 1,
            stride = 1,
            act = None)
        # short
        self.short = ShortCut(
            in_channels = in_channels,
            out_channels = out_channels * self.expansion,
            stride = stride,
            is_first = False)
        self.out_channels = out_channels * self.expansion 
        
    def forward(self, x):
        y = self.conv0(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y += self.short(x)
        y = F.relu(y)
        return y 

class Fuse(nn.Module):
    '''
    '''
    def __init__(self,
                 x_channels,
                 y_channels):
        super(Fuse, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels = x_channels + y_channels,
            out_channels = x_channels,
            kernel_size = 1)
        self.conv2 = ConvBNLayer(
            in_channels = x_channels,
            out_channels = x_channels,
            kernel_size = 3,
            stride = 1,
            act='relu')
        
    def forward(self, x, y):
        z = torch.cat([x, y], dim=1)
        z = self.conv1(z)
        z = self.conv2(z)
        return z 
        
class FPN(nn.Module):
    def __init__(self,
                 x_channels,
                 y_channels,
                 z_channels,
                 out_channels = 512):
        super(FPN, self).__init__()
        self.fuse2 = Fuse(y_channels, z_channels)
        self.fuse1 = Fuse(x_channels, y_channels)
        self.conv = nn.Conv2d(
            in_channels = x_channels,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1)
    
    def forward(self, x, y, z):
        f2 = self.fuse2(y, z)
        f1 = self.fuse1(x, f2)
        f = self.conv(f1)
        return f
    

class ResNetFPN(nn.Module):
    '''
    :param depth: the list of depth of each stage.
    :param block: the type of block to be used.
    :param in_channels: the channels of input feature.
    '''
    def __init__(self,
                 depth : list,
                 block : nn.Module,
                 in_channels = 1,
                 use_3x3 = True):
        super(ResNetFPN, self).__init__()
        # split 7x7 -> 3x3 3x3 3x3?
        self.use_3x3 = use_3x3
        if use_3x3:
            self.conv0_0 = ConvBNLayer(
                in_channels = in_channels,
                out_channels = 32,
                kernel_size = 3,
                act = 'relu')
            self.conv0_1 = ConvBNLayer(
                in_channels = 32,
                out_channels = 32,
                kernel_size = 3,
                act = 'relu')
            self.conv0_2 = ConvBNLayer(
                in_channels = 32,
                out_channels = 64,
                kernel_size = 3,
                act = 'relu')
        else:
            self.conv0 = ConvBNLayer(
                in_channels = in_channels,
                out_channels = 64,
                kernel_size = 7,
                stride = 2,
                act = 'relu')

        self.in_channels = 64
        self.layer1 = self._make_layer(block, 64,  depth[0], (2, 2), True)
        self.layer2 = self._make_layer(block, 128, depth[1], (2, 2))
        self.layer3 = self._make_layer(block, 256, depth[2], (1, 1))
        self.layer4 = self._make_layer(block, 512, depth[3], (1, 1))
        
        self.fpn = FPN(512, 1024, 2048)

        self._init_weight()        
        

    def _make_layer(self, 
                    block: nn.Module, 
                    out_channels : int, 
                    depth: list, 
                    stride, 
                    if_first=False):
        '''in_channels define by self.in_channels.
        '''
        layers = []
        for i in range(depth):
            layers.append(block(
                in_channels = self.in_channels, 
                out_channels = out_channels, 
                stride = stride if i == 0 else 1,
                if_first = i == 0 and if_first))
            # update self.in_channels
            self.in_channels = block.expansion * out_channels 
        return nn.Sequential(*layers)
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
    def forward(self, x):
        if self.use_3x3:
            x1 = self.conv0_0(x)
            x1 = self.conv0_1(x1)
            x1 = self.conv0_2(x1)
        else:
            x1 = self.conv0(x)
        
        y1 = self.layer1(x1)
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)
        y4 = self.layer4(y3)
        
        y = self.fpn(y2, y3, y4)
        
        return y
        
      
      
# define basic models.
def _resnet_fpn(block, layers, in_channels=1, use_3x3=False, pretrained=None):
    model = ResNetFPN(layers, block, in_channels, use_3x3=False)
    if pretrained:
        state_dict = torch.load(pretrained)
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    return model
     
     
def resnet18_fpn(pretrained=False, in_channels=1, use_3x3=False):
    '''ResNet18 with FPN'''
    return _resnet_fpn(BasicBlock, [2, 2, 2, 2], in_channels, use_3x3, pretrained)       
     
def resnet34_fpn(pretrained=False, in_channels=1, use_3x3=False):
    '''ResNet24 with FPN'''
    return _resnet_fpn(BasicBlock, [3, 4, 6, 3], in_channels, use_3x3, pretrained)     
     
def resnet50_fpn(pretrained=False, in_channels=1, use_3x3=False):
    '''ResNet50 with FPN'''
    return _resnet_fpn(BottleneckBlock, [3, 4, 6, 3], in_channels, use_3x3, pretrained)


def resnet101_fpn(pretrained=False, in_channels=1, use_3x3=False):
    '''ResNet101 with FPN'''
    return _resnet_fpn(BottleneckBlock, [3, 4, 23, 3], in_channels, use_3x3, pretrained)

def resnet152_fpn(pretrained=False, in_channels=1, use_3x3=False):
    '''ResNet152 with FPN'''
    return _resnet_fpn(BottleneckBlock, [3, 8, 36, 3], in_channels, use_3x3, pretrained)


        
'''
USE3x3
================================================================
Total params: 37,884,192
Trainable params: 37,884,192
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.06
Forward/backward pass size (MB): 1240.00
Params size (MB): 144.52
Estimated Total Size (MB): 1384.58
----------------------------------------------------------------

NOT USE3x3
================================================================
Total params: 37,859,264
Trainable params: 37,859,264
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.06
Forward/backward pass size (MB): 302.00
Params size (MB): 144.42
Estimated Total Size (MB): 446.48
----------------------------------------------------------------
'''                

    
        

if __name__ == "__main__":
    import time

    import torchinfo
    import torchsummary 
    resnet50_fpn = ResNetFPN([3, 4, 6, 3],
                             BottleneckBlock,
                             use_3x3=False)
    print(resnet50_fpn)
    x = torch.randn(size = (1, 1, 64, 256))
    print("input.size():", x.size())
    torch.save(resnet50_fpn.state_dict(), 'res50_fpn.pth')
    
    # print("output.size():", y.size())
    torchinfo.summary(resnet50_fpn, (1, 1, 64, 256))
    
