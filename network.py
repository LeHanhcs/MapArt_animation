import torch
import torch.nn as nn
import numpy as np
from torchvision.models import vgg16 ,vgg19
from collections import namedtuple

vgglayers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4', 'conv4_1',
'conv4_2', 'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']
# From https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/vgg.py

class Vgg19(torch.nn.Module):
    def __init__(self, device='cpu'):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = vgg19(pretrained=True).features
        self.conv1_1 = torch.nn.Sequential()
        self.conv1_2 = torch.nn.Sequential()
        self.conv2_1 = torch.nn.Sequential()
        self.conv2_2 = torch.nn.Sequential()
        self.conv3_2 = torch.nn.Sequential()
        self.conv3_3 = torch.nn.Sequential()
        self.conv3_4 = torch.nn.Sequential()
        self.conv4_2 = torch.nn.Sequential()
        self.conv4_3 = torch.nn.Sequential()
        self.conv4_4 = torch.nn.Sequential()
        self.conv5_2 = torch.nn.Sequential()
        self.conv5_3 = torch.nn.Sequential()
        self.conv5_4 = torch.nn.Sequential()

        for x in range(2):
            self.conv1_1.add_module(str(x), vgg_pretrained_features[x].to(device))
        for x in range(2, 4):
            self.conv1_2.add_module(str(x), vgg_pretrained_features[x].to(device))
        for x in range(4, 7):
            self.conv2_1.add_module(str(x), vgg_pretrained_features[x].to(device))
        for x in range(7, 9):
            self.conv2_2.add_module(str(x), vgg_pretrained_features[x].to(device))
        for x in range(9, 14):
            self.conv3_2.add_module(str(x), vgg_pretrained_features[x].to(device))
        for x in range(14, 16):
            self.conv3_3.add_module(str(x), vgg_pretrained_features[x].to(device))
        for x in range(16, 18):
            self.conv3_4.add_module(str(x), vgg_pretrained_features[x].to(device))
        for x in range(18, 23):
            self.conv4_2.add_module(str(x), vgg_pretrained_features[x].to(device))
        for x in range(23, 25):
            self.conv4_3.add_module(str(x), vgg_pretrained_features[x].to(device))
        for x in range(25, 27):
            self.conv4_4.add_module(str(x), vgg_pretrained_features[x].to(device))
        for x in range(27, 32):
            self.conv5_2.add_module(str(x), vgg_pretrained_features[x].to(device))
        for x in range(32, 34):
            self.conv5_3.add_module(str(x), vgg_pretrained_features[x].to(device))
        for x in range(34, 36):
            self.conv5_4.add_module(str(x), vgg_pretrained_features[x].to(device))
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X):
        model={}
        h = self.conv1_1(X)
        # model['conv1_1'] = h
        h = self.conv1_2(h)
        model['conv1_2'] = h
        h = self.conv2_1(h)
        # model['conv2_1'] = h
        h = self.conv2_2(h)
        model['conv2_2'] = h
        h = self.conv3_2(h)
        model['conv3_2'] = h
        h = self.conv3_3(h)
        # model['conv3_3'] = h
        h = self.conv3_4(h)
        # model['conv3_4'] = h
        h = self.conv4_2(h)
        model['conv4_2'] = h
        h = self.conv4_3(h)
        model['conv4_3'] = h
        h = self.conv4_4(h)
        # model['conv4_4'] = h
        h = self.conv5_2(h)
        model['conv5_2'] = h
        h = self.conv5_3(h)
        # model['conv5_3'] = h
        h = self.conv5_4(h)
        # model['conv5_4'] = h
        return model

class TLU(nn.Module):
    def __init__(self, num_features):
        """max(y, tau) = max(y - tau, 0) + tau = ReLU(y - tau) + tau"""
        super(TLU, self).__init__()
        self.num_features = num_features
        self.tau = nn.parameter.Parameter(
            torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.tau)

    def extra_repr(self):
        return 'num_features={num_features}'.format(**self.__dict__)

    def forward(self, x):
        return torch.max(x, self.tau)


class FRN(nn.Module):
    def __init__(self, num_features, eps=1e-6, is_eps_leanable=False):
        """
        weight = gamma, bias = beta
        beta, gamma:
            Variables of shape [1, 1, 1, C]. if TensorFlow
            Variables of shape [1, C, 1, 1]. if PyTorch
        eps: A scalar constant or learnable variable.
        """
        super(FRN, self).__init__()

        self.num_features = num_features
        self.init_eps = eps
        self.is_eps_leanable = is_eps_leanable

        self.weight = nn.parameter.Parameter(
            torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.bias = nn.parameter.Parameter(
            torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        if is_eps_leanable:
            self.eps = nn.parameter.Parameter(torch.Tensor(1), requires_grad=True)
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.is_eps_leanable:
            nn.init.constant_(self.eps, self.init_eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={init_eps}'.format(**self.__dict__)

    def forward(self, x):
        """
        0, 1, 2, 3 -> (B, H, W, C) in TensorFlow
        0, 1, 2, 3 -> (B, C, H, W) in PyTorch
        TensorFlow code
            nu2 = tf.reduce_mean(tf.square(x), axis=[1, 2], keepdims=True)
            x = x * tf.rsqrt(nu2 + tf.abs(eps))
            # This Code include TLU function max(y, tau)
            return tf.maximum(gamma * x + beta, tau)
        """
        # Compute the mean norm of activations per channel.
        nu2 = x.pow(2).mean(dim=[2, 3], keepdim=True)

        # Perform FRN.
        x = x * torch.rsqrt(nu2 + self.eps.abs())

        # Scale and Bias
        x = self.weight * x + self.bias
        return x

class SelectiveLoadModule(torch.nn.Module):
    """Only load layers in trained models with the same name."""
    def __init__(self):
        super(SelectiveLoadModule, self).__init__()

    def forward(self, x):
        return x

    def load_state_dict(self, state_dict):
        """Override the function to ignore redundant weights."""
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                own_state[name].copy_(param)


class ConvLayer(torch.nn.Module):
    """Reflection padded convolution layer."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=True):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=bias)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ConvTanh(ConvLayer):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvTanh, self).__init__(in_channels, out_channels, kernel_size, stride)
        self.tanh = nn.Tanh()
    def forward(self, x):
        out = super(ConvTanh, self).forward(x)
        return self.tanh(out/255) * 150 + 255/2
class ConvInstRelu(ConvLayer):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvInstRelu, self).__init__(in_channels, out_channels, kernel_size, stride)
        # self.instance = torch.nn.InstanceNorm2d(out_channels, affine=True)
        # self.relu = torch.nn.ReLU()
        self.instance = FRN(out_channels)
        self.relu = TLU(out_channels)

    def forward(self, x):
        out = super(ConvInstRelu, self).forward(x)
        out = self.instance(out)
        out = self.relu(out)
        return out


class UpsampleConvLayer(torch.nn.Module):
    """Upsamples the input and then does a convolution.
    This method gives better results compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class UpsampleConvInstRelu(UpsampleConvLayer):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvInstRelu, self).__init__(in_channels, out_channels, kernel_size, stride, upsample)
        # self.instance = torch.nn.InstanceNorm2d(out_channels, affine=True)
        # self.relu = torch.nn.ReLU()
        self.instance = FRN(out_channels)
        self.relu = TLU(out_channels)


    def forward(self, x):
        out = super(UpsampleConvInstRelu, self).forward(x)
        out = self.instance(out)
        out = self.relu(out)
        return out
class FuseResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(FuseResidualBlock, self).__init__()
        self.identity_block = nn.Sequential(
            ConvLayer(in_channels, out_channels//4, kernel_size=kernel_size, stride=1),
            # torch.nn.InstanceNorm2d(out_channels//4, affine=True),
            # torch.nn.ReLU(),

            FRN(out_channels//4),
            TLU(out_channels//4),

            # ConvLayer(out_channels//4, out_channels//4, kernel_size=kernel_size, stride=stride),
            # torch.nn.InstanceNorm2d(out_channels//4, affine=True),
            # torch.nn.ReLU(),
            
            ConvLayer(out_channels//4, out_channels, kernel_size=kernel_size, stride=1),
            # torch.nn.InstanceNorm2d(out_channels, affine=True),
            # torch.nn.ReLU()
            FRN(out_channels),
            TLU(out_channels)
        )
        self.shortcut = nn.Sequential(
            ConvLayer(in_channels, out_channels, 1, stride),
            FRN(out_channels)
            # torch.nn.InstanceNorm2d(out_channels, affine=True)
        )
    def forward(self, x):
        out = self.identity_block(x)
        residual = self.shortcut(x)
        out = out + residual
        return out 
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride)
        self.in1 = FRN(out_channels)
        # self.in1 = torch.nn.InstanceNorm2d(out_channels, affine=True)

        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size, stride)
        # self.in2 = torch.nn.InstanceNorm2d(out_channels, affine=True)
        # self.relu = torch.nn.ReLU()
        self.in2 = FRN(out_channels)
        self.relu = TLU(out_channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out
def upsample(input, scale_factor):
    return nn.functional.interpolate(input=input, scale_factor=scale_factor, mode='bilinear', align_corners=False)

class Encoder(SelectiveLoadModule):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = ConvInstRelu(3, 32, kernel_size=9, stride=1)
        self.conv2 = ConvInstRelu(32, 48, kernel_size=3, stride=2)
        self.conv3 = ConvInstRelu(48, 64, kernel_size=3, stride=2)
        self.Fuse2 = FuseResidualBlock(48, 64)

        self.res1 = ResidualBlock(128, 128)
        self.res2 = ResidualBlock(128, 128)
        self.res3 = ResidualBlock(128, 128)
        self.res4 = ResidualBlock(128, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        x = torch.cat((self.Fuse2(x), upsample(self.conv3(x),2)), 1)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        return x
class Decoder(SelectiveLoadModule):
    def __init__(self):
        super(Decoder, self).__init__()

        self.deconv1 = UpsampleConvInstRelu(128, 64, kernel_size=3, stride=1, upsample=1)
        # self.deconv1 = UpsampleConvInstRelu(128, 64, kernel_size=3, stride=1, upsample=2)
        self.deconv2 = UpsampleConvInstRelu(64, 32, kernel_size=3, stride=1, upsample=2)
        self.deconv3 = ConvTanh(32, 3, kernel_size=9, stride=1)
    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x
class ReCoNet(SelectiveLoadModule):
    def __init__(self):
        super(ReCoNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

