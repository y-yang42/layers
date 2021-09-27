# PyTorch implementation of locally connected 2d layer
# which is similar to convolutional layer
# except that the weights are not shared across different windows
# Can only do zero padding

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch import Tensor
from torch.nn.modules.utils import _pair

import math


class LocallyConnected2dFunc(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, weight, output_size, kernel_size, bias=None, stride=1, padding=0, dilation=1):
        ctx.save_for_backward(input, weight, bias) 
        ctx.output_size = output_size
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding 
        ctx.dilation = dilation
        
        input_unf = F.unfold(input, kernel_size, dilation, padding, stride).unsqueeze(1)
        weight_unf = weight.view(weight.size(0), weight.size(1)*weight.size(2), -1).transpose(1,2)
        output_unf = (input_unf * weight_unf).sum(2) # (batch_size, out_channels, prod_output_size)
        if bias is not None:
            output_unf += bias.unsqueeze(1)
        output = F.fold(output_unf, output_size, (1,1))
        
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        output_size = ctx.output_size
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        padding = ctx.padding 
        dilation = ctx.dilation

        grad_input = grad_weight = grad_bias = None

        grad_output_unf = grad_output.view(grad_output.size(0), grad_output.size(1), -1)
        if ctx.needs_input_grad[0]:
            grad_input_unf = grad_output_unf.unsqueeze(2) * weight.view(weight.size(0), weight.size(1)*weight.size(2), -1).transpose(1,2)
            grad_input = F.fold(grad_input_unf.sum(1), input.shape[2:], kernel_size, dilation, padding, stride)
        if ctx.needs_input_grad[1]:
            grad_weight_unf = grad_output_unf.unsqueeze(2) * F.unfold(input, kernel_size, dilation, padding, stride).unsqueeze(1)
            grad_weight = grad_weight_unf.sum(0).transpose(1,2).view(weight.shape)
        if bias is not None and ctx.needs_input_grad[4]:
            grad_bias = grad_output.sum((0,2,3))

        return grad_input, grad_weight, None, None, grad_bias, None, None, None


class LocallyConnected2d(nn.Module):
    def __init__(self, in_features, in_channels, out_channels, kernel_size, stride=1,
             padding=0, dilation=1, bias=True):
        super(LocallyConnected2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_features = _pair(in_features)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        
        output_size = ((self.in_features[0]+2*self.padding[0]-self.dilation[0] * (self.kernel_size[0]-1)-1)//self.stride[0] + 1,
                            (self.in_features[1]+2*self.padding[1]-self.dilation[1] * (self.kernel_size[1]-1)-1)//self.stride[1] + 1)
        self.output_size = _pair(output_size)

        self.weight = nn.Parameter(
            torch.randn(out_channels, *self.output_size, in_channels, *self.kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('in_features={in_features}, {in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)
        
    def forward(self, input):
        return LocallyConnected2dFunc.apply(input, self.weight, self.output_size, self.kernel_size,
                                            self.bias, self.stride, self.padding, self.dilation)
