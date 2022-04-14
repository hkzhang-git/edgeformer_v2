import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

# Convnext like Blocks (trunc_normal weight init)
class gcc_Conv2d(nn.Module):
    def __init__(self, dim, type, meta_kernel_size, instance_kernel_method=None):
        super().__init__()
        # super(gcc_Conv2d, self).__init__()
        self.type = type    # H or W
        self.dim = dim
        self.instance_kernel_method = instance_kernel_method
        self.meta_kernel_size = (meta_kernel_size, 1) if self.type=='H' else (1, meta_kernel_size)
        self.weight = nn.Conv2d(dim, dim, self.meta_kernel_size, groups=dim).weight
        self.bias   = nn.Parameter(torch.randn(dim))

    def gcc_init(self):
        trunc_normal_(self.weight, std=.02)
        nn.init.constant_(self.bias, 0)

    def get_instance_kernel(self, instance_kernel_size):
        # if no use of dynamic resolution, keep a static kernel
        if self.instance_kernel_method is None:
            return  self.weight
        elif self.instance_kernel_method == 'interpolation_bilinear':
            instance_kernel_size =  (instance_kernel_size, 1) if self.type=='H' else (1, instance_kernel_size)
            return  F.interpolate(self.weight, instance_kernel_size, mode='bilinear', align_corners=True)

    def forward(self, x):
        _, _, f_s, _ = x.shape
        weight = self.get_instance_kernel(f_s)
        x_cat = torch.cat((x, x[:, :, :-1, :]), dim=2) if self.type=='H' else torch.cat((x, x[:, :, :, :-1]), dim=3)
        x = F.conv2d(x_cat, weight=weight, bias=self.bias, padding=0, groups=self.dim)
        return x

class gcc_cvx_Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, meta_kernel_size=16, instance_kernel_method=None):
        super().__init__()
        # super(gcc_cvx_Block, self).__init__()
        self.gcc_conv_H = gcc_Conv2d(dim//2, type='H', meta_kernel_size=meta_kernel_size, instance_kernel_method=instance_kernel_method) 
        self.gcc_conv_W = gcc_Conv2d(dim//2, type='W', meta_kernel_size=meta_kernel_size, instance_kernel_method=instance_kernel_method)
        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x_1, x_2 = torch.chunk(x, 2, 1)
        x_1, x_2 = self.gcc_conv_H(x_1), self.gcc_conv_W(x_2)
        x = torch.cat((x_1, x_2), dim=1)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        # super(Block, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        # super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x