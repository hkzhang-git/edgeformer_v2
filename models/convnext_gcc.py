import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

from convnext import Block
from gcc_modules import gcc_Block

class ConvNeXt_gcc(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            if i < 2:   # for stage 0 and 1, no gcc
                stage = nn.Sequential(*[
                    Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) \
                    for j in range(depths[i])
                ])
            else:       # for stage 2 and 3, gcc modules is used
                stage = nn.Sequential(*[
                    Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) \
                    if j+1>(depths[i]-depths[i]//3) else \
                    gcc_Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) \
                    for j in range(depths[i]) # here we use gcc in the last 1/3 blocks
                ])                            # e.g., j+1=7 > 9-9//3=6, so block 678 is gcc_block, where block 0~5 is normal
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
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

@register_model
def convnext_gcc_tiny(pretrained=False,in_22k=False, **kwargs):
    # model = ConvNeXt_gcc(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    model = ConvNeXt_gcc(depths=[3, 3, 9, 3], dims=[48, 96, 192, 384], **kwargs)
    # model = ConvNeXt_gcc(depths=[3, 3, 9, 3], dims=[24, 48, 96, 192], **kwargs)
    if pretrained or in_22k:
        raise AttributeError("no pretrained model")
    return model

@register_model
def convnext_gcc_small(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt_gcc(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained or in_22k:
        raise AttributeError("no pretrained model")
    return model

@register_model
def convnext_gcc_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt_gcc(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained or in_22k:
        raise AttributeError("no pretrained model")
    return model

@register_model
def convnext_gcc_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt_gcc(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained or in_22k:
        raise AttributeError("no pretrained model")
    return model

@register_model
def convnext_gcc_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt_gcc(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained or in_22k:
        raise AttributeError("no pretrained model")
    return model
