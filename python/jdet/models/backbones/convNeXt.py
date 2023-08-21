import jittor
import jittor as jt
from jittor import nn

from jdet.utils.registry import BACKBONES

__all__ = ['ConvNeXt', 'ConvNeXt_tiny', 'ConvNeXt_small', 'ConvNeXt_base', 'ConvNeXt_large', 'ConvNeXt_xlarge']


class CSKConv(nn.Module):
    def __init__(self, features, M=2, G=32, r=16, stride=1, L=32):
        super().__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1 + i, dilation=1 + i, groups=G,
                          bias=False),
                LayerNorm(features, eps=1e-6, data_format="channels_first"),
                nn.GELU()
            ))

        self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
                                LayerNorm(d, eps=1e-6, data_format="channels_first"),
                                nn.GELU())
        self.fcs = nn.ModuleList([])

        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            try:
                nn.init.constant_(m.bias, 0)
            except:
                pass

    def execute(self, x):

        batch_size = x.shape[0]

        feats = [conv(x) for conv in self.convs]
        feats = jt.concat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])

        feats_U = feats.sum(dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = jt.concat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)

        feats_V = (feats * attention_vectors).sum(dim=1)

        return feats_V


class SSKConv(nn.Module):
    def __init__(self, features, M=2, G=32, r=16, stride=1, L=32):
        super().__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1 + i, dilation=1 + i, groups=G,
                          bias=False),
                LayerNorm(features, eps=1e-6, data_format="channels_first"),
                nn.GELU()
            ))

        self.fc = nn.Sequential(nn.Conv2d(2, 1, kernel_size=1, stride=1, bias=False),
                                LayerNorm(d, eps=1e-6, data_format="channels_first"),
                                nn.GELU())

        self.fcs = nn.ModuleList([])

        for i in range(M):
            self.fcs.append(
                nn.Conv2d(1, 1, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=2)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            try:
                nn.init.constant_(m.bias, 0)
            except:
                pass

    def execute(self, x):

        batch_size, c, h, w = x.shape

        feats = [conv(x) for conv in self.convs]
        feats = jt.concat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])

        feats_U = feats.sum(dim=1)
        max_out = feats_U.max(dim=1, keepdims=True)
        avg_out = feats_U.mean(dim=1, keepdims=True)

        mid_feats = jt.concat([max_out, avg_out], dim=1)
        feats_Z = self.fc(mid_feats)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = jt.concat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, 1, h, w)
        attention_vectors = self.softmax(attention_vectors)

        feats_V = (feats * attention_vectors).sum(dim=1)

        return feats_V


class SPCSKConv(nn.Module):
    def __init__(self, features, M=2, G=32, r=16, stride=1, L=32):
        super().__init__()
        d = max(int(features * 21 / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1 + i, dilation=1 + i, groups=G,
                          bias=False),
                LayerNorm(features, eps=1e-6, data_format="channels_first"),
                nn.GELU()
            ))
        self.gap1 = nn.AdaptiveAvgPool2d((1, 1))
        self.gap2 = nn.AdaptiveAvgPool2d((2, 2))
        self.gap4 = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(nn.Conv2d(features * 21, d, kernel_size=1, stride=1, bias=False),
                                LayerNorm(d, eps=1e-6, data_format="channels_first"),
                                nn.GELU())
        self.fcs = nn.ModuleList([])

        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, kernel_size=1, stride=1)
            )

        self.softmax = nn.Softmax(dim=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            try:
                nn.init.constant_(m.bias, 0)
            except:
                pass

    def execute(self, x):
        batch_size = x.shape[0]

        feats = [conv(x) for conv in self.convs]
        feats = jt.concat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])

        feats_U = feats.sum(dim=1)
        feats_S1 = self.gap1(feats_U)
        feats_S2 = self.gap2(feats_U)
        feats_S4 = self.gap4(feats_U)
        feats_S = jt.concat([feats_S1, feats_S2, feats_S4], dim=1)
        feats_Z = self.fc(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = jt.concat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)

        feats_V = (feats * attention_vectors).sum(dim=1)

        return feats_V


class SCSKConv(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.s = SSKConv(features)
        self.c = CSKConv(features)

    def execute(self, x):
        return self.s(x) + self.c(x)

class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, CBAM_type=None):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # layer_scale 操作有助于MLP的收敛和提升精度
        self.gamma = nn.Parameter(layer_scale_init_value * jittor.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.CBAM_type = CBAM_type
        if self.CBAM_type is not None:
            if self.CBAM_type == "old":
                self.CBAM = CBAMLayer(channel=dim)
            if self.CBAM_type == "new":
                self.CBAM = CBAMLayer_Xt(channel=dim)
        # self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def execute(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        if self.CBAM_type is not None:
            x = self.CBAM(x)
        x = input + x
        return x

class ConvNeXt(nn.Module):
    def __init__(self, layers=[3, 3, 9, 3], dims=[96, 192, 384, 768], return_stages=["layer4"], num_classes=None,  layer_scale_init_value=1e-6, head_init_scale=1., norm_eval=True, CBAM_type=None):
        super(ConvNeXt, self).__init__()

        in_chans = 3
        depths = layers
        drop_path_rate = 0.
        self.norm_eval = True
        self.num_classes = num_classes
        self.return_stages = return_stages
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        self.CBAM_type = CBAM_type
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in jittor.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value, CBAM_type=self.CBAM_type) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        if num_classes is not None:
            self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
            self.head = nn.Linear(dims[-1], num_classes)
            self.head.weight.__mul__(head_init_scale)
            self.head.bias.__mul__(head_init_scale)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            try:
                nn.init.constant_(m.bias, 0)
            except:
                pass

    def execute(self, x):
        outputs = []
        for i in range(4):
            name = f"layer{i+1}"
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if name in self.return_stages:
                outputs.append(x)
        if self.num_classes is not None:
            x = self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)
            x = self.head(x)
            if "fc" in self.return_stages:
                outputs.append(x)
        return tuple(outputs)

    def _freeze_stages(self):
        if False:
            print("freeze stage1")
            for m in [self.downsample_layers]:
                for param in m.parameters():
                    param.stop_grad()

    def train(self):
        super(ConvNeXt, self).train()
        self._freeze_stages()


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(jittor.ones(normalized_shape))
        self.bias = nn.Parameter(jittor.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def execute(self, x):
        if self.data_format == "channels_last":
            return nn.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = jittor.mean(x, dim=1, keepdims=True)
            s = jittor.mean((x - u).pow(2), dim=1, keepdims=True)
            x = (x - u) / jittor.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


@BACKBONES.register_module()
def ConvNeXt_tiny(pretrained=True, **kwargs):
    model = ConvNeXt(layers=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        checkpoint = jittor.load('/home/lab315-server/project_zjh/JDet/work_dirs/pre_train/convnext_tiny_22k_224.pth')
        model.load_state_dict(checkpoint["model"])
    return model
convnext_tiny = ConvNeXt_tiny

@BACKBONES.register_module()
def ConvNeXt_small(pretrained=True, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        checkpoint = jittor.load('/home/msi/project/JDet-master/work_dirs/pre_train/convnext_small_1k_224_ema.pth')
        model.load_state_dict(checkpoint["model"])
    return model

@BACKBONES.register_module()
def ConvNeXt_base(pretrained=True, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        checkpoint = jittor.load('/home/msi/project/JDet-master/work_dirs/pre_train/convnext_base_1k_224_ema.pth')
        model.load_state_dict(checkpoint["model"])
    return model

@BACKBONES.register_module()
def ConvNeXt_large(pretrained=True, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        checkpoint = jittor.load('/home/msi/project/JDet-master/work_dirs/pre_train/convnext_large_1k_224_ema.pth')
        model.load_state_dict(checkpoint["model"])
    return model

@BACKBONES.register_module()
def ConvNeXt_xlarge(pretrained=True, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        checkpoint = jittor.load('/home/msi/project/JDet-master/work_dirs/pre_train/convnext_xlarge_22k_1k_224_ema.pth')
        model.load_state_dict(checkpoint["model"])
    return model
