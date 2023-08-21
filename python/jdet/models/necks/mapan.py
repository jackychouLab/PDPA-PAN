import jittor as jt
from jittor import nn
import warnings
import random
from jdet.utils.registry import NECKS
from jdet.models.utils.modules import ConvModule
from jdet.models.utils.weight_init import xavier_init

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(jt.ones(normalized_shape))
        self.bias = nn.Parameter(jt.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def execute(self, x):
        if self.data_format == "channels_last":
            return nn.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = jt.mean(x, dim=1, keepdims=True)
            s = jt.mean((x - u).pow(2), dim=1, keepdims=True)
            x = (x - u) / jt.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class SPPM(nn.Module):
    def __init__(self, inchannel):
        super().__init__()
        self.pools = nn.ModuleList()

        self.pool5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool9 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.pool13 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)

        self.conv = nn.Conv2d(inchannel*4, inchannel, kernel_size=1, stride=1, padding=0)
        self.ln = LayerNorm(inchannel, eps=1e-6, data_format="channels_first")
        self.gelu = nn.GELU()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def execute(self, x):
        outputs = [x]
        outputs.append(self.pool5(x))
        outputs.append(self.pool9(x))
        outputs.append(self.pool13(x))
        x = jt.concat(outputs, dim=1)
        x = self.gelu(self.ln(self.conv(x)))
        return x

class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def execute(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out = x.max(dim=1, keepdims=True)
        avg_out = x.mean(dim=1, keepdims=True)
        spatial_out = self.sigmoid(self.conv(jt.concat([max_out, avg_out], dim=1)))
        x = spatial_out * x

        return x

class scSE(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(channel, channel//reduction, 1, bias=False),
                                 nn.GELU(),
                                 nn.Conv2d(channel//reduction, channel, 1, bias=False),
                                 nn.Sigmoid())

        self.sSE = nn.Sequential(nn.Conv2d(channel, channel, 1, bias=False),
                                 nn.Sigmoid())
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def execute(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class GC(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super().__init__()
        self.ratio = ratio
        self.conv_theta = nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_phi = nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_out = nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def execute(self, x):
        batch_size, channels, height, width = x.size()

        theta = self.conv_theta(x).view(batch_size, channels // self.ratio, -1)
        phi = self.conv_phi(x).view(batch_size, channels // self.ratio, -1).permute(0, 2, 1)
        g = self.conv_g(x).view(batch_size, channels // self.ratio, -1)

        attention = jt.bmm(phi, theta)
        attention = self.softmax(attention)

        out = jt.bmm(g, attention).contiguous()
        out = out.view(batch_size, channels // self.ratio, height, width)
        out = self.conv_out(out)

        return out + x


class CSKConv(nn.Module):
    def __init__(self, features, M=2, G=32, r=16, stride=1, L=32):
        super().__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1 + i, dilation=1 + i, groups=G, bias=False),
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

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

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
                nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1 + i, dilation=1 + i, groups=G, bias=False),
                LayerNorm(features, eps=1e-6, data_format="channels_first"),
                nn.GELU()
            ))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Conv2d(features, d, kernel_size=1, stride=1, bias=False),
                                LayerNorm(d, eps=1e-6, data_format="channels_first"),
                                nn.GELU())
        self.fcs = nn.ModuleList([])

        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

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

class SPSKConv(nn.Module):
    def __init__(self, features, M=2, G=32, r=16, stride=1, L=32):
        super().__init__()
        d = max(int(features * 21 / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1 + i, dilation=1 + i, groups=G, bias=False),
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

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

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

class mixA(nn.Module):
    def __init__(self, features, C_a="SKConv"):
        super().__init__()
        if C_a == "SKConv":
            self.c = SKConv(features)
            self.s = SKConv(features)
        if C_a == "SPSKConv":
            self.c = SPSKConv(features)
            self.s = SPSKConv(features)

    def execute(self, x):
        return self.s(x) + self.c(x)

@NECKS.register_module()
class MAPAN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_SPP=False,
                 add_atten=False,
                 Atten_before_re=False,
                 Attention_type="mix",  # S_atten, C_atten, mix, CBAM, scSE, mixA
                 S_Atten="GC",
                 C_Atten="SKConv",      # SKConv, SPSKConv
                 ):
        super(MAPAN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        self.add_SPP = add_SPP
        if self.add_SPP:
            self.SPP = SPPM(in_channels[-1])
        self.add_atte = add_atten
        if self.add_atte == True:
            self.Atten_before = Atten_before_re
            self.Attention_type = Attention_type
            self.S_Atten = S_Atten
            self.C_Atten = C_Atten

        self.up_attens = nn.ModuleList()
        self.down_attens = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level - 1):
            if self.Attention_type == "CBAM":
                up_atten = CBAM(out_channels)
                down_atten = CBAM(out_channels)
                self.up_attens.append(up_atten)
                self.down_attens.append(down_atten)
            if self.Attention_type == "scSE":
                up_atten = scSE(out_channels)
                down_atten = scSE(out_channels)
                self.up_attens.append(up_atten)
                self.down_attens.append(down_atten)
            if self.Attention_type == "mixA":
                up_atten = mixA(out_channels)
                down_atten = mixA(out_channels)
                self.up_attens.append(up_atten)
                self.down_attens.append(down_atten)
            if self.Attention_type == "mix":
                if self.C_Atten == "SKConv":
                    up_atten = SKConv(out_channels)
                if self.C_Atten == "SPSKConv":
                    up_atten = SPSKConv(out_channels)
                down_atten = GC(out_channels)
                self.up_attens.append(up_atten)
                self.down_attens.append(down_atten)

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = nn.Conv2d(in_channels[i], out_channels, 1, stride=1, padding=0)
            fpn_conv = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = nn.Conv2d(in_channels, out_channels, 2, stride=2, padding=0)
                self.fpn_convs.append(extra_fpn_conv)

        self.up_convs = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level - 1):
            self.up_conv = nn.ConvTranspose(out_channels,  out_channels, kernel_size=2, stride=2, padding=0)
            self.down_conv = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0)
            self.up_convs.append(self.up_conv)
            self.down_convs.append(self.down_conv)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def execute(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            if i == len(self.lateral_convs) - 1 and self.add_SPP:

                laterals.append(lateral_conv(self.SPP(inputs[i + self.start_level])))
            else:
                laterals.append(lateral_conv(inputs[i + self.start_level]))

        # [4, 8, 16]
        # build small-big path
        used_backbone_levels = len(laterals)
        if self.add_atte:
            for i, up_conv, up_atten in zip(range(used_backbone_levels-1, 0, -1), self.up_convs, self.up_attens):
                # add attention
                if self.Atten_before:
                    laterals[i - 1] += up_conv(up_atten(laterals[i]))
                else:
                    laterals[i - 1] += up_atten(up_conv(laterals[i]))
        else:
            for i, up_conv in zip(range(used_backbone_levels-1, 0, -1), self.up_convs):
                laterals[i - 1] += up_conv(laterals[i])


        # build big-small path
        if self.add_atte:
            for i, down_conv, down_atten in zip(range(1, used_backbone_levels), self.down_convs, self.down_attens):
                # add attention
                if self.Atten_before:
                    laterals[i] = down_conv(down_atten(laterals[i-1]))
                else:
                    laterals[i] = down_atten(down_conv(laterals[i-1]))
        else:
            for i, down_conv in zip(range(1, used_backbone_levels), self.down_convs):
                laterals[i] = down_conv(laterals[i-1])

        # build outputs
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        if self.num_outs > len(outs):
            extra_source = inputs[self.backbone_end_level - 1]
            outs.append(self.fpn_convs[used_backbone_levels](extra_source))
            for i in range(used_backbone_levels + 1, self.num_outs):
                outs.append(self.fpn_convs[i](outs[-1]))

        return tuple(outs)

