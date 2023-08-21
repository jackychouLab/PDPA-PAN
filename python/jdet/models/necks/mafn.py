import jittor as jt
from jittor import nn
from jdet.utils.registry import NECKS

class down_sample(nn.Module):
    def __init__(self, inchannel=256, down_step=2):
        super().__init__()
        self.down_conv = nn.Conv2d(inchannel, inchannel, kernel_size=down_step, stride=down_step, padding=0)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.constant_(m.bias, 0)

    def execute(self, x):
        x = self.down_conv(x)
        return x

class up_sample(nn.Module):
    def __init__(self, inchannel=256, up_step=2):
        super().__init__()
        self.up_conv = nn.ConvTranspose(inchannel,  inchannel, kernel_size=up_step, stride=up_step, padding=0)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.constant_(m.bias, 0)

    def execute(self, x):
        x = self.up_conv(x)
        return x


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=4, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                # nn.init.constant_(m.bias, 0)

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

class CBAMLayer_Xt(nn.Module):
    def __init__(self, channel, expansion=4, spatial_kernel=7, layer_scale_init_value=1e-6):
        super(CBAMLayer_Xt, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.gama = nn.Parameter(layer_scale_init_value * jt.ones((channel)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel * expansion, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channel * expansion, channel, 1, bias=False),
        )

        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, stride=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def execute(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        if self.gama is not None:
            max_out = max_out.permute(0, 2, 3, 1) * self.gama
            avg_out = avg_out.permute(0, 2, 3, 1) * self.gama
            max_out = max_out.permute(0, 3, 1, 2)
            avg_out = avg_out.permute(0, 3, 1, 2)
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out = x.max(dim=1, keepdims=True)
        avg_out = x.mean(dim=1, keepdims=True)
        spatial_out = self.sigmoid(self.conv(jt.concat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

@NECKS.register_module()
class MAFN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 merge_level=3,
                 merge_conv_size=3,
                 merge_type="group",
                 CBAM = None,
                 out_conv_size=3):
        super(MAFN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.out_conv_size = out_conv_size
        self.num_outs = num_outs
        self.CBAM = CBAM
        self.merge_type = merge_type
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.merge_level = merge_level
        self.merge_conv_size = merge_conv_size
        self.input_convs = nn.ModuleList()
        self.modify1_convs = nn.ModuleList()
        self.modify2_convs = nn.ModuleList()
        if self.merge_type == "group":
            self.group_convs = nn.ModuleList()
        if self.merge_type == "concat":
            self.cat_convs = nn.ModuleList()
        self.CBAM = CBAM
        if self.out_conv_size == "mix":
            self.out_size = 7
        for i in range(self.start_level, self.backbone_end_level):
            input_conv = nn.Sequential(
                nn.Conv2d(in_channels[i], out_channels, 1)
            )
            self.input_convs.append(input_conv)
            if i != self.merge_level:
                if i < self.merge_level:
                    down_step = (self.merge_level - i) * 2
                    modify1_conv = nn.Sequential(*[down_sample(inchannel=out_channels, down_step=down_step)])
                    modify2_conv = nn.Sequential(*[up_sample(inchannel=out_channels, up_step=down_step)])
                    self.modify1_convs.append(modify1_conv)
                    self.modify2_convs.append(modify2_conv)
                else:
                    up_step = (i - self.merge_level) * 2
                    modify1_conv = nn.Sequential(*[up_sample(inchannel=out_channels, up_step=up_step)])
                    modify2_conv = nn.Sequential(*[down_sample(inchannel=out_channels, down_step=up_step)])
                    self.modify1_convs.append(modify1_conv)
                    self.modify2_convs.append(modify2_conv)
            if self.merge_type == "group":
                if self.out_conv_size == "mix":
                    group_conv = nn.Conv2d(out_channels * 2, out_channels, self.out_size, stride=1, padding=self.out_size//2, groups=out_channels)
                    self.out_size -= 2
                else:
                    group_conv = nn.Conv2d(out_channels * 2, out_channels, self.out_conv_size, stride=1, padding=self.out_conv_size // 2, groups=out_channels)
                self.group_convs.append(group_conv)
            if self.merge_type == "concat":
                if self.out_conv_size == 0:
                    cat_conv = nn.Conv2d(out_channels * 2, out_channels, self.out_size, stride=1, padding=self.out_size//2)
                    self.out_size -= 2
                else:
                    cat_conv = nn.Conv2d(out_channels * 2, out_channels, self.out_conv_size, stride=1, padding=self.out_conv_size // 2)
                self.cat_convs.append(cat_conv)

        self.merge_conv = nn.Conv2d(out_channels * (len(in_channels) - start_level), out_channels, self.merge_conv_size, stride=1, padding=self.merge_conv_size//2)

        self.extra_convs = nn.ModuleList()
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if extra_levels >= 1:
            for i in range(extra_levels):
                extra_conv = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2),
                )
                self.extra_convs.append(extra_conv)

        self.init_weights()

        if self.CBAM is not None:
            if self.CBAM == "old":
                self.CBAM_module = CBAMLayer(out_channels * (len(in_channels) - start_level))
            elif self.CBAM == "new":
                self.CBAM_module = CBAMLayer_Xt(out_channels * (len(in_channels) - start_level))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.constant_(m.bias, 0)

    def execute(self, inputs):
        assert len(inputs) == len(self.in_channels)
        laterals = [
            input_conv(inputs[i + self.start_level])
            for i, input_conv in enumerate(self.input_convs)
        ]

        original_f = []
        for t_i in laterals:
            temp_i = t_i
            original_f.append(temp_i)

        # modify_size for merging
        temp_i = 0
        merge_level = self.merge_level - self.start_level
        for t_i in range(len(laterals)):
            if t_i == merge_level:
                continue
            laterals[t_i] = self.modify1_convs[temp_i](laterals[t_i])
            temp_i += 1

        # merging features
        merge_inf = laterals[0]
        for t_i in laterals[1:]:
            temp = t_i
            merge_inf = jt.concat([merge_inf, temp], dim=1)

        # CBAM
        if self.CBAM is not None:
            merge_inf = merge_inf + self.CBAM_module(merge_inf)

        # fine features
        merge_inf = self.merge_conv(merge_inf)

        # modify for outing
        outs = []
        temp_i = 0
        for t_i in range(len(laterals)):
            if t_i == merge_level:
                outs.append(merge_inf)
            else:
                outs.append(self.modify2_convs[temp_i](merge_inf))
                temp_i += 1

        # group fine
        for t_i in range(len(outs)):

            if self.merge_type == "group":
                b, c, h, w = outs[t_i].shape
                outs[t_i] = jt.stack([original_f[t_i], outs[t_i]], dim=2)
                outs[t_i] = jt.reshape(outs[t_i], (b, -1, h, w))
                outs[t_i] = self.group_convs[t_i](outs[t_i])
            elif self.merge_type == "add":
                outs[t_i] = original_f[t_i] + outs[t_i]
            elif self.merge_type == "concat":
                outs[t_i] = jt.concat([outs[t_i], original_f[t_i]], dim=1)
                outs[t_i] = self.cat_convs[t_i](outs[t_i])

        for i in range(len(self.extra_convs)):
            outs.append(self.extra_convs[i](outs[-1]))
        return tuple(outs)

