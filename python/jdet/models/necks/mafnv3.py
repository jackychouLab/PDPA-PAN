import jittor as jt
from jittor import nn
from jdet.utils.registry import NECKS

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


class down_sample(nn.Module):
    def __init__(self, inchannel, down_step):
        super().__init__()
        self.down_conv = nn.Conv2d(inchannel, inchannel, kernel_size=down_step, stride=down_step, padding=0)
        self.norm = LayerNorm(inchannel, eps=1e-6, data_format="channels_first")
        self.gelu = nn.GELU()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.constant_(m.bias, 0)

    def execute(self, x):
        x = self.down_conv(x)
        x = self.norm(x)
        x = self.gelu(x)
        return x

class up_sample(nn.Module):
    def __init__(self, inchannel, up_step):
        super().__init__()
        self.up_conv = nn.ConvTranspose(inchannel,  inchannel, kernel_size=up_step, stride=up_step, padding=0)
        self.norm = LayerNorm(inchannel, eps=1e-6, data_format="channels_first")
        self.gelu = nn.GELU()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.constant_(m.bias, 0)

    def execute(self, x):
        x = self.up_conv(x)
        x = self.norm(x)
        x = self.gelu(x)
        return x

class SPP_G(nn.Module):
    def __init__(self, in_channel):
        super(SPP_G, self).__init__()
        self.max_pooling_5 = nn.MaxPool2d(5, stride=1, padding=2)
        self.max_pooling_9 = nn.MaxPool2d(9, stride=1, padding=4)
        self.max_pooling_13 = nn.MaxPool2d(13, stride=1, padding=6)
        self.group_conv = nn.Conv2d(in_channel * 4, in_channel, 3, stride=1, padding=1, groups=in_channel)
        self.norm = LayerNorm(in_channel, eps=1e-6, data_format="channels_first")
        self.gelu = nn.GELU()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def execute(self, x):
        m5 = self.max_pooling_5(x)
        m9 = self.max_pooling_9(x)
        m13 = self.max_pooling_13(x)
        b, c, h, w = x.shape
        x = jt.stack([x, m5, m9, m13], dim=2)
        x = jt.reshape(x, (b, -1, h, w))
        x = self.group_conv(x)
        x = self.norm(x)
        x = self.gelu(x)
        return x

class CSP_Neck_G(nn.Module):
    def __init__(self, in_channel):
        super(CSP_Neck_G, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, stride=1, padding=1),
            LayerNorm(in_channel, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(in_channel, in_channel, 3, stride=1, padding=1),
            LayerNorm(in_channel, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(in_channel, in_channel, 3, stride=1, padding=1),
        )
        self.conv2 = nn.Conv2d(in_channel, in_channel, 3, stride=1, padding=1)
        self.norm = LayerNorm(in_channel * 2, eps=1e-6, data_format="channels_first")
        self.act = nn.GELU()

        self.g_conv = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel, 3, stride=1, padding=1, groups=in_channel),
            LayerNorm(in_channel, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
        )

    def execute(self, x):
        b, c, h, w = x.shape
        original = x
        x = self.conv1(x)
        original = self.conv2(original)
        x = jt.stack([x, original], dim=2)
        x = jt.reshape(x, (b, -1, h, w))
        x = self.norm(x)
        x = self.act(x)
        x = self.g_conv(x)
        return x

@NECKS.register_module()
class MAFNv3(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 merge_level=2,
                 merge_conv_size=3,
                 merge_type="group",
                 out_conv_size=3):
        super(MAFNv3, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.out_conv_size = out_conv_size
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
        self.final_convs = nn.ModuleList()

        if self.merge_type == "group":
            self.group_convs = nn.ModuleList()
        if self.merge_type == "concat":
            self.cat_convs = nn.ModuleList()
        if self.merge_type == "add":
            self.add_convs = nn.ModuleList()
        if self.out_conv_size == "mix":
            self.out_size = 7
        for i in range(self.start_level, self.backbone_end_level):
            input_conv = nn.Sequential(
                nn.Conv2d(in_channels[i], out_channels, 3, stride=1, padding=1),
                LayerNorm(out_channels, eps=1e-6, data_format="channels_first"),
                nn.GELU(),
            )
            final_conv = nn.Sequential(
                *[CSP_Neck_G(out_channels)]
            )

            self.input_convs.append(input_conv)
            self.final_convs.append(final_conv)

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
                    group_conv = nn.Conv2d(out_channels * 2, out_channels, self.out_conv_size, stride=1, padding=self.out_conv_size//2, groups=out_channels)
                self.group_convs.append(group_conv)
            if self.merge_type == "concat":
                if self.out_conv_size == "mix":
                    cat_conv = nn.Conv2d(out_channels * 2, out_channels, self.out_size, stride=1, padding=self.out_size//2)
                    self.out_size -= 2
                else:
                    cat_conv = nn.Conv2d(out_channels * 2, out_channels, self.out_conv_size, stride=1, padding=self.out_conv_size//2)
                self.cat_convs.append(cat_conv)
            if self.merge_type == "add":
                if self.out_conv_size == "mix":
                    add_conv = nn.Conv2d(out_channels * 2, out_channels, self.out_size, stride=1, padding=self.out_size//2)
                    self.out_size -= 2
                else:
                    add_conv = nn.Conv2d(out_channels * 2, out_channels, self.out_conv_size, stride=1, padding=self.out_conv_size//2)
                self.add_convs.append(add_conv)

        self.merge_conv = nn.Sequential(
            nn.Conv2d(out_channels * (len(in_channels) - start_level), out_channels, self.merge_conv_size, stride=1, padding=self.merge_conv_size // 2),
            LayerNorm(out_channels, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            *[CSP_Neck_G(out_channels)],
            *[SPP_G(out_channels)],
            *[CSP_Neck_G(out_channels)],
        )

        self.extra_convs = nn.ModuleList()
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if extra_levels >= 1:
            for i in range(extra_levels):
                extra_conv = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2),
                    LayerNorm(out_channels, eps=1e-6, data_format="channels_first"),
                    nn.GELU(),
                )
                self.extra_convs.append(extra_conv)

        self.init_weights()

        if self.merge_type != "add":
            self.out_norm = LayerNorm(out_channels, eps=1e-6, data_format="channels_first")
            self.out_act = nn.GELU()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.constant_(m.bias, 0)

    def execute(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # 调整 channel
        laterals = [
            input_conv(inputs[i + self.start_level])
            for i, input_conv in enumerate(self.input_convs)
        ]

        original_f = []
        for t_i in laterals:
            temp_i = t_i
            original_f.append(temp_i)

        # 调整 H，W至融合大小
        temp_i = 0
        merge_level = self.merge_level - self.start_level
        for t_i in range(len(laterals)):
            if t_i == merge_level:
                continue
            laterals[t_i] = self.modify1_convs[temp_i](laterals[t_i])
            temp_i += 1

        # 对特征进行concat
        merge_inf = laterals[0]
        for t_i in laterals[1:]:
            temp = t_i
            merge_inf = jt.concat([merge_inf, temp], dim=1)

        # 进行特征融合
        merge_inf = self.merge_conv(merge_inf)

        # 恢复 H，W至原来大小
        outs = []
        temp_i = 0
        for t_i in range(len(laterals)):
            if t_i == merge_level:
                outs.append(merge_inf)
            else:
                outs.append(self.modify2_convs[temp_i](merge_inf))
                temp_i += 1

        # 将学习到的通用特征叠加到原本得特征图上
        for t_i in range(len(outs)):
            if self.merge_type == "group":
                b, c, h, w = outs[t_i].shape
                outs[t_i] = jt.stack([original_f[t_i], outs[t_i]], dim=2)
                outs[t_i] = jt.reshape(outs[t_i], (b, -1, h, w))
                outs[t_i] = self.group_convs[t_i](outs[t_i])
            elif self.merge_type == "add":
                outs[t_i] = original_f[t_i] + outs[t_i]
                outs[t_i] = self.add_convs[t_i](outs[t_i])
            elif self.merge_type == "concat":
                outs[t_i] = jt.concat([outs[t_i], original_f[t_i]], dim=1)
                outs[t_i] = self.cat_convs[t_i](outs[t_i])

        if self.merge_type != "add":
            for t_i in range(len(outs)):
                outs[t_i] = self.out_norm(outs[t_i])
                outs[t_i] = self.out_act(outs[t_i])

        # 对没一个特征进行深度提炼
        for t_i in range(len(outs)):
            outs[t_i] = self.final_convs[t_i](outs[t_i])

        for i in range(len(self.extra_convs)):
            outs.append(self.extra_convs[i](outs[-1]))

        return tuple(outs)

