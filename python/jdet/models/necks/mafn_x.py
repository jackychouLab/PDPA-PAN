import jittor as jt
from jittor import nn
from jdet.utils.registry import NECKS
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

class down_sample(nn.Module):
    def __init__(self, inchannel, down_step):
        super(down_sample, self).__init__()
        self.conv = nn.Conv2d(inchannel, inchannel, kernel_size=down_step, stride=down_step, padding=0)
        self.norm = LayerNorm(inchannel, eps=1e-6, data_format="channels_first")
        self.act = nn.GELU()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def execute(self, x):
        return self.act(self.norm(self.conv(x)))

class up_sample(nn.Module):
    def __init__(self, inchannel, up_step):
        super(up_sample, self).__init__()
        self.conv = nn.ConvTranspose(inchannel,  inchannel, kernel_size=up_step, stride=up_step, padding=0)
        self.norm = LayerNorm(inchannel, eps=1e-6, data_format="channels_first")
        self.act = nn.GELU()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def execute(self, x):
        return self.act(self.norm(self.conv(x)))

class MAF_module(nn.Module):
    def __init__(self, in_channel, merge_type, SPP_type, refine_type):
        super(MAF_module, self).__init__()
        self.merge_in_type = merge_type
        self.SPP_type = SPP_type
        self.merge_out_type =  refine_type
        self.max_pooling_5 = nn.MaxPool2d(5, stride=1, padding=2)
        self.max_pooling_9 = nn.MaxPool2d(9, stride=1, padding=4)
        self.max_pooling_13 = nn.MaxPool2d(13, stride=1, padding=6)
        self.CLGs = nn.ModuleList()

        if merge_type == "concat" or merge_type == "group":
            clg = CLG_conv(in_channel * 3, in_channel)
            self.CLGs.append(clg)
        if merge_type == "add":
            clg = CLG_conv(in_channel, in_channel)
            self.CLGs.append(clg)

        for _ in range(2):
            clg = CLG_conv(in_channel, in_channel)
            self.CLGs.append(clg)

        if SPP_type == "concat" or SPP_type == "group":
            clg = CLG_conv(in_channel * 4, in_channel)
            self.CLGs.append(clg)
        if SPP_type == "add":
            clg = CLG_conv(in_channel, in_channel)
            self.CLGs.append(clg)

        if refine_type == "concat" or refine_type == "group":
            clg = CLG_conv(in_channel * 2, in_channel)
            self.CLGs.append(clg)
        if refine_type == "add":
            clg = CLG_conv(in_channel, in_channel)
            self.CLGs.append(clg)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def execute(self, x):
        exc = self.CLGs[0](x)
        x = self.CLGs[1](exc)
        x = self.CLGs[2](x)
        m5 = self.max_pooling_5(x)
        m9 = self.max_pooling_9(x)
        m13 = self.max_pooling_13(x)
        if self.SPP_type == "group":
            b, c, h, w = x.shape
            x = jt.stack([x, m5, m9, m13], dim=2)
            x = jt.reshape(x, (b, -1, h, w))
        if self.SPP_type == "concat":
            x = jt.concat([x, m5, m9, m13], dim=1)
        if self.SPP_type == "add":
            x = x + m5 + m9 + m13
        x = self.CLGs[3](x)

        if self.merge_out_type == "group":
            b, c, h, w = x.shape
            x = jt.stack([x, exc], dim=2)
            x = jt.reshape(x, (b, -1, h, w))
        elif self.merge_out_type == "concat":
            x = jt.concat([x, exc], dim=1)
        elif self.SPP_type == "add":
            x = x + exc
        x = self.CLGs[4](x)

        return x


class refine_module(nn.Module):
    def __init__(self, in_channel, out_channel, refine_type):
        super(refine_module, self).__init__()
        self.refine_type = refine_type
        self.CLGs = nn.ModuleList()
        clg = CLG_conv(in_channel, out_channel)
        self.CLGs.append(clg)
        for _ in range(2):
            clg = CLG_conv(out_channel, out_channel)
            self.CLGs.append(clg)
        if refine_type == "add":
            clg = CLG_conv(out_channel, out_channel)
            self.CLGs.append(clg)
        if refine_type == "concat" or refine_type == "group":
            clg = CLG_conv(out_channel * 2, out_channel)
            self.CLGs.append(clg)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def execute(self, x):
        exc = self.CLGs[0](x)
        x = self.CLGs[1](exc)
        x = self.CLGs[2](x)
        if self.refine_type == "group":
            b, c, h, w = x.shape
            x = jt.stack([x, exc], dim=2)
            x = jt.reshape(x, (b, -1, h, w))
        elif self.refine_type == "concat":
            x = jt.concat([x, exc], dim=1)
        elif self.refine_type == "add":
            x = x + exc
        x = self.CLGs[3](x)
        return x

class CLG_conv(nn.Module):
    def __init__(self, inchannel, outchannel, conv_size=3):
        super(CLG_conv, self).__init__()
        self.conv = nn.Conv2d(inchannel, outchannel, conv_size, stride=1, padding=int((conv_size-1)/2))
        self.norm = LayerNorm(outchannel, eps=1e-6, data_format="channels_first")
        self.act = nn.GELU()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def execute(self, x):
        x = self.act(self.norm(self.conv(x)))
        return x

@NECKS.register_module()
class MAFNX(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 merge_level=2,
                 merge_type="group",
                 spp_type="concat",
                 refine_type="add"):
        super(MAFNX, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
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
        self.input_convs = nn.ModuleList()
        self.modify1_convs = nn.ModuleList()
        self.mafn_module = MAF_module(out_channels, merge_type, spp_type, refine_type)
        self.modify2_convs = nn.ModuleList()
        self.refine_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            input_conv = CLG_conv(in_channels[i], out_channels, 1)
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

            # if merge_type == "concat" or merge_type == "group":
            #     refine_conv = refine_module(out_channels * 2, out_channels, refine_type)
            #     self.refine_convs.append(refine_conv)
            # elif merge_type == "add":
            if True:
                refine_conv = refine_module(out_channels, out_channels, refine_type)
                self.refine_convs.append(refine_conv)

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

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

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
        if self.merge_type == "add":
            merge_inf = merge_inf + laterals[1] + laterals[2]
        if self.merge_type == "concat":
            merge_inf = jt.concat([merge_inf, laterals[1], laterals[2]], dim=1)
        if self.merge_type == "group":
            b, c, h, w = laterals[0].shape
            merge_inf = jt.stack([merge_inf, laterals[1], laterals[2]], dim=2)
            merge_inf = jt.reshape(merge_inf, (b, -1, h, w))

        # 进行特征融合
        merge_inf = self.mafn_module(merge_inf)

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
            # if self.merge_type == "group":
            #     b, c, h, w = outs[t_i].shape
            #     outs[t_i] = jt.stack([original_f[t_i], outs[t_i]], dim=2)
            #     outs[t_i] = jt.reshape(outs[t_i], (b, -1, h, w))
            #     outs[t_i] = self.refine_convs[t_i](outs[t_i])
            # elif self.merge_type == "add":
            #     outs[t_i] = original_f[t_i] + outs[t_i]
            #     outs[t_i] = self.refine_convs[t_i](outs[t_i])
            # elif self.merge_type == "concat":
            #     outs[t_i] = jt.concat([outs[t_i], original_f[t_i]], dim=1)
            outs[t_i] = self.refine_convs[t_i](outs[t_i])

        for i in range(len(self.extra_convs)):
            outs.append(self.extra_convs[i](outs[-1]))

        return tuple(outs)

