import jittor as jt
from jittor import nn
import warnings
import random
from jdet.utils.registry import NECKS
from jdet.models.utils.modules import ConvModule
from jdet.models.utils.weight_init import xavier_init
from jdet.models.utils.attention import SPCAM_mix2_add
from jdet.models.utils.attention import SAM_mix_mix7

@NECKS.register_module()
class PAN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform'),
                 upsample_div_factor=1,
                 attention=False,
                 attention_type="CBAM",
                 attention_loc="out"):
        super(PAN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg = upsample_cfg.copy()
        self.upsample_div_factor = upsample_div_factor
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
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        if attention:
            self.attens = nn.ModuleList()
            self.attention = True
        else:
            self.attention = False
        if attention and (attention_type == "finish" or attention_type == "CBAM" or attention_type == "scSE" or attention_type == "BAM"):
            self.attention_f = True
            self.atten_fs = nn.ModuleList()
        else:
            self.attention_f = False


        for i in range(self.start_level, self.backbone_end_level):
            if attention:
                if attention_loc == "out":
                    atten_channel = in_channels[i]
                    self.loc_in = False
                if attention_loc == "in":
                    atten_channel = out_channels
                    self.loc_in = True
                if attention_type == "CAM_avg":
                    atten = CAM_avg(atten_channel)
                if attention_type == "CAM_max":
                    atten = CAM_max(atten_channel)
                if attention_type == "CAM_mix":
                    atten = CAM_mix(atten_channel)
                if attention_type == "SPCAM_avg3":
                    atten = SPCAM_avg3(atten_channel)
                if attention_type == "SPCAM_max3":
                    atten = SPCAM_max3(atten_channel)
                if attention_type == "SPCAM_avg2":
                    atten = SPCAM_avg2(atten_channel)
                if attention_type == "SPCAM_max2":
                    atten = SPCAM_max2(atten_channel)
                if attention_type == "SPCAM_mix2":
                    atten = SPCAM_mix2(atten_channel)
                if attention_type == "SPCAM_mix3":
                    atten = SPCAM_mix3(atten_channel)
                if attention_type == "SAM_avg":
                    atten = SAM_avg(atten_channel)
                if attention_type == "SAM_max":
                    atten = SAM_max(atten_channel)
                if attention_type == "SAM_mix_add":
                    atten = SAM_mix_add(atten_channel)
                if attention_type == "SAM_mix_concat":
                    atten = SAM_mix_concat(atten_channel)
                if attention_type == "SPCAM_mix2_add":
                    atten = SPCAM_mix2_add(atten_channel)
                if attention_type == "SPCAM_mix3_add":
                    atten = SPCAM_mix3_add(atten_channel)
                if attention_type == "SAM_mix_mix":
                    atten = SAM_mix_mix(atten_channel)
                if attention_type == "SAM_GE":
                    atten = SAM_GE(atten_channel)
                if attention_type == "SAM_GE_new":
                    atten = SAM_GE_new(atten_channel)
                if attention_type == "SC":
                    atten = SC(atten_channel)
                if attention_type == "CS":
                    atten = CS(atten_channel)
                if attention_type == "PA":
                    atten = PA(atten_channel)
                if attention_type == "SAM_mix_mix5":
                    atten = SAM_mix_mix5(atten_channel)
                if attention_type == "SAM_mix_mix7":
                    atten = SAM_mix_mix7(atten_channel)
                if attention_type == "SAM_mix_mix9":
                    atten = SAM_mix_mix9(atten_channel)
                if attention_type == "finish":
                    atten = PA(atten_channel)
                    atten_f = PA(out_channels)
                    self.atten_fs.append(atten_f)
                if attention_type == "CBAM":
                    atten = CBAM(atten_channel)
                    atten_f = CBAM(out_channels)
                    self.atten_fs.append(atten_f)
                if attention_type == "scSE":
                    atten = scSE(atten_channel)
                    atten_f = scSE(out_channels)
                    self.atten_fs.append(atten_f)
                if attention_type == "BAM":
                    atten = BAM(out_channels)
                    atten_f = BAM(out_channels)
                    self.atten_fs.append(atten_f)
                self.attens.append(atten)
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                self.fpn_convs.append(extra_fpn_conv)

        self.down_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level - 1):
            down_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.down_convs.append(down_conv)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def execute(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        if self.attention:
            if self.loc_in:
                laterals = [
                    atten(lateral_conv(inputs[i + self.start_level]))
                    for i, lateral_conv, atten in zip(range(len(self.lateral_convs)), self.lateral_convs, self.attens)
                ]
            else:
                laterals = [
                    lateral_conv(atten(inputs[i + self.start_level]))
                    for i, lateral_conv, atten in zip(range(len(self.lateral_convs)), self.lateral_convs, self.attens)
                ]
        else:
            laterals = [
                lateral_conv(inputs[i + self.start_level])
                for i, lateral_conv in enumerate(self.lateral_convs)
            ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += nn.interpolate(laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += nn.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)
            laterals[i - 1] /= self.upsample_div_factor

        # build down-top path second
        for i, down_conv in zip(range(1, used_backbone_levels), self.down_convs):
            laterals[i] += down_conv(laterals[i - 1]) + laterals[i]

        # build outputs
        # part 1: from original levels
        if self.attention_f:
            outs = [
                atten_f(self.fpn_convs[i](laterals[i]))
                for i,atten_f in zip(range(used_backbone_levels), self.atten_fs)
            ]
        else:
            outs = [
                self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
            ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(nn.pool(outs[-1], 1, stride=2,op="maximum"))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](nn.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

class CS(nn.Module):
    def __init__(self, channel):
        super(CS, self).__init__()
        self.c_attention = SPCAM_mix2_add(channel)
        self.s_attention = SAM_mix_mix7(channel)

    def execute(self, x):
        x = self.c_attention(x)
        x = self.s_attention(x)

        return x

class SC(nn.Module):
    def __init__(self, channel):
        super(SC, self).__init__()
        self.c_attention = SPCAM_mix2_add(channel)
        self.s_attention = SAM_mix_mix7(channel)

    def execute(self, x):
        x = self.s_attention(x)
        x = self.c_attention(x)

        return x


class PA(nn.Module):
    def __init__(self, channel):
        super(PA, self).__init__()
        self.c_attention = SPCAM_mix2_add(channel)
        self.s_attention = SAM_mix_mix7(channel)

    def execute(self, x):

        return self.c_attention(x) + self.s_attention(x)
