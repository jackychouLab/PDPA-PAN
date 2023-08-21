import jittor as jt
from jittor import nn
import warnings

from jdet.utils.registry import NECKS
from jdet.models.utils.modules import ConvModule
from jdet.models.utils.weight_init import xavier_init

class neck_bottle(nn.Module):
    def __init__(self, channels):
        super(neck_bottle, self).__init__()
        self.relu = nn.Relu()
        self.bottle_conv1 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm(channels)
        self.bottle_conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(channels)

    def execute(self, x):
        x = self.bottle_conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.bottle_conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class neck_bottleX3(nn.Module):
    def __init__(self, channels):
        super(neck_bottleX3, self).__init__()
        self.bottle1 = neck_bottle(channels)
        self.bottle2 = neck_bottle(channels)
        self.bottle3 = neck_bottle(channels)

    def execute(self, x):
        x = self.bottle1(x)
        x = self.bottle2(x)
        x = self.bottle3(x)
        return x

class silu(nn.Module):
    def __init__(self):
        super(silu, self).__init__()
        self.sigmoid = nn.Sigmoid()
    def execute(self, x):
        x = self.sigmoid(x) * x
        return x

@NECKS.register_module()
class YOLOv5Neck(nn.Module):
    r"""Feature Pyramid Network.
    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.
    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed
            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.
    Example:
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [jt.randn(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = ([1, 11, 340, 340])
        outputs[1].shape = ([1, 11, 170, 170])
        outputs[2].shape = ([1, 11, 84, 84])
        outputs[3].shape = ([1, 11, 43, 43])
    """

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
                 upsample_div_factor=1):
        super(YOLOv5Neck, self).__init__()
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
        self.neck_channel = [128, 256, 512, 1024]
        temp = 0
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                self.neck_channel[temp],
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg)
            self.lateral_convs.append(l_conv)
            temp += 1


        self.relu = silu()
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.neck_conv1_1 = nn.Conv2d(4096, 1024, kernel_size=1, stride=1, bias=False)
        self.bn1_1 = nn.BatchNorm(1024)
        self.neck_conv1_2 = nn.Conv2d(1024, 512 , kernel_size=1, stride=1, bias=False)
        self.bn1_2 = nn.BatchNorm(512)

        self.neck_conv2_1_1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False)
        self.bn2_1_1 = nn.BatchNorm(512)
        self.neck_conv2_1_2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False)
        self.bn2_1_2 = nn.BatchNorm(512)
        self.bot1 = neck_bottleX3(512)
        self.neck_conv2_2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False)
        self.bn2_2 = nn.BatchNorm(512)
        self.neck_conv2_3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False)
        self.bn2_3 = nn.BatchNorm(256)

        self.neck_conv3_1_1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False)
        self.bn3_1_1 = nn.BatchNorm(256)
        self.neck_conv3_1_2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False)
        self.bn3_1_2 = nn.BatchNorm(256)
        self.bot2 = neck_bottleX3(256)
        self.neck_conv3_2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False)
        self.bn3_2 = nn.BatchNorm(256)
        self.neck_conv3_3 = nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False)
        self.bn3_3 = nn.BatchNorm(128)

        self.neck_conv4_1_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False)
        self.bn4_1_1 = nn.BatchNorm(128)
        self.neck_conv4_1_2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False)
        self.bn4_1_2 = nn.BatchNorm(128)
        self.bot3 = neck_bottleX3(128)
        self.neck_conv4_2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False)
        self.bn4_2 = nn.BatchNorm(128)

        self.neck_conv5_1 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1,  bias=False)
        self.bn5_1 = nn.BatchNorm(128)
        self.neck_conv5_2_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False)
        self.bn5_2_1 = nn.BatchNorm(128)
        self.neck_conv5_2_2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False)
        self.bn5_2_2 = nn.BatchNorm(128)
        self.bot4 = neck_bottleX3(128)
        self.neck_conv5_3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False)
        self.bn5_3 = nn.BatchNorm(256)

        self.neck_conv6_1 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn6_1 = nn.BatchNorm(256)
        self.neck_conv6_2_1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False)
        self.bn6_2_1 = nn.BatchNorm(256)
        self.neck_conv6_2_2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False)
        self.bn6_2_2 = nn.BatchNorm(256)
        self.bot5 = neck_bottleX3(256)
        self.neck_conv6_3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False)
        self.bn6_3 = nn.BatchNorm(512)

        self.neck_conv7_1 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn7_1 = nn.BatchNorm(512)
        self.neck_conv7_2_1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False)
        self.bn7_2_1 = nn.BatchNorm(512)
        self.neck_conv7_2_2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False)
        self.bn7_2_2 = nn.BatchNorm(512)
        self.bot6 = neck_bottleX3(512)
        self.neck_conv7_3 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, bias=False)
        self.bn7_3 = nn.BatchNorm(1024)

        self.out0 = nn.Conv2d(128, out_channels, kernel_size=1, stride=1, bias=False)
        self.out1 = nn.Conv2d(256, out_channels, kernel_size=1, stride=1, bias=False)
        self.out2 = nn.Conv2d(512, out_channels, kernel_size=1, stride=1, bias=False)
        self.out3 = nn.Conv2d(1024, out_channels, kernel_size=1, stride=1, bias=False)
        self.out4_1 = nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1, bias=False)
        self.obn4_1 = nn.BatchNorm(1024)
        self.out4_2 = nn.Conv2d(1024, out_channels, kernel_size=1, stride=1, bias=False)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def execute(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # part 1: build top-down path
        used_backbone_levels = len(laterals)
        max1 = self.maxpool(laterals[used_backbone_levels - 1])
        max2 = self.maxpool(max1)
        max3 = self.maxpool(max2)
        laterals[used_backbone_levels - 1] = self.relu(self.bn1_1(self.neck_conv1_1(jt.concat([laterals[used_backbone_levels - 1], max1, max2, max3], dim=1))))
        laterals[used_backbone_levels - 1] = self.relu(self.bn1_2(self.neck_conv1_2(laterals[used_backbone_levels - 1])))

        # upsample start
        prev_shape = laterals[used_backbone_levels - 2].shape[2:]
        temp = nn.interpolate(laterals[used_backbone_levels - 1], size=prev_shape, **self.upsample_cfg)
        # upsample stop
        laterals[used_backbone_levels - 2] = jt.concat([laterals[used_backbone_levels - 2], temp], dim=1)
        temp2 = laterals[used_backbone_levels - 2]
        temp2 = self.relu(self.bn2_1_2(self.neck_conv2_1_2(temp2)))
        laterals[used_backbone_levels - 2] = self.bot1(self.relu(self.bn2_1_1(self.neck_conv2_1_1(laterals[used_backbone_levels - 2]))))
        laterals[used_backbone_levels - 2] = self.relu(self.bn2_2(self.neck_conv2_2(jt.concat([laterals[used_backbone_levels - 2], temp2], dim=1))))
        laterals[used_backbone_levels - 2] = self.relu(self.bn2_3(self.neck_conv2_3(laterals[used_backbone_levels - 2])))

        # upsample start
        prev_shape = laterals[used_backbone_levels - 3].shape[2:]
        temp = nn.interpolate(laterals[used_backbone_levels - 2], size=prev_shape, **self.upsample_cfg)
        # upsample stop
        laterals[used_backbone_levels - 3] = jt.concat([laterals[used_backbone_levels - 3], temp], dim=1)
        temp3 = laterals[used_backbone_levels - 3]
        temp3 = self.relu(self.bn3_1_2(self.neck_conv3_1_2(temp3)))
        laterals[used_backbone_levels - 3] = self.bot2(self.relu(self.bn3_1_1(self.neck_conv3_1_1(laterals[used_backbone_levels - 3]))))
        laterals[used_backbone_levels - 3] = self.relu(self.bn3_2(self.neck_conv3_2(jt.concat([laterals[used_backbone_levels - 3], temp3], dim=1))))
        laterals[used_backbone_levels - 3] = self.relu(self.bn3_3(self.neck_conv3_3(laterals[used_backbone_levels - 3])))

        # upsample start
        prev_shape = laterals[used_backbone_levels - 4].shape[2:]
        temp = nn.interpolate(laterals[used_backbone_levels - 3], size=prev_shape, **self.upsample_cfg)
        # upsample stop
        laterals[used_backbone_levels - 4] = jt.concat([laterals[used_backbone_levels - 4], temp], dim=1)
        temp4 = laterals[used_backbone_levels - 4]
        temp4 = self.relu(self.bn4_1_2(self.neck_conv4_1_2(temp4)))
        laterals[used_backbone_levels - 4] = self.bot3(self.relu(self.bn4_1_1(self.neck_conv4_1_1(laterals[used_backbone_levels - 4]))))
        laterals[used_backbone_levels - 4] = self.relu(self.bn4_2(self.neck_conv4_2(jt.concat([laterals[used_backbone_levels - 4], temp4], dim=1))))

        laterals[used_backbone_levels - 3] = jt.concat([laterals[used_backbone_levels - 3], self.relu(self.bn5_1(self.neck_conv5_1(laterals[used_backbone_levels - 4])))], dim=1)
        temp5 = laterals[used_backbone_levels - 3]
        temp5 = self.relu(self.bn5_2_2(self.neck_conv5_2_2(temp5)))
        laterals[used_backbone_levels - 3] = self.bot4(self.relu(self.bn5_2_1(self.neck_conv5_2_1(laterals[used_backbone_levels - 3]))))
        laterals[used_backbone_levels - 3] = self.relu(self.bn5_3(self.neck_conv5_3(jt.concat([laterals[used_backbone_levels - 3], temp5], dim=1))))

        laterals[used_backbone_levels - 2] = jt.concat([laterals[used_backbone_levels - 2], self.relu(self.bn6_1(self.neck_conv6_1(laterals[used_backbone_levels - 3])))], dim=1)
        temp6 = laterals[used_backbone_levels - 2]
        temp6 = self.relu(self.bn6_2_2(self.neck_conv6_2_2(temp6)))
        laterals[used_backbone_levels - 2] = self.bot5(self.relu(self.bn6_2_1(self.neck_conv6_2_1(laterals[used_backbone_levels - 2]))))
        laterals[used_backbone_levels - 2] = self.relu(self.bn6_3(self.neck_conv6_3(jt.concat([laterals[used_backbone_levels - 2], temp6], dim=1))))

        laterals[used_backbone_levels - 1] = jt.concat([laterals[used_backbone_levels - 1], self.relu(self.bn7_1(self.neck_conv7_1(laterals[used_backbone_levels - 2])))], dim=1)
        temp7 = laterals[used_backbone_levels - 1]
        temp7 = self.relu(self.bn7_2_2(self.neck_conv7_2_2(temp7)))
        laterals[used_backbone_levels - 1] = self.bot6(self.relu(self.bn7_2_1(self.neck_conv7_2_1(laterals[used_backbone_levels - 1]))))
        laterals[used_backbone_levels - 1] = self.relu(self.bn7_3(self.neck_conv7_3(jt.concat([laterals[used_backbone_levels - 1], temp7], dim=1))))

        outs = [
            laterals[i] for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        outs.append(self.out4_2(self.relu(self.obn4_1(self.out4_1(outs[-1])))))

        # part 3: adjust channel
        # pop bottom 128
        outs[0] = self.out0(laterals[used_backbone_levels - 4])
        # pop bottom - 1 256
        outs[1] = self.out1(laterals[used_backbone_levels - 3])
        # pop bottom - 2 512
        outs[2] = self.out2(laterals[used_backbone_levels - 2])
        # pop bottom - 3 1024
        outs[3] = self.out3(laterals[used_backbone_levels - 1])

        return tuple(outs)

