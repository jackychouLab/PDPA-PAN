# model settings
model = dict(
    type='FasterRCNNOBB',
    pretrained='modelzoo://resnet50',
    backbone=dict(
        type='Resnet50',
        frozen_stages=1,
        return_stages=["layer1","layer2","layer3","layer4"],
        pretrained=True),
    neck=dict(
        type='PAN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs=False,
        num_outs=5),
    rpn_head=dict(
        type='FasterrcnnHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLossForRcnn', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='ROIAlign', output_size=7, sampling_ratio=2, version=1),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHeadRbbox',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=5,
        target_means=[0., 0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2, 0.1],
        reg_class_agnostic=False,
        with_module=False,
        hbb_trans='hbbpolyobb',
        loss_cls=dict(
            type='CrossEntropyLossForRcnn', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
# model training and testing settings
    train_cfg = dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='BboxOverlaps2D_v1')),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_num=2000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='BboxOverlaps2D_v1')),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg = dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_num=2000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=dict(
            # score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=1000)
        score_thr = 0.05, nms = dict(type='py_cpu_nms_poly_fast', iou_thr=0.1), max_per_img = 2000)
    # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
    )
)
# dataset settings
train_path = "/media/lab315-server/dev/zjh_datasets/L315_vehicle_640/trainval_640_160_1.0"
val_path = "/media/lab315-server/dev/zjh_datasets/L315_vehicle_640/val_640_160_1.0"
test_path = "/media/lab315-server/dev/zjh_datasets/L315_vehicle_640/test_640_160_1.0/images/"

dataset = dict(
    train=dict(
        type="Vehicle4Dataset",
        dataset_dir=train_path,
        transforms=[
            dict(
                type="RotatedResize",
                min_size=640,
                max_size=640
            ),
            dict(type='RotatedRandomFlip', prob=0.5),
            dict(
                type="RandomRotateAug",
                random_rotate_on=True,
            ),
            dict(
                type = "Pad",
                size_divisor=32),
            dict(
                type = "Normalize",
                mean =  [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=False,)
            
        ],
        batch_size=2,
        num_workers=4,
        shuffle=True,
        filter_empty_gt=False,
    ),
    val=dict(
        type="Vehicle4Dataset",
        dataset_dir=val_path,
        transforms=[
            dict(
                type="RotatedResize",
                min_size=640,
                max_size=640
            ),
            dict(
                type = "Pad",
                size_divisor=32),
            dict(
                type = "Normalize",
                mean =  [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=False),
        ],
        batch_size=2,
        num_workers=4,
        shuffle=False
    ),
    test=dict(
        type="ImageDataset",
        dataset_type="vehicle4",
        images_dir=test_path,
        transforms=[
            dict(
                type="RotatedResize",
                min_size=640,
                max_size=640
            ),
            dict(
                type = "Pad",
                size_divisor=32),
            dict(
                type = "Normalize",
                mean =  [123.675, 116.28, 103.53],
                std = [58.395, 57.12, 57.375],
                to_bgr=False,),
        ],
        num_workers=4,
        batch_size=1,
    )
)
# optimizer
optimizer = dict(
    type='SGD', 
    lr=0.01, 
    momentum=0.9, 
    weight_decay=0.0001,
    grad_clip=dict(
        max_norm=35, 
        norm_type=2))

# learning policy
scheduler = dict(
    type='StepLR',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
#    milestones=[7, 10],
    milestones=[15, 21]
)

# yapf:disable
logger = dict(
    type="RunLogger")
# yapf:enable
# runtime settings
max_epoch = 24
eval_interval = 24
checkpoint_interval = 1
log_interval = 50
