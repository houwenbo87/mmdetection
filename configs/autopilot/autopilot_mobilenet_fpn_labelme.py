# fp16 settings
#fp16 = dict(loss_scale=512.)
norm_cfg = dict(type='BN', requires_grad=False)

# model settings
model = dict(
    type='Autopilot',
    pretrained=None,
    backbone=dict(
        type='MobileNetV2',
        out_indices=(0, 1, 2, 4, 6),
        width_mult=1.0,
        frozen_stages=-1),
    neck=dict(
        type='FPN',
        in_channels=[16, 24, 32, 96, 320],
        out_channels=64,
        start_level=0,
        add_extra_convs=True,
        extra_convs_on_inputs=False,  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=10,
        in_channels=64,
        stacked_convs=2,
        feat_channels=32,
        strides=[2, 4, 8, 16, 32],
        #regress_ranges=((-1, 128), (128, 512), (512, 10000)),
        regress_ranges=((-1, 10000), (-1, 10000), (-1, 10000), (-1, 10000), (-1, 10000)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        norm_cfg=norm_cfg,
    ),
    parkspace_head=dict(
        type='ParkingspotsHead',
        num_classes=3,
        num_keypts=4,
        in_channels=64,
        stacked_convs=2,
        feat_channels=32,
        strides=[2], #[2, 4, 8, 16, 32],
        in_feat_index=[0],
        #regress_ranges=((-1, 128), (128, 512), (512, 10000)),
        regress_ranges=((-1, 10000)),
        loss_hm=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0),
        loss_hm_kp=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0),
        loss_kps=dict(
            type='SmoothL1Loss',
            reduction='sum',
            loss_weight=0.1),
        norm_cfg=norm_cfg,
    ),
)
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=True)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)
# dataset settings
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_parkspace=True), #with_freespace=False),
    dict(type='Resize', img_scale=[(360,360)], keep_ratio=True), #resize (w, h)
    dict(type='RandomFlip', flip_ratio=0),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_parkspaces', 'gt_parkspace_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='LoadAnnotations', with_bbox=False, with_keypt=False, with_obj_keypt=True, with_obj_label=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(960, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            #dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            #dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'gt_obj_keypts', 'gt_obj_labels']),
            dict(type='Collect', keys=['img']),
        ])
]
dataset_type = 'LabelmeDataset'
data_root = '/media/houwenbo/Workspace/data/dataset/apa/parking_slot/'
data = dict(
    imgs_per_gpu=8,
    workers_per_gpu=0, #12,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train.txt',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val.txt',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline)
)
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=5e-05)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11, 13])
# yapf:enable
# runtime settings
total_epochs = 32
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
#work_dir = './work_dirs/autopilot_mobilenet_fpn_labelme'
load_from = None
resume_from = None
workflow = [('train', 1)]
