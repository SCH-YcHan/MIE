# File path: mmsegmentation/configs/pidnet/pidnet-s_2xb6-120k_256x256-glomer.py
_base_ = [
    '../_base_/datasets/glomer_x4_fold1.py',
    '../_base_/default_runtime.py'
]

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/pidnet/pidnet-s_imagenet1k_20230306-715e6273.pth'
crop_size = (256, 256)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[162.428, 126.460, 173.635], # train rgb mean
    std=[47.576, 60.515, 44.643], # train rgb std
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size
)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='PIDNet',
        in_channels=3,
        channels=32,
        ppm_channels=96,
        num_stem_blocks=2,
        num_branch_blocks=3,
        align_corners=False,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    decode_head=dict(
        type='PIDHead',
        in_channels=128,
        channels=128,
        num_classes=2,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        align_corners=True,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=0.4
            ),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                loss_weight=1.0
            ),
            dict(type='BoundaryLoss', loss_weight=20.0),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                loss_weight=1.0
            )
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# optimizer
iters = 80000
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=iters,
        by_epoch=False)
]

# setting visualizer
visualizer = dict(
    type="Visualizer", 
    vis_backends=[dict(type='LocalVisBackend'),
                      dict(type="WandbVisBackend")]
)

# settiing logger
log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs={'project': 'mmsegmentation'},
            interval=10,
            log_dir='./work_dirs/PIDNet_log'
        ),
    ]
)

log_processor = dict(
    type='LogProcessor',
    window_size=50,
    by_epoch=True
)

# setting trian test cfg
train_cfg = dict(
    type='IterBasedTrainLoop', 
    max_iters=iters, 
    val_interval=iters // 10
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# setting hook
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', 
        interval=iters // 10, 
        by_epoch=False
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1)
)

randomness = dict(seed=42)