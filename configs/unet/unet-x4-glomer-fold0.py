_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py',
    '../_base_/datasets/glomer_x4_fold0.py',
    '../_base_/default_runtime.py'
]

crop_size = (256, 256)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[163.304, 126.902, 173.385], # train rgb mean
    std=[48.577, 61.773, 46.175], # train rgb std
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size
)

model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(mode='whole')
)

iters = 80000

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=iters,
        by_epoch=False)
]

# settiing logger
log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs={'project': 'mmsegmentation'},
            interval=10,
            log_dir='./work_dirs/UNet_log'
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
