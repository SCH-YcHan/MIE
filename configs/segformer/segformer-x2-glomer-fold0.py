_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/glomer_x2_fold0.py',
    '../_base_/default_runtime.py'
]

crop_size = (256, 256)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[152.638, 121.595, 159.898], # train rgb mean
    std=[67.481, 71.658, 68.081], # train rgb std
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size
)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint))
)

iters = 80000
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=iters,
        by_epoch=False,
    )
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
            log_dir='./work_dirs/SegFormer_log'
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