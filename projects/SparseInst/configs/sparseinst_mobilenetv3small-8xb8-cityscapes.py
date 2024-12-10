_base_ = [
    'mmdet::_base_/datasets/cityscapes_instance.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['projects.SparseInst.sparseinst', 'mmpretrain.models'],
    allow_failed_imports=False
)

# Define the instance classes (the standard 8 instance classes in Cityscapes)
classes = [
    'person', 'rider', 'car', 'truck', 'bus',
    'train', 'motorcycle', 'bicycle'
]

model = dict(
    type='SparseInst',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32
    ),
    backbone=dict(
        type='mmpretrain.TIMMBackbone',
        model_name='mobilenetv3_small_100',
        features_only=True,
        out_indices=(2, 3, 4),
        pretrained=True,
        num_classes=0,
        global_pool=''
    ),
    encoder=dict(
        type='InstanceContextEncoder',
        in_channels=[24, 48, 576],
        out_channels=256
    ),
    decoder=dict(
        type='BaseIAMDecoder',
        in_channels=256 + 2,
        num_classes=len(classes),  # Use the length of the classes list
        ins_dim=256,
        ins_conv=4,
        mask_dim=256,
        mask_conv=4,
        kernel_dim=128,
        scale_factor=2.0,
        output_iam=False,
        num_masks=100
    ),
    criterion=dict(
        type='SparseInstCriterion',
        num_classes=len(classes),
        assigner=dict(type='SparseInstMatcher', alpha=0.8, beta=0.2),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            alpha=0.25,
            gamma=2.0,
            reduction='sum',
            loss_weight=2.0
        ),
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=1.0
        ),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0
        ),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            reduction='sum',
            eps=5e-5,
            loss_weight=2.0
        )
    ),
    test_cfg=dict(score_thr=0.005, mask_thr_binary=0.45)
)

backend = 'pillow'
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}, imdecode_backend=backend),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
    dict(
        type='RandomChoiceResize',
        scales=[(512, 1024), (768, 1536)],
        keep_ratio=True,
        backend=backend
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}, imdecode_backend=backend),
    dict(type='Resize', scale=(1024, 2048), keep_ratio=True, backend=backend),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    sampler=dict(type='InfiniteSampler'),
    # dataset=dict(pipeline=train_pipeline)
)

test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
val_dataloader = test_dataloader

# If the base config used a list for evaluators, we keep it a list here too.
# val_evaluator = [dict(metric='segm')]
# test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(_delete_=True, type='AdamW', lr=0.00005, weight_decay=0.05)
)

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=23720,
    # max_epochs=64,
    val_interval=1000
)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=64,
        by_epoch=True,
        milestones=[7*8],
        gamma=0.1)
]


default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=10000, max_keep_ckpts=3)
)
log_processor = dict(by_epoch=False)

auto_scale_lr = dict(base_batch_size=64, enable=True)

_base_.visualizer.vis_backends = [
    dict(
        type='WandbVisBackend',
        init_kwargs={'project': 'mmdet-sparseinst-cityscapes'}
    ),
]
