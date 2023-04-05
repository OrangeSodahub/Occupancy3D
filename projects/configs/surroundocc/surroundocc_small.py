_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
occ_size = [200, 200, 16]
use_semantic = True
use_mask = False

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

class_names =  ['other', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
                'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface',
                'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation', 'free']

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = [128, 256]
_pos_dim_ = _dim_[0] // 2
_ffn_dim_ = [256, 512]
volume_h_ = [100, 50]
volume_w_ = [100, 50]
volume_z_ = [8, 4]
_num_points_ = [2, 4]
_num_layers_ = [1, 3]
_num_levels_ = 2
_num_featrue_levels_ = 3
queue_length = 2 # each sequence contains `queue_length` frames.

model = dict(
    type='SurroundOcc',
    use_grid_mask=True,
    use_semantic=use_semantic,
    img_backbone=dict(
       type='ResNet',
       depth=101,
       num_stages=4,
       out_indices=(2, 3),
       frozen_stages=1,
       norm_cfg=dict(type='BN2d', requires_grad=False),
       norm_eval=True,
       style='caffe',
       #with_cp=True, # using checkpoint to save GPU memory
       dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
       stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[1024, 2048],
        out_channels=512,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='OccHead',
        volume_h=volume_h_,
        volume_w=volume_w_,
        volume_z=volume_z_,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        num_query=900,
        num_classes=18,
        conv_input=[_dim_[1], 128, _dim_[0], 64, 64],
        conv_output=[128, _dim_[0], 64, 64, 32],
        out_indices=[0, 2, 4],
        upsample_strides=[1, 2, 1, 2, 1],
        embed_dims=_dim_,
        img_channels=[512, 512],
        use_semantic=use_semantic,
        use_mask=use_mask,
        transformer_template=dict(
            type='PerceptionTransformer',
            embed_dims=_dim_,
            num_feature_levels=_num_featrue_levels_,
            encoder=dict(
                type='OccEncoder',
                num_layers=_num_layers_,
                pc_range=point_cloud_range,
                return_intermediate=False,
                transformerlayers=dict(
                    type='OccLayer',
                    attn_cfgs=[
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=_num_points_,
                                num_levels=1),
                            embed_dims=_dim_,
                        ),
                        dict(
                            type='TemporalSelfAttention',
                            # TODO: verify, only use temporal attention
                            # upon the last volume feature map
                            embed_dims=_dim_[0]*volume_z_[0],
                            num_levels=1
                        ),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    embed_dims=_dim_,
                    conv_num=2,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm', 'conv')))),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            # TODO: verify
            num_feats=_pos_dim_*volume_z_[0],
            row_num_embed=volume_h_[0],
            col_num_embed=volume_w_[0],
        ),
    ),
)

dataset_type = 'CustomNuScenesOccDataset'
data_root = 'data/occ3d-nus/'
file_client_args = dict(backend='disk')
occ_gt_data_root='data/occ3d-nus'

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadOccupancy', data_root=occ_gt_data_root, use_semantic=use_semantic),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.8]),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D', keys=[ 'img', 'voxel_semantics', 'mask_lidar', 'mask_camera'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccupancy', data_root=occ_gt_data_root, use_semantic=use_semantic),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.8]),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D', keys=[ 'img'])
]

find_unused_parameters = True
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='data/occ3d-nus/occ_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        use_semantic=use_semantic,
        classes=class_names,
        queue_length=queue_length,
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
        data_root=data_root,
        ann_file='data/occ3d-nus/occ_infos_temporal_val.pkl',
        pipeline=test_pipeline,  
        occ_size=occ_size,
        pc_range=point_cloud_range,
        use_semantic=use_semantic,
        classes=class_names,
        modality=input_modality,
        eval_fscore=True),
    test=dict(type=dataset_type,
        data_root=data_root,
        ann_file='data/occ3d-nus/occ_infos_temporal_val.pkl',
        pipeline=test_pipeline, 
        occ_size=occ_size,
        pc_range=point_cloud_range,
        use_semantic=use_semantic,
        classes=class_names,
        modality=input_modality,
        eval_fscore=True),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24
evaluation = dict(interval=1, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1)