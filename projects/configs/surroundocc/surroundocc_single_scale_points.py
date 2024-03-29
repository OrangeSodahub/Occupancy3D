_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
len_queue=4
point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
occ_size = [200, 200, 16]
use_semantic = True
use_mask = False
use_points = True # knowledge distillation
use_sequential = False # test pipeline

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

class_names =  ['other', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
                'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface',
                'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation', 'free']
class_weight = [0.05597741, 0.05857186, 0.07012177, 0.05821387, 0.05237201,
                0.06030229, 0.0685634 , 0.05849956, 0.06577655, 0.05758299,
                0.05514106, 0.04643295, 0.05634901, 0.04929424, 0.04858398,
                0.04741097, 0.04701869, 0.03516321]

input_modality = dict(
    use_lidar=use_points,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = [128, 256, 512]
_ffn_dim_ = [256, 512, 1024]
volume_h_ = [100, 50, 25]
volume_w_ = [100, 50, 25]
volume_z_ = [8, 4, 2]
_num_points_ = [2, 4, 8]
_num_layers_ = [1, 3, 6]

model = dict(
    type='SurroundOcc',
    use_grid_mask=True,
    use_semantic=use_semantic,
    use_points=use_points,
    img_backbone=dict(
       type='ResNet',
       depth=101,
       num_stages=4,
       out_indices=(1,2,3),
       frozen_stages=1,
       norm_cfg=dict(type='BN2d', requires_grad=False),
       norm_eval=True,
       style='caffe',
       #with_cp=True, # using checkpoint to save GPU memory
       dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
       stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=512,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=3,
        relu_before_extra_convs=True),
    pts_voxel_layer=dict(
        max_num_points=10, 
        point_cloud_range=point_cloud_range,
        voxel_size=[0.1, 0.1, 0.1],  # xy size follow centerpoint
        max_voxels=(90000, 120000)),
    # mean VFE used in SECOND
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    pts_middle_encoder=dict(
        type='SparseLiDAREnc8x',
        # input channels equals to num_features of VFE
        input_channel=4,
        base_channel=16,
        # equals to the fused image feature dim at the last level
        out_channel=32,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        sparse_shape_xyz=[1600, 1600, 80],  # modified by 200 * 8 = 1600
        ),
    occ_fuser=dict(
        type='VisFuser',
        # single scale
        embed_dims=[32],
    ),
    # TODO: encoder backbone and fpn
    # after the feature fusion?
    # occ_encoder_backbone=dict(
    #     type='CustomResNet3D',
    #     depth=18,
    #     n_input_channels=numC_Trans,
    #     block_inplanes=voxel_channels,
    #     out_indices=voxel_out_indices,
    #     norm_cfg=dict(type='SyncBN', requires_grad=True),
    # ),
    # occ_encoder_neck=dict(
    #     type='FPN3D',
    #     with_cp=True,
    #     in_channels=voxel_channels,
    #     out_channels=voxel_out_channel,
    #     norm_cfg=dict(type='SyncBN', requires_grad=True),
    # ),
    pts_bbox_head=dict(
        type='OccHead',
        volume_h=volume_h_,
        volume_w=volume_w_,
        volume_z=volume_z_,
        occ_size=occ_size,
        num_query=900,
        num_classes=18,
        conv_input=[_dim_[2], 256, _dim_[1], 128, _dim_[0], 64, 64],
        conv_output=[256, _dim_[1], 128, _dim_[0], 64, 64, 32],
        out_indices=[0, 2, 4, 6],
        upsample_strides=[1, 2, 1, 2, 1, 2, 1],
        embed_dims=_dim_,
        single_scale_fusion=True,
        img_channels=[512, 512, 512],
        use_semantic=use_semantic,
        use_mask=use_mask,
        use_points=use_points,
        len_queue=len_queue,
        transformer_template=dict(
            type='PerceptionTransformer',
            embed_dims=_dim_,
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
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    embed_dims=_dim_,
                    conv_num=2,
                    operation_order=('cross_attn', 'norm',
                                     'ffn', 'norm', 'conv')))),
    ce_loss_cfg=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        class_weight=class_weight,
        loss_weight=1.0),
    geo_loss=True,
    sem_loss=True,
),
)

dataset_type = 'CustomNuScenesOccDataset'
data_root = 'data/occ3d-nus/'
file_client_args = dict(backend='disk')
occ_gt_data_root='data/occ3d-nus'

train_pipeline = [
    dict(type='LoadOccPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadOccupancy', data_root=occ_gt_data_root, use_semantic=use_semantic),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D', keys=['img', 'points', 'voxel_semantics'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccupancy', data_root=occ_gt_data_root, use_semantic=use_semantic),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D', keys=['img'])
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
            box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
            data_root=data_root,
            ann_file='data/occ3d-nus/occ_infos_temporal_val.pkl',
            pipeline=test_pipeline,  
            occ_size=occ_size,
            pc_range=point_cloud_range,
            use_semantic=use_semantic,
            use_sequential=use_sequential,
            classes=class_names,
            len_queue=len_queue,
            modality=input_modality,
            eval_fscore=True),
    test=dict(type=dataset_type,
            data_root=data_root,
            ann_file='data/occ3d-nus/occ_infos_temporal_val.pkl',
            pipeline=test_pipeline, 
            occ_size=occ_size,
            pc_range=point_cloud_range,
            use_semantic=use_semantic,
            use_sequential=use_sequential,
            classes=class_names,
            len_queue=len_queue,
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
