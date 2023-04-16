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
volume_size = [100, 100, 8]
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

model = dict(
    type='SurroundOcc',
    stage='stage1',
    use_grid_mask=True,
    use_semantic=use_semantic,
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
    pts_backbone=dict(
        type='BaseDepthNet',
        occ_size=occ_size,       # (200, 200, 16)
        volume_size=volume_size, # (100, 100, 8)
        pc_range=point_cloud_range,
        x_bound=[-40, 40, 0.4],
        y_bound=[-40, 40, 0.4],
        z_bound=[-1, 5.4, 0.4],
        # TODO: verify d_bound
        d_bound=[2, 58, 0.5],
        output_channels=80,
        depth_net_conf=dict(
            in_channels=512,
            mid_channels=512),
        agg_voxel_mode='mean',
        ssc_net_conf=dict(
            class_num=2,
            input_dimensions=volume_size, # (100, 100, 8)
            out_scale="1_2")),
)

dataset_type = 'CustomNuScenesOccDataset'
data_root = 'data/occ3d-nus/'
file_client_args = dict(backend='disk')
depth_gt_data_root='data/depth_gt'

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    # TODO: fix depth gt
    # dict(type='LoadDepthGT', data_root=depth_gt_data_root),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D', keys=['img', 'voxel_semantics', 'mask_lidar', 'mask_camera', 'depth_gt'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
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
