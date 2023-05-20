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
use_mask = True

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

class_names =  ['other', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
                'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface',
                'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation', 'free']

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (768, 1408),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Model
grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 0.4],
    'depth': [1.0, 45.0, 0.5],
}

# TODO: fix flip
bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.,
    flip_dy_ratio=0.)

_dim_ = [64, 128, 256]
_ffn_dim_ = [128, 256, 512]
volume_h_ = [100, 50, 25]
volume_w_ = [100, 50, 25]
volume_z_ = [8, 4, 2]
_num_points_ = [2, 4, 8]
_num_layers_ = [1, 3, 6]
numC_Trans = 32
multi_adj_frame_id_cfg = (1, 1+1, 1)

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

model = dict(
    type='SurroundOcc',
    use_grid_mask=True,
    use_semantic=use_semantic,
    img_backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        patch_size=4,
        window_size=12,
        mlp_ratio=4,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        strides=(4, 2, 2, 2),
        out_indices=(1, 2, 3, 4),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        use_abs_pos_embed=False,
        return_stereo_feat=True,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        pretrain_style='official',
        output_missing_index_as_none=False),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=3,
        relu_before_extra_convs=True),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVStereo',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        # equals to `out_channels` in `img_neck
        in_channels=256,
        out_channels=numC_Trans,
        sid=False,
        collapse_z=False,
        loss_depth_weight=0.2,
        depthnet_cfg=dict(use_dcn=False,
                          aspp_mid_channels=96,
                          stereo=True,
                          bias=5.),
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet3D',
        numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg))+1),
        num_layer=[1, 2, 4],
        with_cp=False,
        num_channels=[numC_Trans,numC_Trans*2,numC_Trans*4],
        stride=[1,2,2],
        backbone_output_ids=[0,1,2]),
    img_bev_encoder_neck=dict(type='LSSFPN3D',
                              in_channels=numC_Trans*7,
                              out_channels=numC_Trans),
    pts_bbox_head=dict(
        type='OccHead',
        volume_h=volume_h_,
        volume_w=volume_w_,
        volume_z=volume_z_,
        occ_size=occ_size,
        num_query=900,
        num_classes=18,
        conv_input=[_dim_[2], 128, _dim_[1], 64, _dim_[0], 32, 32],
        conv_output=[128, _dim_[1], 64, _dim_[0], 32, 32, 16],
        out_indices=[0, 2, 4, 6],
        upsample_strides=[1, 2, 1, 2, 1, 2, 1],
        embed_dims=_dim_,
        img_channels=[256, 256, 256],
        use_semantic=use_semantic,
        use_mask=use_mask,
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
    occ_fuser=dict(
        type='OccFuser',
        img_dims=[numC_Trans],
    ),
    ce_loss_cfg=dict(
        type='CrossEntropyLoss',
        use_sigmoid=False,
        loss_weight=1.0),
    geo_loss=True,
    sem_loss=True,
    ),
)

dataset_type = 'CustomNuScenesOccDataset'
data_root = 'data/occ3d-nus/'
file_client_args = dict(backend='disk')
occ_gt_data_root='data/occ3d-nus'
depth_gt_data_root='data/depth_gt'

train_pipeline = [
    dict(type='PrepareImageInputs', is_train=True, data_config=data_config, sequential=True),
    dict(type='LoadOccupancy', data_root=occ_gt_data_root, use_semantic=use_semantic, bda_aug_conf=bda_aug_conf, is_train=True),
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5, file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D', keys=['img_inputs', 'voxel_semantics', 'mask_camera', 'depth_gt'])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=True),
    # TODO: optimze this
    dict(type='LoadOccupancy', data_root=occ_gt_data_root, use_semantic=use_semantic, bda_aug_conf=bda_aug_conf, is_train=False),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D', keys=['img_inputs'])
]

share_data_config = dict(
    stereo=True,
    filter_empty_gt=False,
    img_info_prototype='bevdet4d',
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
)

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

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

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
