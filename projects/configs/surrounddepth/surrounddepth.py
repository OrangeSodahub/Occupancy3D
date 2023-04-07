_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

dataset_type = 'NuscDepthDataset'
raw_data_root = 'data/raw-nus/'
occ_data_root = 'data/occ3d-nus/'

model = dict(
    type='SurroundDepth',
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
    pts_bbox_head=dict(
        type='DepthHead',
        shape=[384, 640],
        skip=False,
        num_ch_enc=[64, 64, 128, 256, 512],
        frame_ids=[0, -1, 1],
        use_fix_mask=True,
        v1_multiscale=True,
        disable_automasking=True,
        avg_reprojection=True,
        predictive_mask=True,
        use_sfm_spatial=True,
        spatial=False,
        scales=[0, 1, 2, 3],
        disparity_smoothness=1e-3,
        match_spatial_weight=0.1,
        spatial_weight=0.1,
        no_ssim=False,
    ),
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        raw_data_root=raw_data_root,
        occ_data_root=occ_data_root,
        ann_file='data/occ3d-nus/occ_infos_temporal_train.pkl',
        height=384,
        width=640,
        frame_ids=[0, -1, 1],
        num_scales=4,
        is_train=True),
    val=dict(
        type=dataset_type,
        raw_data_root=raw_data_root,
        occ_data_root=occ_data_root,
        ann_file='data/occ3d-nus/occ_infos_temporal_val.pkl',
        height=384,
        width=640,
        frame_ids=[0, -1, 1],
        num_scales=4,
        is_train=False),
    test=dict(
        type=dataset_type,
        raw_data_root=raw_data_root,
        occ_data_root=occ_data_root,
        ann_file='data/occ3d-nus/occ_infos_temporal_val.pkl',
        height=384,
        width=640,
        frame_ids=[0, -1, 1],
        num_scales=4,
        is_train=False),
)