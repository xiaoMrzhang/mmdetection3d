voxel_size = [0.16, 0.16, 4]
model = dict(
    type='VoxelNetPillar',
    voxel_layer=dict(
        max_num_points=32,
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
        voxel_size=voxel_size,
        max_voxels=(16000, 40000)),
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1]),
    middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[496, 432]),
    backbone=dict(
        type='SECOND_RAN',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    neck=dict(
        type='SECONDFPN_RAN',
        in_channels=[64, 128, 256],
        # upsample_strides=[1, 2, 4],
        upsample_strides=[0.5, 1, 2],
        out_channels=[128, 128, 128],
        use_conv_for_no_stride=True),
    bbox_head=dict(
        type='CenterHead',
        in_channels=sum([128, 128, 128]),
        tasks=[
            dict(num_class=1, class_names=['Pedestrian']),
            dict(num_class=1, class_names=['Cyclist']),
            dict(num_class=1, class_names=['Car']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)), #, vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            # post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            post_center_range=[0, -39.68, -3, 69.12, 39.68, 1],
            max_num=500,
            score_threshold=0.1,
            # make sure the out_size_factor is 2 or 4
            out_size_factor=4,
            pc_range=[0, -39.68],
            voxel_size=voxel_size[:2],
            # code size 7 or 9
            code_size=7,),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True))
# model training and testing settings
train_cfg = dict(
    assigner=[
        dict(  # for Pedestrian
            type='MaxIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.5,
            neg_iou_thr=0.35,
            min_pos_iou=0.35,
            ignore_iof_thr=-1),
        dict(  # for Cyclist
            type='MaxIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.5,
            neg_iou_thr=0.35,
            min_pos_iou=0.35,
            ignore_iof_thr=-1),
        dict(  # for Car
            type='MaxIoUAssigner',
            iou_calculator=dict(type='BboxOverlapsNearest3D'),
            pos_iou_thr=0.6,
            neg_iou_thr=0.45,
            min_pos_iou=0.45,
            ignore_iof_thr=-1),
    ],
    allowed_border=0,
    pos_weight=-1,
    debug=False,
    grid_size=[432, 496, 1],
    voxel_size=voxel_size,
    out_size_factor=4,
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    # for each channel, 8 for kitti 10 for nuscence
    code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) # 0.2, 0.2])
test_cfg = dict(
    # use_rotate_nms=True,
    # nms_across_levels=False,
    # nms_thr=0.01,
    # score_thr=0.1,
    min_bbox_size=0,
    nms_pre=100,
    max_num=50,
    post_center_limit_range=[0, -39.68, -3, 69.12, 39.68, 1],
    max_per_img=500,
    max_pool_nms=False,
    min_radius=[4, 12, 10, 1, 0.85, 0.175],
    score_threshold=0.1,
    pc_range=[0, -39.68],
    out_size_factor=4,
    voxel_size=voxel_size[:2],
    nms_type='rotate',
    pre_max_size=1000,
    post_max_size=83,
    nms_thr=0.2,)
