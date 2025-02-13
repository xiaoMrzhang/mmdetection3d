_base_ = [
    '../_base_/models/hv_pointpillars_pillar_supervise_second_waymo.py',
    '../_base_/datasets/waymoD5-3d-3class.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
]

# data settings
data = dict(train=dict(dataset=dict(load_interval=5)),
            test=dict(load_interval=1),
            samples_per_gpu=4)