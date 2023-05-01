import os
import argparse
import numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet.datasets import replace_ImageToTensor
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model

import sys
sys.path.append('./')
import projects

def parse_args():
    parser = argparse.ArgumentParser(
        description='Test SurroundOcc')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument(
        '--results_path',
        type=str,
        default='./save_results',
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    print(args)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # read results
    dataset = build_dataset(cfg.data.test)
    file_list = os.listdir(args.results_path)
    file_list.sort(key=lambda x:int(x[0:4]))
    outputs = [np.load(open(os.path.join(args.results_path, file), 'rb'))['pred'] for file in file_list]

    print(f"Loaded {len(outputs)} results.")

    dataset.evaluate(outputs)


if __name__ == '__main__':
    main()

# python tools/test_local.py --config ./projects/configs/surroundocc/surroundocc.py --eval bbox --results_path ./save_results

# Starting Evaluation...
# 6019it [07:18, 13.72it/s]
# ===> per class IoU of 6018 samples:
# ===> others - IoU = 6.67
# ===> barrier - IoU = 38.79
# ===> bicycle - IoU = 21.47
# ===> bus - IoU = 42.37
# ===> car - IoU = 46.19
# ===> construction_vehicle - IoU = 19.05
# ===> motorcycle - IoU = 25.94
# ===> pedestrian - IoU = 25.59
# ===> traffic_cone - IoU = 21.56
# ===> trailer - IoU = 22.58
# ===> truck - IoU = 33.29
# ===> driveable_surface - IoU = 60.96
# ===> other_flat - IoU = 32.11
# ===> sidewalk - IoU = 37.96
# ===> terrain - IoU = 33.62
# ===> manmade - IoU = 21.02
# ===> vegetation - IoU = 22.13
# ===> mIoU of 6018 samples: 30.08

# ######## F score: 0.6543626205952127 #######