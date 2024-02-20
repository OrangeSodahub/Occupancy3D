import sys
sys.path.append('./')
from proj.datasets import BlenderObject
from proj.modules.nerf import Wizard, config_parser


def main():
    parser = config_parser()
    args = parser.parse_args()

    # create data
    blender_data = BlenderObject(root_dir=args.datadir, half_res=args.half_res,
                                testskip=args.testskip, white_bkgd=args.white_bkgd)
    print(f"Loaded blender data:, images shape: {blender_data.imgs.shape}, " \
        f"c2w mat shape: {blender_data.render_poses.shape}, hwf: {blender_data.hwf}")

    # create wizard (create model and load checkpoint if possible)
    wizard = Wizard(args)
    # train
    wizard.train(blender_data)
    print("Training done.")


if __name__ == '__main__':
    main()