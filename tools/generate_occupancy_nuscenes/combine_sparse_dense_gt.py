# This script is used to combine the sparse and dense ground truth
# from the challenge dataset and the OpenOccupancy dataset respectively.

import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from collections import Counter
try:
    from mayavi import mlab
except:
    print("Mayavi is not installed. Please install it if you want to visualize the results.")


colors = np.array(
    [
            [0, 0, 0, 255],
            [255, 120, 50, 255],    # barrier              orangey
            [255, 192, 203, 255],   # bicycle              pink
            [255, 255, 0, 255],     # bus                  yellow
            [0, 150, 245, 255],     # car                  blue
            [0, 255, 255, 255],     # construction_vehicle cyan
            [200, 180, 0, 255],     # motorcycle           dark orange
            [255, 0, 0, 255],       # pedestrian           red
            [255, 240, 150, 255],   # traffic_cone         light yellow
            [135, 60, 0, 255],      # trailer              brown
            [160, 32, 240, 255],    # truck                purple
            [255, 0, 255, 255],     # driveable_surface    dark pink
            [139, 137, 137, 255],   # other_flat           dark red
            [75, 0, 75, 255],       # sidewalk             dard purple
            [150, 240, 80, 255],    # terrain              light green
            [230, 230, 250, 255],   # manmade              white
            [0, 175, 0, 255],       # vegetation           green
            [255, 255, 255, 255]    # free                 
    ]
).astype(np.uint8)


def get_grid_coords(dims, resolution, ratio=1):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [200, 200, 16])
    :param resolution: the voxel size of [0.4, 0.4, 0.4]
    :return coords_grid: is the center coords of voxels in the grid
    """

    # Obtaining the index_grid
    g_xx = np.arange(0, dims[0]) # [0, 1, ..., 199]
    g_yy = np.arange(0, dims[1]) # [0, 1, ..., 199]
    g_zz = np.arange(0, dims[2]) # [0, 1, ..., 15]

    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    index_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    index_grid = index_grid.astype(np.int32)

    # Obtaining the coords_grid
    g_xx = np.arange(0, dims[0] // ratio) # [0, 1, ..., 199]
    g_yy = np.arange(0, dims[1] // ratio) # [0, 1, ..., 199]
    g_zz = np.arange(0, dims[2] // ratio) # [0, 1, ..., 15]

    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])
    coords_grid = (coords_grid * resolution) + resolution / 2

    index_semantics = np.zeros([dims[0] // ratio, dims[1] // ratio, dims[2] // ratio])

    return index_grid, index_semantics, coords_grid


def combine(dense_occ, sparse_occ):
    """
    :param dense_occ: shape (N, 4)
    :param sparse_occ: shape (200, 200, 16)

    return:
        dense_occ_converted: shape (200, 200, 16)
    """
    # convert the dense occ format
    index_grid = dense_occ[:, :3].astype(np.int32)
    dense_occ_converted = np.zeros_like(sparse_occ) + 17
    dense_occ_converted[index_grid[:, 0], index_grid[:, 1], index_grid[:, 2]] = dense_occ[:, 3]

    # combine the dense and sparse occ
    dense_occ_converted[sparse_occ != 17] = sparse_occ[sparse_occ != 17]

    return dense_occ_converted


def draw(
        dense_occ,
        sparse_occ=None,
        resolution=[0.4, 0.4, 0.4],
        version=None):
    """
    :param dense_occ: the dense ground truth
    :param sparse_occ: the sparse ground truth
    :param resolution: the voxel size of [0.4, 0.4, 0.4]
    :param version: the version of the dense occ
                    `SurrOcc` or `OpenOcc`
    """
    if version == 'OpenOcc':
        # Obtaining the coords_grid
        h, w, z = sparse_occ.shape
        grid_index, semantics_index, coords_grid = get_grid_coords([h, w, z], resolution)
        
        # dense occ
        dense_coords = dense_occ[:, :3]
        if version == 'OpenOcc':
            dense_grid_coords = np.zeros([512, 512, 40]) + 17
            dense_grid_coords[dense_coords[:, 2], dense_coords[:, 1], dense_coords[:, 0]] = dense_occ[:, 3]
            dense_grid_coords = dense_grid_coords[56:456, 56:456, 8:]
            dense_grid_coords = downsample(dense_grid_coords)
        dense_grid_coords = np.vstack([coords_grid.T, dense_grid_coords.reshape(-1)]).T
        # sparse occ
        semantics_index[grid_index[:, 0], grid_index[:, 1], grid_index[:, 2]] = \
            sparse_occ[grid_index[:, 0], grid_index[:, 1], grid_index[:, 2]]
        sparse_grid_coords = np.vstack([coords_grid.T, semantics_index.reshape(-1)]).T

        # compare
        # TODO: align the coordinate system
        new_grid_coords = sparse_grid_coords.copy()
        new_grid_coords[
            (sparse_grid_coords[:, 3] == dense_grid_coords[:, 3]) &
            (sparse_grid_coords[:, 3] != 17) &
            (dense_grid_coords[:, 3] != 17), 3] = 17
        new_grid_coords[(sparse_grid_coords[:, 3] != dense_grid_coords[:, 3]) & (sparse_grid_coords[:, 3] != 17), 3] \
                = sparse_grid_coords[(sparse_grid_coords[:, 3] != dense_grid_coords[:, 3]) & (sparse_grid_coords[:, 3] != 17), 3]
        new_grid_coords[(sparse_grid_coords[:, 3] != dense_grid_coords[:, 3]) & (dense_grid_coords[:, 3] != 17), 3] \
                = dense_grid_coords[(sparse_grid_coords[:, 3] != dense_grid_coords[:, 3]) & (dense_grid_coords[:, 3] != 17), 3]

    elif version == 'SurrOcc':
        h, w, z = dense_occ.shape
        grid_index, semantics_index, coords_grid = get_grid_coords([h, w, z], resolution)

        # dense
        semantics_index[grid_index[:, 0], grid_index[:, 1], grid_index[:, 2]] = \
            dense_occ[grid_index[:, 0], grid_index[:, 1], grid_index[:, 2]]
        dense_grid_coords = np.vstack([coords_grid.T, semantics_index.reshape(-1)]).T
        # sparse
        semantics_index[grid_index[:, 0], grid_index[:, 1], grid_index[:, 2]] = \
            sparse_occ[grid_index[:, 0], grid_index[:, 1], grid_index[:, 2]]
        sparse_grid_coords = np.vstack([coords_grid.T, semantics_index.reshape(-1)]).T

        # compare
        new_grid_coords = sparse_grid_coords.copy()
        # if sparse occ is not None, then compare the dense and sparse occ
        # the same part will be labeled as the white color.
        # if sparse occ is None, then the dense occ will be showed.
        if sparse_occ is not None:
            mask = dense_grid_coords[:, 3] != 17
            new_grid_coords[
                (sparse_grid_coords[:, 3] == dense_grid_coords[:, 3]) &
                (sparse_grid_coords[:, 3] != 17) &
                (dense_grid_coords[:, 3] != 17), 3] = 17
            new_grid_coords[(sparse_grid_coords[:, 3] != dense_grid_coords[:, 3]) & (sparse_grid_coords[:, 3] != 17), 3] \
                    = sparse_grid_coords[(sparse_grid_coords[:, 3] != dense_grid_coords[:, 3]) & (sparse_grid_coords[:, 3] != 17), 3]
            new_grid_coords[(sparse_grid_coords[:, 3] != dense_grid_coords[:, 3]) & (dense_grid_coords[:, 3] != 17), 3] \
                    = dense_grid_coords[(sparse_grid_coords[:, 3] != dense_grid_coords[:, 3]) & (dense_grid_coords[:, 3] != 17), 3]        

    # draw
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    if version == 'OpenOcc':
        mask = new_grid_coords[:, 3] < 17
    fov_voxels = new_grid_coords[(new_grid_coords[:, 3] > 0) & mask]
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 1],
        fov_voxels[:, 0],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        colormap="viridis",
        scale_factor=0.95 * 0.4,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=17,
    )

    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors

    scene = figure.scene
    scene.camera.position = [  0.75131739, -35.08337438,  16.71378558]
    scene.camera.focal_point = [  0.75131739, -34.21734897,  16.21378558]
    scene.camera.view_angle = 40.0
    scene.camera.view_up = [0.0, 0.0, 1.0]
    scene.camera.clipping_range = [0.01, 300.]
    scene.camera.compute_view_plane_normal()
    scene.render()

    mlab.show()


def downsample(voxels):
    """
    Downsample the voxel grid by 2x2x2, only for the OpenOcc.
    """
    def sample(voxel):
        voxel = voxel.flatten()
        voxel = voxel[voxel != 0]
        if len(voxel) == 0:
            return 0
        c = Counter(voxel)
        return c.most_common(1)[0][0]
    
    new_voxels = np.zeros([200, 200, 16])
    for x in range(0, 400, 2):
        for y in  range(0, 400, 2):
            for z in range(0, 32, 2):
                new_voxels[x // 2, y // 2, z // 2] = sample(voxels[x:x+2, y:y+2, z:z+2])

    return new_voxels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='SurrOcc', help='OpenOcc or SurrOcc')
    parser.add_argument('--dense_occ_root', type=str, default='../../data/dense_occ/', help='dense occupancy path')
    parser.add_argument('--sparse_occ_root', type=str, default='../../data/sparse_occ/', help='sparse occupancy path')
    parser.add_argument('--save_root', type=str, default='../../data/new_occ/', help='save path')
    parser.add_argument('--train_pkl', type=str, default='../../data/occ3d_nus/occ_infos_temporal_train.pkl', help='train pkl')
    parser.add_argument('--val_pkl', type=str, default='../../data/occ3d_nus/occ_infos_temporal_val.pkl', help='val pkl')
    parser.add_argument('--draw', action='store_true', help='draw the occupancy')
    args = parser.parse_args()
    
    if args.version == 'OpenOcc':
        # NOTE: not implemented yet
        raise NotImplementedError
        dense_occ_path = './69b793ec8dc44e2fbd33d8cdd16b5a31.npy'

    elif args.version == 'SurrOcc':
        for pkl in [args.train_pkl, args.val_pkl]:
            print(f'====> Load {pkl}')
            infos = pickle.load(open(pkl, 'rb'))['infos']
            dense_occ_names = os.listdir(args.dense_occ_root)
            timestamps = [int(dense_occ_name.split('__')[-1].split('.')[0]) for dense_occ_name in dense_occ_names]
            dense_occ_map = dict(zip(timestamps, dense_occ_names))
            for info in tqdm(infos):
                timestamp = int(info['timestamp'])
                dense_occ_name = dense_occ_map[timestamp]
                dense_occ_path = os.path.join(args.dense_occ_root, dense_occ_name)
                sparse_occ_path = os.path.join(args.sparse_occ_root, info['occ_gt_path'])
                dense_occ = np.load(open(dense_occ_path, "rb"))
                sparse_file = np.load(open(sparse_occ_path, "rb"))
                sparse_occ = sparse_file['semantics']
                mask_lidar, mask_camera = sparse_file['mask_lidar'], sparse_file['mask_camera']
                dense_occ_converted = combine(dense_occ, sparse_occ)
                save_path = os.path.join(args.save_root, info['occ_gt_path'])
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.savez(open(dense_occ_path, 'wb'), semantics=dense_occ_converted, mask_lidar=mask_lidar, mask_camera=mask_camera)
                if args.draw:
                    draw(dense_occ_converted, sparse_occ, version='SurrOcc')
        print('====> Done.')


        # dense_occ_path = './data/n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883530949817.pcd.bin.npy'
        # sparse_occ_path = './data/labels.npz'
        # dense_occ = np.load(open(dense_occ_path, "rb"))
        # sparse_occ = np.load(open(sparse_occ_path, "rb"))['semantics']
        # dense_occ_converted = combine(dense_occ, sparse_occ)
        # if args.draw:
        #     draw(dense_occ_converted, sparse_occ, version='SurrOcc')
