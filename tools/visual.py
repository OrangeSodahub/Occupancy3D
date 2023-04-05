import numpy as np
from mayavi import mlab
# mlab.options.offscreen = True
print("Set mlab.options.offscreen={}".format(mlab.options.offscreen))


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


def get_grid_coords(dims, resolution, ratio):
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


def draw(
    voxels,                         # semantic occupancy predictions
    mask_lidar,
    mask_camera,
    voxel_origin=[-40, -40, -1],      # the original of the whole space
    voxel_size=[0.4, 0.4, 0.4],     # voxel size in the real world
    ratio=1,                        # scale
    remove_free=False,
):
    h, w, z = voxels.shape

    # Compute the voxels coordinates
    grid_index, semantics_index, grid_coords = get_grid_coords([h, w, z], voxel_size, ratio)

    if remove_free:
        voxels = np.vstack([grid_index.T, voxels.reshape(-1)]).T
        voxels = voxels[voxels[:, 3] < 17]
        grid_index = voxels[:, :3] // ratio
        semantics_index = np.zeros([h // ratio, w // ratio, z // ratio])
        semantics_index[grid_index[:, 0], grid_index[:, 1], grid_index[:, 2]] = voxels[:, 3]
        grid_coords[:, :3] += np.array(voxel_origin, dtype=np.float32).reshape([1, 3])
        grid_coords = np.vstack([grid_coords.T, semantics_index.reshape(-1)]).T
    else:
        semantics_index[grid_index[:, 0], grid_index[:, 1], grid_index[:, 2]] = \
            voxels[grid_index[:, 0]*ratio, grid_index[:, 1]*ratio, grid_index[:, 2]*ratio]
        grid_coords += np.array(voxel_origin, dtype=np.float32).reshape([1, 3])
        # grid_coords: (H*W*Z, 4) with the real xyz and predicted label(0~17)
        grid_coords = np.vstack([grid_coords.T, semantics_index.reshape(-1)]).T

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords

    # Remove empty and unknown voxels (label==0)
    fov_voxels = fov_grid_coords
    # TODO: mask in ratio != 1
    if False:
        fov_voxels = fov_grid_coords[
            (mask_camera.reshape(-1) == 1) & (mask_lidar.reshape(-1) == 1), :
        ]
    fov_voxels = fov_voxels[
        (fov_voxels[:, 3] > 0) & (fov_voxels[:, 3] < 17)
    ]

    print(len(fov_voxels))

    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    # Draw occupied inside FOV voxels
    voxel_size = sum(voxel_size) / 3
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 1],
        fov_voxels[:, 0],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        colormap="viridis",
        scale_factor=0.95 * voxel_size,
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


if __name__ == '__main__':
    occ_path = './data/labels.npz'
    occ = np.load(open(occ_path, "rb"))
    ratio = 1
    draw(occ['semantics'], occ['mask_lidar'], occ['mask_camera'], ratio=ratio, remove_free=True)
