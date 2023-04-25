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


def draw(
    voxels,                         # semantic occupancy predictions
    resolution=[0.5, 0.5, 0.5],
    voxel_origin=[-50, -50, -1],
    ratio=1,                        # scale
):
    coords = voxels[:, :3] // ratio
    # Compute the voxels coordinates
    grid_coords = np.zeros([200 // ratio, 200 // ratio, 16 // ratio])
    grid_coords[coords[:, 0], coords[:, 1], coords[:, 2]] = voxels[:, 3]

    # Obtaining the coords_grid
    g_xx = np.arange(0, 200 // ratio) # [0, 1, ..., 199]
    g_yy = np.arange(0, 200 // ratio) # [0, 1, ..., 199]
    g_zz = np.arange(0, 16 // ratio)  # [0, 1, ..., 15]

    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])
    coords_grid = (coords_grid * resolution) + resolution / 2
    coords_grid += np.array(voxel_origin, dtype=np.float32).reshape([1, 3])

    # grid_coords: (H*W*Z, 4) with the real xyz and predicted label(0~17)
    grid_coords = np.vstack([coords_grid.T, grid_coords.reshape(-1)]).T

    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    fov_voxels = grid_coords[grid_coords[:, 3] > 0]
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 1],
        fov_voxels[:, 0],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        colormap="viridis",
        scale_factor=0.95 * 0.5,
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
    occ_path = './data/n015-2018-07-18-11-07-57+0800__LIDAR_TOP__1531883531950107.pcd.bin.npy'
    occ = np.load(open(occ_path, "rb"))
    ratio = 1
    draw(occ, ratio=ratio)
