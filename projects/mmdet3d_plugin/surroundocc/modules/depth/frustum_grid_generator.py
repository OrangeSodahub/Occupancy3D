import torch
import torch.nn as nn
import kornia

from .utils import transform_utils, grid_utils, depth_utils


class FrustumGridGenerator(nn.Module):
    def __init__(self, grid_size, pc_range, disc_cfg):
        """
        Initializes Grid Generator for frustum features
        Args:
            grid_size [np.array(3)]: Voxel grid shape [X, Y, Z]
            pc_range [list]: Voxelization point cloud range [X_min, Y_min, Z_min, X_max, Y_max, Z_max]
            disc_cfg [int]: Depth discretiziation configuration
        """
        super().__init__()
        self.dtype = torch.float32
        self.grid_size = torch.as_tensor(grid_size)
        self.pc_range = pc_range
        self.out_of_bounds_val = -2
        self.disc_cfg = disc_cfg

        # Calculate voxel size
        pc_range = torch.as_tensor(pc_range).reshape(2, 3)
        self.pc_min = pc_range[0]
        self.pc_max = pc_range[1]
        self.voxel_size = (self.pc_max - self.pc_min) / self.grid_size # (0.4, 0.4, 0.4)

        # Create voxel grid
        self.depth, self.width, self.height = self.grid_size.int()
        xs = torch.linspace(0, self.depth - 1, self.depth)
        ys = torch.linspace(0, self.width - 1, self.width)
        zs = torch.linspace(0, self.height - 1, self.height)
        base_grid = torch.stack(torch.meshgrid([xs, ys, zs]), dim=-1) # (D, W, H, 3)->(200, 200, 16, 3)
        self.voxel_grid = base_grid.unsqueeze(0)  # (1, D, W, H, 3) -> XYZ
        # Add offsets to center of voxel
        self.voxel_grid += 0.5
        self.grid_to_ego = self.grid_to_ego_unproject(
            pc_min=self.pc_min, voxel_size=self.voxel_size
        )

    def grid_to_ego_unproject(self, pc_min, voxel_size):
        """
        Calculate grid to LiDAR unprojection for each plane
        Args:
            pc_min [torch.Tensor(3)]: Minimum of point cloud range [X, Y, Z] (m)
            voxel_size [torch.Tensor(3)]: Size of each voxel [X, Y, Z] (m)
        Returns:
            unproject [torch.Tensor(4, 4)]: Voxel grid to LiDAR unprojection matrix
        """
        x_size, y_size, z_size = voxel_size
        x_min, y_min, z_min = pc_min
        unproject = torch.tensor(
            [
                [x_size, 0, 0, x_min],
                [0, y_size, 0, y_min],
                [0, 0, z_size, z_min],
                [0, 0, 0, 1],
            ],
            dtype=self.dtype,
        )  # (4, 4)

        return unproject

    def transform_grid(
        self, voxel_grid, grid_to_ego, ego_to_lidar, lidar_to_cam, cam_to_img
    ):
        """
        Transforms voxel sampling grid into frustum sampling grid
        Args:
            grid [torch.Tensor(B, X, Y, Z, 3)]: Voxel sampling grid
            grid_to_ego [torch.Tensor(4, 4)]: Voxel grid to ego unprojection matrix
            ego_to_lidar [torch.Tensor(4, 4)]: Ego to LiDAR frame transformation
            lidar_to_cam [torch.Tensor(B, 4, 4)]: LiDAR to camera frame transformation
            cam_to_img [torch.Tensor(B, 3, 4)]: Camera projection matrix
        Returns:
            frustum_grid [torch.Tensor(B, X, Y, Z, 3)]: Frustum sampling grid
        """
        B = cam_to_img.shape[0]
        lidar_to_cam = lidar_to_cam.unsqueeze(0).repeat(B, 1, 1)

        # Create transformation matricies
        G_E = grid_to_ego.float()  # Voxel Grid -> Ego (4, 4)
        E_L = ego_to_lidar.float() # Ego -> LiDAR (4, 4)
        L_C = lidar_to_cam.float()  # LiDAR -> Camera (B, 4, 4)
        C_I = cam_to_img.float()  # Camera -> Image (B, 3, 4)
        trans = L_C @ (E_L @ G_E) # (B, 4, 4)

        # Reshape to match dimensions
        # voxel_grid: (B, D, W, H)
        trans = trans.reshape(B, 1, 1, 4, 4)
        voxel_grid = voxel_grid.repeat_interleave(repeats=B, dim=0)

        # Transform to camera frame
        # camera_grid: (B, D, W, H)
        camera_grid = kornia.geometry.transform_points(trans_01=trans, points_1=voxel_grid)

        # Project to image
        # image_grid: shape (1, D, W, H, 2) -> (1, 100, 100, 8, 2)
        # image_depths: shape (1, D, W, H) -> (1, 100, 100, 8)
        C_I = C_I.reshape(B, 1, 1, 3, 4)
        image_grid, image_depths = transform_utils.project_to_image(
            project=C_I, points=camera_grid
        )

        # Convert depths to depth bins
        image_depths = depth_utils.bin_depths(depth_map=image_depths, **self.disc_cfg)

        # Stack to form frustum grid
        # frustum_grid: shape (bs, D, W, H, 3) -> (1, 100, 100, 8, 3)
        image_depths = image_depths.unsqueeze(-1)
        frustum_grid = torch.cat((image_grid, image_depths), dim=-1)

        return frustum_grid

    def forward(self, ego_to_lidar, lidar_to_cam, cam_to_img, image_shape):
        """
        Generates sampling grid for frustum features
        Args:
            ego_to_lidar [torch.Tensor(4, 4)]: Ego to LiDAR frame transformation
            lidar_to_cam [torch.Tensor(4, 4)]: LiDAR to camera frame transformation
            cam_to_img [torch.Tensor(B, 3, 4)]: Camera projection matrix
            image_shape [torch.Tensor(B, 2)]: Image shape [H, W]
        Returns:
            frustum_grid [torch.Tensor(B, X, Y, Z, 3)]: Sampling grids for frustum features
        """

        # frustum grid: (B, D, W, H, 3)
        frustum_grid = self.transform_grid(
            voxel_grid=self.voxel_grid.to(cam_to_img.device),
            grid_to_ego=self.grid_to_ego.to(cam_to_img.device),
            ego_to_lidar=torch.from_numpy(ego_to_lidar).to(cam_to_img.device),
            lidar_to_cam=torch.from_numpy(lidar_to_cam).to(cam_to_img.device),
            cam_to_img=cam_to_img,
        )

        # TODO: fix
        # Normalize grid on X Y Z
        image_shape, _ = torch.max(image_shape, dim=0)
        image_depth = torch.tensor(
            [self.disc_cfg["num_bins"]],
            device=image_shape.device,
            dtype=image_shape.dtype,
        )
        # frustum_shape: (d, v, u)->(112, 900, 1600) (3 of [B, D, W, H, 3])
        # frustum_grid: (B, D, H, W, 3) with coords between (-1, 1)
        frustum_shape = torch.cat((image_depth, image_shape))
        frustum_grid = grid_utils.normalize_coords(
            coords=frustum_grid, shape=frustum_shape
        )

        # Replace any NaNs or infinites with out of bounds
        mask = ~torch.isfinite(frustum_grid)
        frustum_grid[mask] = self.out_of_bounds_val

        return frustum_grid
