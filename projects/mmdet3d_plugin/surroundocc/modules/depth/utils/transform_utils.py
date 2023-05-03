import torch
import kornia


def project_to_image(project, points):
    """
    Project points to image
    Args:
        project [torch.tensor(..., 3, 4)]: Projection matrix
        points [torch.Tensor(..., 3)]: 3D points
    Returns:
        points_img [torch.Tensor(..., 2)]: Points in image
        points_depth [torch.Tensor(...)]: Depth of each point
    """
    # Reshape tensors to expected shape
    points = kornia.geometry.convert_points_to_homogeneous(points) # (100, 100, 8, 4, 1)
    points = points.unsqueeze(dim=-1) # (100, 100, 8, 4)
    project = project.unsqueeze(dim=1) # (1, 1, 1, 1, 3, 4)

    # Transform points to image and get depths
    points_t = project @ points # (1, 100, 100, 8, 3, 1)
    points_t = points_t.squeeze(dim=-1) # (1, 100, 100, 8, 3)
    points_img = kornia.geometry.convert_points_from_homogeneous(points_t) # (1, 100, 100, 8, 2)
    points_depth = points_t[..., -1] - project[..., 2, 3] # (1, 100, 100, 8)

    return points_img, points_depth
