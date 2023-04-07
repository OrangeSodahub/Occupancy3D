import torch

bs = 2
H, W, Z = 100, 100, 8
pc_range = torch.Tensor([-40, -40, -1, 40, 40, 5.4])
num_cam = 6

zs = torch.linspace(0.5, Z - 0.5, Z).view(Z, 1, 1).expand(Z, H, W) / Z
xs = torch.linspace(0.5, W - 0.5, W).view(1, 1, W).expand(Z, H, W) / W
ys = torch.linspace(0.5, H - 0.5, H).view(1, H, 1).expand(Z, H, W) / H
ref_3d = torch.stack((xs, ys, zs), -1)
ref_3d = ref_3d.permute(3, 0, 1, 2).flatten(1).permute(1, 0)
ref_3d = ref_3d[None, None].repeat(bs, 1, 1, 1)


reference_points = ref_3d.clone()
reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
reference_points = reference_points.permute(1, 0, 2, 3) # (num_points_in_pillar, bs, h*w, 4)
D, B, num_query = reference_points.size()[:3] # D=num_points_in_pillar , num_query=h*w
reference_points = reference_points.view(
    D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1) # (num_points_in_pillar, bs, num_cam, h*w, 4)


print(reference_points)
print(reference_points.shape)