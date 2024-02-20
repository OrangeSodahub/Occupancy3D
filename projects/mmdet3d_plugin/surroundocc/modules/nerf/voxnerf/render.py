import torch
import numpy as np
from torch import Tensor
from typing import Sequence

from proj import utils
from proj.modules.nerf.model import Renderer
from proj.modules.nerf.voxnerf.vox import VoxNeRF


class VoxRenderer(Renderer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _ray_box_intersect(ro: np.ndarray, rd: np.ndarray, aabb: list) -> Sequence[np.ndarray]:
        """
        Intersection of ray with axis-aligned bounding box
        This routine works for arbitrary dimensions; commonly d = 2 or 3
        only works for numpy, not torch (which has slightly diff api for min, max, and clone)

        Args:
            ro: [n, d] ray origin
            rd: [n, d] ray direction (assumed to be already normalized;
                if not still fine, meaning of t as time of flight holds true)
            aabb: [d, 2] bbox bound on each dim
        Return:
            is_intersect: [n,] of bool, whether the particular ray intersects the bbox
            t_min: [n,] ray entrance time
            t_max: [n,] ray exit time
        """
        N = ro.shape[0]
        D = aabb.shape[0]
        assert aabb.shape == (D, 2)
        assert ro.shape == (N, D) and rd.shape == (N, D)

        rd = rd.copy()
        rd[rd == 0] = 1e-6  # avoid div overflow; logically safe to give it big t

        ro = ro.reshape(N, D, 1)
        rd = rd.reshape(N, D, 1)
        ts = (aabb - ro) / rd  # [n, d, 2]
        t_min = ts.min(-1).max(-1)  # [n,] last of entrance
        t_max = ts.max(-1).min(-1)  # [n,] first of exit
        is_intersect = t_min < t_max

        return is_intersect, t_min, t_max

    @classmethod
    def scene_box_filter(cls, ro: np.ndarray, rd: np.ndarray, aabb: list) -> Sequence[np.ndarray]:
        _, t_min, t_max = cls._ray_box_intersect(ro, rd, aabb)
        # do not render what's behind the ray origin
        t_min, t_max = np.maximum(t_min, 0), np.maximum(t_max, 0)
        return t_min, t_max

    # TODO: this function could be simplified, now use this one
    def _render_rays(self, model: VoxNeRF, ro: Tensor, rd: Tensor, t_min: Tensor, t_max: Tensor) -> Sequence[Tensor]:
        """
        The working shape is (k, n, 3) where k is num of samples per ray, n the ray batch size
        During integration the reduction is applied on k

        chain of filtering
        starting with ro, rd (from cameras), and a scene bbox
        - rays that do not intersect scene bbox; sample pts that fall outside the bbox
        - samples that do not fall within alpha mask
        - samples whose densities are very low; no need to compute colors on them
        """
        num_samples, step_size = model.get_num_samples((t_max - t_min).max())
        N_rays, N_samples = len(ro), num_samples

        ticks = step_size * torch.arange(N_samples, device=ro.device)
        ticks = ticks.view(N_samples, 1, 1).contiguous()
        t_min = t_min.view(N_rays, 1).contiguous()
        t_max = t_max.view(N_rays, 1).contiguous()
        dists = t_min + ticks  # [N_rays, 1], [N_samples, 1, 1] -> [N_samples, N_rays, 1]
        pts = ro + rd * dists  # [N_rays, 3], [N_rays, 3], [N_samples, N_rays, 1] -> [N_samples, N_rays, 3]
        mask = (ticks < (t_max - t_min)).squeeze(-1)  # [N_samples, 1, 1], [N_rays, 1] -> [N_samples, N_rays, 1] -> [N_samples, N_rays]
        sample_pts = pts[mask]

        if model.alphaMask is not None:
            alphas = model.alphaMask.sample_alpha(sample_pts)
            alpha_mask = alphas > 0
            mask[mask.clone()] = alpha_mask
            sample_pts = pts[mask]

        σ = torch.zeros(N_samples, N_rays, device=ro.device)
        σ[mask] = model.sample_density(sample_pts)
        weights = volume_rend_weights(σ, step_size)
        mask = weights > model.ray_march_weight_thres
        sample_pts = pts[mask]

        app_feats = model.sample_feature(sample_pts)
        # viewdirs = rd.view(1, n, 3).expand(k, n, 3)[mask]  # ray dirs for each point
        # additional wild factors here as in nerf-w; wild factors are optimizable
        c_dim = app_feats.shape[-1]
        colors = torch.zeros(N_samples, N_rays, c_dim, device=ro.device)
        colors[mask] = model.feats2color(app_feats)

        weights = weights.view(N_samples, N_rays, 1)  # can be used to compute other expected vals e.g. depth
        bg_weight = 1. - weights.sum(dim=0)  # [n, 1]

        rgbs = (weights * colors).sum(dim=0)  # [n, 3]

        if model.blend_bg_texture:
            uv = utils.spherical_xyz_to_uv(rd)
            bg_feats = model.sample_bg(uv)
            bg_color = model.feats2color(bg_feats)
            rgbs = rgbs + bg_weight * bg_color
        else:
            rgbs = rgbs + bg_weight * 1.  # blend white bg color

        # rgbs = rgbs.clamp(0, 1)  # don't clamp since this is can be SD latent features

        depths = (weights * dists).sum(dim=0)
        bg_dist = 10.  # blend bg distance; just don't make it too large
        depths = depths + bg_weight * bg_dist
        return rgbs, depths, weights.squeeze(-1)


def volume_rend_weights(σ, dist):
    α = 1 - torch.exp(-σ * dist)
    T = torch.ones_like(α)
    T[1:] = (1 - α).cumprod(dim=0)[:-1]
    assert (T >= 0).all()
    weights = α * T
    return weights
