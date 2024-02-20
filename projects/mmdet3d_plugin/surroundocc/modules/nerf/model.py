import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple, List, Union, Callable

from proj import utils


# Positional encoding (section 5.1)
class Embedder:
    """
        Given input as x (3-dimension coordinate)
        And the periodic_fns (sin, cos), freq_bands-l:
        positional encoding (log_sampling):
            [x, sin(2^0*x), cos(2^0*x), sin(2^1*x), cos(2^1*x), ...
             sin(2^(l-1)*x), cos(2*(l-1)*x)] (default l=10)
        out dims =  3 + 3*N_freqs (default 10) = 33
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """Official NeRF implementation.
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs


class Renderer:
    def __init__(self, hw: list, K: Tensor=None, act_fns: dict=None,
                 embedder: Embedder=None, nerf_model: NeRF=None, nerf_model_fine: NeRF=None,
                 N_samples: int=64, N_importance: int=0, embedderdirs: Embedder=None,
                 retraw: bool=False, white_bkgd: bool=False, use_viewdirs: bool=False,
                 no_ndc: bool=False, lindisp: bool=False, use_intervals: bool=True,
                 dataset_type: str='blender',
                 **kwargs) -> None:
        """
        Args:
            H: int. Height of image in pixels.
            W: int. Width of image in pixels.
            K: Tensor. Camera intrinsics matrix.
        """
        self.kwargs = kwargs
        # Basic render requirements and models
        self._H, self._W = hw
        self._K = K
        self._embedder = embedder
        self._embedderdirs = embedderdirs
        self._nerf_coarse = nerf_model
        self._nerf_fine = nerf_model_fine
        self._act_fns = act_fns

        # Render settings
        self._N_importance = N_importance
        self._N_samples = N_samples
        self._retraw = retraw
        self._white_bkgd = white_bkgd
        self._use_viewdirs = use_viewdirs
        self._use_intervals = use_intervals

        # NDC only good for LLFF-style forward facing data
        self._ndc = not no_ndc
        self._lindisp = False
        if dataset_type != 'llff' or no_ndc:
            self._ndc = False
            self._lindisp = lindisp

    @staticmethod
    def _get_rays(H, W, K, c2w, invert: bool=False, mode: str='lefttop', normalize_dir: bool=False) -> Tuple[Tensor, Tensor]:
        H, W = int(H), int(W)
        i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='xy')  # pytorch's meshgrid has indexing='ij'
        i = i.t().to(c2w.device)
        j = j.t().to(c2w.device)
        if mode == 'center':
            i = i + 0.5; j = j + 0.5
        # Get the unit direction vector representation of per pixel on the image plane relative to the camera center
        # NOTE: this step depends on the camera coordinates frame
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1).to(c2w.device)
        if invert:
            dirs[..., 1:] = -dirs[..., 1:]
        # Rotate ray directions from camera frame to the world frame
        # rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1) # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        rays_d = c2w[:3, :3].matmul(dirs[..., None]).squeeze(-1)
        if normalize_dir:
            rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = c2w[:3, -1].expand(rays_d.shape)
        return rays_o, rays_d

    @staticmethod
    def _get_rays_np(H, W, K, c2w, invert: bool=False, mode: str='lefttop', normalize_dir: bool=False) -> Tuple[np.ndarray, np.ndarray]:
        H, W = int(H), int(W)
        i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
        if mode == 'center':
            i = i + 0.5; j = j + 0.5
        dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
        if invert:
            dirs[..., 1:] = -dirs[..., 1:]
        rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1) # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        if normalize_dir:
            rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
        rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
        return rays_o, rays_d

    @staticmethod
    def _rays_from_image(H, W, K, c2w, normalize_dir: bool=False) -> Tuple[np.ndarray, np.ndarray]:
        H, W = int(H), int(W)
        # pixel coords
        i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
        xy_coords = np.stack([i, j], axis=-1) # [H, W, 2]

        # camera coords
        pts = utils.unproject(K, xy_coords, depth=1) # [H, W, 4]
        # global coords
        pts = pts @ c2w.T
        rays_d = pts - c2w[:, -1]  # equivalently can subtract [0,0,0,1] before pose transform
        rays_d = rays_d[..., :3]
        if normalize_dir:
            rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
        rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
        return rays_o, rays_d

    @staticmethod
    def _ndc_rays(H, W, focal, near, rays_o, rays_d) -> Tuple[Tensor, Tensor]:
        # Shift ray origins to near plane
        t = -(near + rays_o[...,2]) / rays_d[...,2]
        rays_o = rays_o + t[...,None] * rays_d
        
        # Projection
        o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
        o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
        o2 = 1. + 2. * near / rays_o[...,2]

        d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
        d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
        d2 = -2. * near / rays_o[...,2]
        
        rays_o = torch.stack([o0,o1,o2], -1)
        rays_d = torch.stack([d0,d1,d2], -1)
        
        return rays_o, rays_d

    # Hierarchical sampling (section 5.2)
    @staticmethod
    def _sample_pdf(bins, weights, N_samples, det=False) -> Tensor:
        # Pad each weight vector (only if necessary) to bring its sum to `eps`. This
        # avoids NaNs when the input is zeros or small, but has no effect otherwise.
        eps = 1e-5
        weight_sum = torch.sum(weights, dim=-1, keepdim=True)
        padding = torch.maximum(torch.zeros_like(weight_sum), eps - weight_sum)
        weights = weights + padding / weights.shape[-1]
        weight_sum = weight_sum + padding

        # Get pdf
        pdf = weights / weight_sum # [B, N_cam, N_rays, N_samples]
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

        # Take uniform samples
        if det:
            u = torch.linspace(0., 1., steps=N_samples)
            u = u.expand(list(cdf.shape[:-1]) + [N_samples])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

        # Invert CDF
        u = u.to(cdf).contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds-1), inds-1)
        above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

        # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        # NOTE: inds_g.shape = [B, N_cam, N_ray, N_sample, 2] or [B, N_sample, 2]
        matched_shape = [*inds_g.shape[:-1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), -1, inds_g)
        bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), -1, inds_g)

        denom = (cdf_g[..., 1] - cdf_g[..., 0])
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

        return samples

    def _raw2outputs(self, raw, z_vals, rays_d, raw_noise_std=0.,
                     act_sigma: Callable=F.relu, act_rgb: Callable=F.sigmoid,
                     return_raw_density: bool=False) -> Tuple[Tensor]:
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
                here 4 => rgb prediction (3) and density prediction (1)
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """
        dists = 1.
        if self._use_intervals:
            dists = z_vals[..., 1:] - z_vals[..., :-1]
            dists = torch.cat([dists, torch.Tensor([1e10]).to(dists).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
            dists = dists * torch.norm(rays_d[..., None, :], dim=-1) # [(B, N_cam,) N_rays, N_samples]

        # rgb(or feature) and density
        # NOTE: feature dim > 3, rgb dim == 3, density dim == 1
        raw_density = raw[..., -1] if return_raw_density else None
        rgb = act_rgb(raw[..., :-1])  # [(B, N_cam,) N_rays, N_samples, (C)3]
        noise = torch.randn(raw[..., -1].shape).to(raw) * raw_noise_std if raw_noise_std > 0. else 0.
        delta = act_sigma(raw[..., -1] + noise) * dists # [(B, N_cam,) N_rays, N_samples]
        alpha = 1 - torch.exp(-delta)

        # calculate transmittance (section 5.1) and weights (section 5.3)
        trans = torch.exp(-torch.cat([
            torch.zeros_like(alpha[..., :1]),
            torch.cumsum(delta[..., :-1], -1)], dim=-1))
        weights = alpha * trans # [(B, N_cam,) N_rays, N_samples]

        # sum the rgb(or feature) weighted by the alpha of all rays
        rgb_map = torch.sum(weights[..., None] * rgb, -2) # [(B, N_cam,) N_rays, (C)3]
        acc_map = torch.sum(weights, -1) # [(B, N_cam,) N_rays]
        depth_map = torch.sum(weights * z_vals, -1) / (torch.sum(weights, dim=-1) + 1e-8) # [(B, N_cam,) N_rays]
        depth_map = torch.clip(torch.nan_to_num(depth_map, float('inf')), z_vals[..., 0], z_vals[..., -1])
        disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)

        if self._white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[..., None])

        return rgb_map, disp_map, acc_map, weights, depth_map, raw_density

    def _run_model(self, inputs: Tensor, viewdirs: Tensor, run_fine: bool=False, **kwargs) -> Tensor:
        """
        Args:
            inputs: points in shape [N_rays, N_samples, 3]
        """
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        # positional encoding of 3d coordinates
        embedded = self._embedder.embed(inputs_flat) # (65536, 63), (196608, 63)

        if viewdirs is not None:
            input_dirs = viewdirs[:, None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            # positional encoding of 2d directions
            embedded_dirs = self._embedderdirs.embed(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        # Run NeRF model
        model = self._nerf_coarse
        if run_fine and self._nerf_fine is not None:
            model = self._nerf_fine
        outputs_flat = model(embedded)
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]]) # (1024, 64, 4), (1024, 192, 4)
        return outputs

    def _render_rays(self, ray_batch: Tensor, perturb: bool=False, raw_noise_std: float=0., **kwargs)-> dict:
        """Volumetric rendering.
        Args:
            ray_batch: array of shape [batch_size, ...]. All information necessary
                for sampling along a ray, including: ray origin, ray direction, min
                dist, max dist, and unit-magnitude viewing direction.

        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
            disp_map: [num_rays]. Disparity map. 1 / depth.
            acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
            raw: [num_rays, num_samples, 4]. Raw predictions from model.
            rgb0: See rgb_map. Output for coarse model.
            disp0: See disp_map. Output for coarse model.
            acc0: See acc_map. Output for coarse model.
            z_std: [num_rays]. Standard deviation of distances along ray for each
                sample.
        """
        rays_o, rays_d = ray_batch[..., 0:3], ray_batch[..., 3:6] # [B, N_cam, N_rays, 3] / [N_rays, 3] each
        viewdirs = ray_batch[..., -3:] if ray_batch.shape[-1] > 8 else None
        near, far = ray_batch[..., 6, None], ray_batch[..., 7, None] # [B, N_cam, N_rays, 1]

        t_vals = torch.linspace(0., 1., steps=self._N_samples).to(rays_o.device) # [N_samples]
        # lindisp is for llff data
        if not self._lindisp:
            z_vals = near * (1.-t_vals) + far * (t_vals)
        else:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

        # random z vals for training
        if perturb:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(z_vals)

            z_vals = lower + (upper - lower) * t_rand

        # get the 3D coordinates in the world frame of N_samples of N_rays
        # rays_o is the origin point, rays_d defines the direction, z_vals is the step along the direction
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None] # [N_rays, N_samples, 3]=(1024, 64, 3)

        raw, view_masks = self._run_model(pts, viewdirs, **kwargs)
        outputs = self._raw2outputs(raw, z_vals, rays_d, raw_noise_std, **self._act_fns,
                                    return_raw_density=kwargs.get('return_raw_density', False))
        rgb_map, disp_map, acc_map, weights, depth_map, opacity = outputs

        if self._N_importance > 0:

            rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = self._sample_pdf(z_vals_mid, weights[..., 1:-1], self._N_importance, det=(perturb))
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None] # [N_rays, N_samples + N_importance, 3]

            # use the fine model
            raw = self._run_model(pts, viewdirs, True, **kwargs)
            outputs = self._raw2outputs(raw, z_vals, rays_d, raw_noise_std)
            rgb_map, disp_map, acc_map, weights, depth_map, opacity = outputs

        ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'depth_map': depth_map}
        if opacity is not None:
            ret['opacity'] = opacity
        if view_masks is not None:
            ret['masks'] = view_masks

        if self._retraw:
            ret['raw'] = raw
        if self._N_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc0'] = acc_map_0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

        return ret
    
    def render(self, chunk: int=1024*32, rays: Tensor=None, c2w: Tensor=None, near: float=0.,
                far: float=1., c2w_staticcam: Tensor=None, perturb: bool=False, raw_noise_std: float=0.,
                render_factor: int=0) -> List[Union[Tensor, dict]]:
        """Render rays
        Args:
            chunk: int. Maximum number of rays to process simultaneously. Used to
                control maximum memory usage. Does not affect final results.
            rays: array of shape [2, batch_size, 3]. Ray origin and direction for
                each example in batch.
            c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
            ndc: bool. If True, represent ray origin, direction in NDC coordinates.
            near: float or array of shape [batch_size]. Nearest distance for a ray.
            far: float or array of shape [batch_size]. Farthest distance for a ray.
            use_viewdirs: bool. If True, use viewing direction of a point in space in model.
            c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
            camera while using other c2w argument for viewing directions.
        Returns:
            rgb_map: [batch_size, 3]. Predicted RGB values for rays.
            disp_map: [batch_size]. Disparity map. Inverse of depth.
            acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
            extras: dict with everything returned by render_rays().
        """
        # Prepare rays first
        if c2w is not None:
            # special case to render full image
            H, W = self._H, self._W
            if render_factor != 0:
                H = H // render_factor
                W = W // render_factor
            rays_o, rays_d = self._get_rays(H, W, self._K, c2w)
        else:
            # use provided ray batch
            rays_o, rays_d = rays
        if self._use_viewdirs:
            # provide ray directions as input
            viewdirs = rays_d
            if c2w_staticcam is not None:
                # special case to visualize effect of viewdirs
                rays_o, rays_d = self._get_rays(self._H, self._W, self._K, c2w_staticcam)
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
        sh = rays_d.shape # [..., 3]
        if self._ndc:
            # for forward facing scenes
            rays_o, rays_d = self._ndc_rays(self._H, self._W, self._K[0][0], 1., rays_o, rays_d)
        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()
        near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
        # fuse the distance bound information of rays into rays
        rays = torch.cat([rays_o, rays_d, near, far], -1)
        if self._use_viewdirs:
            rays = torch.cat([rays, viewdirs], -1)

        # Render and reshape (batchify to avoid OOM)
        # rays shape: (N_rand, 11)=（1024，11）
        all_ret = {}
        for i in range(0, rays.shape[0], chunk):
            ret = self._render_rays(rays[i:i+chunk], perturb, raw_noise_std)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        k_extract = ['rgb_map', 'disp_map', 'acc_map']
        ret_list = [all_ret[k] for k in k_extract]
        ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
        return ret_list + [ret_dict]
