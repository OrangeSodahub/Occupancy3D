"""Implementation of vanilla NeRF, adapted from pytorch-nerf.
"""
import os
import torch
import wandb
import imageio
import numpy as np
import torch.nn as nn
import configargparse
from torch import Tensor
from tqdm import tqdm, trange
from typing import Union, Tuple
from proj.datasets import BlenderObject
from proj.modules.nerf import Embedder, Renderer, NeRF


class Wizard:
    def __init__(self, args):
        self._args = args
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._create_model()
        self._optimizer = torch.optim.Adam(params=self.grad_vars, lr=self._args.lrate, betas=(0.9, 0.999))
        self._load_ckpt()
        self._global_step = self._start
        self._renderer = None

        # loss functions
        self.img2mse = lambda x, y : torch.mean((x - y) ** 2)
        self.mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(x))
        self.to8b = lambda x : (255 * np.clip(x, 0, 1)).astype(np.uint8)

    def _load_ckpt(self):
        # load checkpoints
        if self._args.ft_path is not None and self._args.ft_path!='None':
            ckpts = [self._args.ft_path]
        else:
            ckpts = [os.path.join(self._args.basedir, self._args.expname, f) for f 
                    in sorted(os.listdir(os.path.join(self._args.basedir, self._args.expname))) if 'tar' in f]
        if len(ckpts) > 0 and not self._args.no_reload:
            print(f"Found ckpts {ckpts} ...")
            ckpt_path = ckpts[-1]
            ckpt = torch.load(ckpt_path)
            self._start = ckpt['global_step']
            print(f"Resume from {ckpt_path} , start from {self._start} step ...")
            self._optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self._nerf_model.load_state_dict(ckpt['model_state_dict'])
            if self._nerf_model_fine is not None:
                self._nerf_model_fine.load_state_dict(ckpt['model_fine_state_dict'])
        else:
            print(f"No available ckpts found .")

    def _get_embedder(self, multires: int, i: int=0):
        """
            multires: log2 of max freq for positional encoding (3D location)
            i_embed: set 0 for default positional encoding, -1 for none
        """
        
        # no positional embeddings
        if i == -1:
            return nn.Identity(), 3
        
        embed_kwargs = {
            'include_input' : True,
            'input_dims' : 3,
            'max_freq_log2' : multires-1,
            'num_freqs' : multires,
            'log_sampling' : True,
            'periodic_fns' : [torch.sin, torch.cos],
        }
        
        # [x, sin(2^0*x), cos(2^0*x), sin(2^1*x), cos(2^1*x), ...
        # sin(2^(l-1)*x), cos(2*(l-1)*x)] (default l=10)
        embedder = Embedder(**embed_kwargs)
        return embedder, embedder.out_dim
    
    def _create_model(self):
        """Instantiate NeRF's MLP model.
        """
        self._start = 0
        # Create log dir and copy the config file
        basedir = self._args.basedir
        expname = self._args.expname
        os.makedirs(os.path.join(basedir, expname), exist_ok=True)
        f = os.path.join(basedir, expname, 'args.txt')
        with open(f, 'w') as file:
            for arg in sorted(vars(self._args)):
                attr = getattr(self._args, arg)
                file.write('{} = {}\n'.format(arg, attr))
        if self._args.config is not None:
            f = os.path.join(basedir, expname, 'config.txt')
            with open(f, 'w') as file:
                file.write(open(self._args.config, 'r').read())
        
        # Create nerf model
        print(f"create nerf model ...")
        self._embedder, input_ch = self._get_embedder(self._args.multires, self._args.i_embed)
        input_ch_views = 0
        self._embedderdirs = None
        # whether the input has the view direction
        # if yes, the input wil be 5D, or will be 3D
        if self._args.use_viewdirs:
            # multires_views: 'log2 of max freq for positional encoding (2D direction)'
            self._embedderdirs, input_ch_views = self._get_embedder(self._args.multires_views, self._args.i_embed)

        # N_importance: number of additional fine samples per ray
        output_ch = 5 if self._args.N_importance > 0 else 4
        skips = [4]
        # netdepth: num layers in network, default 8
        # netwidth: num channels per layer, default 256
        self._nerf_model = NeRF(D=self._args.netdepth, W=self._args.netwidth, input_ch=input_ch,
                                output_ch=output_ch, skips=skips, input_ch_views=input_ch_views,
                                use_viewdirs=self._args.use_viewdirs).to(self._device)

        self.grad_vars = list(self._nerf_model.parameters())
        self._nerf_model_fine = None
        if self._args.N_importance > 0:
            self._nerf_model_fine = NeRF(D=self._args.netdepth_fine, W=self._args.netwidth_fine, input_ch=input_ch,
                                        output_ch=output_ch, skips=skips, input_ch_views=input_ch_views,
                                        use_viewdirs=self._args.use_viewdirs).to(self._device)
            self.grad_vars += list(self._nerf_model_fine.parameters())
        
        # build wandb
        if self._args.wandb:
            try:
                wandb.init(project="Base-Nerf", entity="gzzyyxy")
            except:
                print('Initialize wandb failed.')
            

    def _train_step(self, i: int, input: BlenderObject, poses: Tensor, images: Union[np.ndarray, Tensor],
                    rays_rgb: Union[np.ndarray, Tensor], N_rand: int, use_batching: bool, i_train: int
                    ) -> Tuple[Tensor, Tensor]:
        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[self._i_batch : self._i_batch + N_rand] # [N_rand, ro+rd+rgb, 3]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            self._i_batch += N_rand
            if self._i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                self._i_batch = 0

        else:
            # Random from one image
            # images: shape (N, H, W, 3)
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(self._device)
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                # rays_o: the origin coords in the world frame of per pixel
                # rays_d: the direction in the world frame of per pixel
                rays_o, rays_d = self._renderer._get_rays(input.H, input.W, input.K, pose)  # (H, W, 3), (H, W, 3)

                if i < self._args.precrop_iters:
                    dH = int(input.H//2 * self._args.precrop_frac)
                    dW = int(input.W//2 * self._args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(input.H//2 - dH, input.H//2 + dH - 1, 2*dH), 
                            torch.linspace(input.W//2 - dW, input.W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == self._start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {self._args.precrop_iters}")                
                else:
                    # 2d coords in the image frame of per pixel
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, input.H-1, input.H), torch.linspace(0, input.W-1, input.W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                # the origin coords and directions of sampled N_rand pixels
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0) # (2, N_rand, 3)=(2, 1024, 3)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        # core render operation
        rgb, disp, acc, extras = self._renderer.render(self._args.chunk, batch_rays, perturb=self._args.perturb, near=input.near,
                                                                            far=input.far, raw_noise_std=self._args.raw_noise_std)
        self._optimizer.zero_grad()
        img_loss = self.img2mse(rgb, target_s)
        loss = img_loss
        psnr = self.mse2psnr(img_loss)
        try:
            wandb.log({'loss_fine': img_loss, 'psnr_fine': psnr})
        except:
            pass

        if 'rgb0' in extras:
            img_loss0 = self.img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = self.mse2psnr(img_loss0)
            try:
                wandb.log({'loss_coarse': img_loss0, 'psnr_coarse': psnr0})
            except:
                pass

        return loss, psnr

    def train(self, input: BlenderObject) -> None:
        self._nerf_model.train()
        if self._nerf_model_fine is not None:
            self._nerf_model_fine.train()

        i_train, i_val, i_test = input.i_split
        # create renderer
        self._renderer = Renderer(self._device, input.hwf[:2], input.K, self._embedder, self._nerf_model, self._args.N_samples,
                            self._args.N_importance, self._embedderdirs, self._nerf_model_fine, True, self._args.white_bkgd,
                            self._args.use_viewdirs, self._args.no_ndc, self._args.lindisp, self._args.dataset_type)

        N_rand = self._args.N_rand
        use_batching = not self._args.no_batching
        # Prepare raybatch tensor if batching random rays
        rays_rgb = None
        if use_batching:
            print('Using batching, and get rays ...')
            rays = np.stack([self._renderer._get_rays_np(input.H, input.W, input.K, p) for p in input.poses[:, :3, :4]], 0) # [N, ro+rd, H, W, 3]
            print(f'done, concat rays shape: {rays.shape}')
            rays_rgb = np.concatenate([rays, input.imgs[:, None]], 1) # [N, ro+rd+rgb, H, W, 3]
            rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4]) # [N, H, W, ro+rd+rgb, 3]
            rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
            rays_rgb = np.reshape(rays_rgb, [-1, 3, 3]) # [(N-1)*H*W, ro+rd+rgb, 3]
            rays_rgb = rays_rgb.astype(np.float32)
            np.random.shuffle(rays_rgb)

            self._i_batch = 0

        # moving to GPU
        images = input.imgs
        poses = torch.Tensor(input.poses).to(self._device)
        if use_batching:
            images = torch.Tensor(images).to(self._device)
            rays_rgb = torch.Tensor(rays_rgb).to(self._device)

        N_iters = 200000 + 1
        print('Begin training ...')
        print('TRAIN views are', i_train)
        print('TEST views are', i_test)
        print('VAL views are', i_val)
        # training
        # N_rand: batch size (number of random rays per gradient step), default 32*32*4
        self._start = self._start + 1
        for i in trange(self._start, N_iters):
            loss, psnr = self._train_step(i, input, poses, images, rays_rgb, N_rand, use_batching, i_train)
            loss.backward()
            self._optimizer.step()

            # update learning rate
            decay_rate = 0.1
            decay_steps = self._args.lrate_decay * 1000
            new_lrate = self._args.lrate * (decay_rate ** (self._global_step / decay_steps))
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = new_lrate

            # backup at interval
            if i % self._args.i_weights == 0:
                path = os.path.join(self._args.basedir, self._args.expname, '{:06d}.tar'.format(i))
                torch.save({
                    'global_step': self._global_step,
                    'model_state_dict': self._nerf_model.state_dict(),
                    'model_fine_state_dict': self._nerf_model_fine.state_dict() if self._nerf_model_fine is not None else None,
                    'optimizer_state_dict': self._optimizer.state_dict(),
                }, path)
                print(f"Saved checkpoints at {path}, global step {self._global_step}")

            # Print training log
            if i % self._args.i_print == 0:
                tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

            # Evaluation
            if i % self._args.i_video == 0:
                self.test(input)
            
            self._global_step += 1

    def test(self, input: BlenderObject) -> None:
        self._nerf_model.eval()
        if self._nerf_model_fine is not None:
            self._nerf_model_fine.eval()

        i_train, i_val, i_test = input.i_split
        # Create renderer
        if self._renderer is None:
            self._renderer = Renderer(self._device, input.hwf[:2], input.K, self._embedder, self._nerf_model, self._args.N_samples,
                                self._args.N_importance, self._embedderdirs, self._nerf_model_fine, True, self._args.white_bkgd,
                                self._args.use_viewdirs, self._args.no_ndc, self._args.lindisp, self._args.dataset_type)

        # render
        print(f"Render test ...")
        with torch.no_grad():
            testsavedir = os.path.join(self._args.basedir, self._args.expname, 'renderonly_{}_{:06d}'.format(
                'test' if self._args.render_test else 'path', self._global_step
            ))
            os.makedirs(testsavedir, exist_ok=True)
            test_poses = torch.Tensor(input.poses[i_test]).to(self._device)
            print(f"Test poses shape: {test_poses.shape}")

            rgbs, disps = [] ,[]
            for i, c2w in enumerate(tqdm(test_poses)):
                rgb, disp, _, _ = self._renderer.render(self._args.chunk, c2w=c2w[:3, :4], perturb=False,
                                                        near=input.near, far=input.far, raw_noise_std=0.,
                                                        render_factor=self._args.render_factor)
                rgbs.append(rgb.cpu().numpy())
                disps.append(disp.cpu().numpy())
                if i == 0:
                    print(f"rgb shape {rgb.shape}, disp shape: {disp.shape}")

                if testsavedir is not None:
                    rgb8 = self.to8b(rgbs[-1])
                    filename = os.path.join(testsavedir, '{:03d}.png'.format(i))
                    imageio.imwrite(filename, rgb8)

            rgbs = np.stack(rgbs, 0)
            disps = np.stack(disps, 0)

            print(f"Rendering done and save to {testsavedir}")
            video_path = os.path.join(testsavedir, 'vieo.mp4')
            imageio.mimwrite(video_path, self.to8b(rgbs), fps=30, quality=8)


# TODO: fix configurations
def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, default='./configs/lego.txt', 
                        help='config file path')
    parser.add_argument("--expname", type=str, default='lego', 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./work_dirs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/lego/', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*4, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--wandb", action='store_true')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    return parser