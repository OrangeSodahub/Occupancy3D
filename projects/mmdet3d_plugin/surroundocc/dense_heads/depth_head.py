from collections import OrderedDict
from mmdet.models import HEADS
from ..modules.layers import *
from ..modules.transformer import *


@HEADS.register_module()
class DepthHead(nn.Module):
    def __init__(self,
                 skip=False,
                 shape=[1080, 1920],
                 num_ch_enc=[64, 64, 128, 256, 512],
                 scales=range(4),
                 num_output_channels=1,
                 use_skips=True,
                 frame_ids=[0, -1, 1],
                 use_fix_mask=True,
                 v1_multiscale=True,
                 disable_automasking=True,
                 avg_reprojection=True,
                 predictive_mask=True,
                 use_sfm_spatial=True,
                 spatial=True,
                 disparity_smoothness=1e-3,
                 match_spatial_weight=0.1,
                 spatial_weight=0.1,
                 no_ssim=False,
                 ):

        super(DepthHead, self).__init__()

        self.height, self.width = shape
        self.skip = skip
        self.num_output_channels = num_output_channels
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.num_scales = len(scales)
        self.use_skips = use_skips

        self.iter_num = [8, 8, 8, 8, 8]
        self.num_ch_enc = np.array(num_ch_enc[1:]*4)
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.frame_ids=frame_ids
        self.use_fix_mask=use_fix_mask
        self.v1_multiscale=v1_multiscale
        self.disable_automasking=disable_automasking
        self.avg_reprojection=avg_reprojection
        self.predictive_mask=predictive_mask
        self.use_sfm_spatial=use_sfm_spatial
        self.spatial=spatial
        self.disparity_smoothness=disparity_smoothness
        self.match_spatial_weight=match_spatial_weight
        self.spatial_weight=spatial_weight
        self.no_ssim=no_ssim

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)


            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)


        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        self.cross = {}

        for i in range(len(self.num_ch_enc)):
            self.cross[i] = CVT(
                input_channel=self.num_ch_enc[i],
                downsample_ratio=2**(len(self.num_ch_enc) -1 - i),
                iter_num=self.iter_num[i])

        self.decoder_cross = nn.ModuleList(list(self.cross.values()))

    def forward(self, input_features):
        self.outputs = {}
        for i in range(len(input_features)):
            B, C, H, W = input_features[i].shape
            if self.skip:
                input_features[i] = input_features[i] + self.cross[i](input_features[i].reshape(-1, 6, C, H, W)).reshape(B, C, H, W)
            else:
                input_features[i] = self.cross[i](input_features[i].reshape(-1, 6, C, H, W)).reshape(B, C, H, W)
        
        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def loss(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.scales:
            loss = 0
            reprojection_losses = []
            if self.use_fix_mask:
                output_mask = []

            if self.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.height, self.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.use_fix_mask:
                reprojection_losses *= inputs["mask"] #* output_mask

            if self.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            if self.use_sfm_spatial:
                depth_losses = []
                for j in range(len(inputs["match_spatial"])):
                    depth_loss = torch.abs(outputs[("depth_match_spatial", scale)][j] - inputs[("depth_match_spatial", scale)][j]).mean()
                    depth_losses.append(depth_loss)
                loss += self.match_spatial_weight * torch.stack(depth_losses).mean()

            if self.spatial:
                reprojection_losses_spatial = []
                spatial_mask = []
                target = inputs[("color", 0, source_scale)]

                for frame_id in self.frame_ids[1:]:
                    pred = outputs[("color_spatial", frame_id, scale)]

                    reprojection_losses_spatial.append(outputs[("color_spatial_mask", frame_id, scale)] * self.compute_reprojection_loss(pred, target))
                    
                reprojection_loss_spatial = torch.cat(reprojection_losses_spatial, 1)
                if self.use_fix_mask:
                    reprojection_loss_spatial *= inputs["mask"]
                
                loss += self.spatial_weight * reprojection_loss_spatial.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses
