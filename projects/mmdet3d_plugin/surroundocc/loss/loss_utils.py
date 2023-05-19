import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

def multiscale_supervision(voxel_semantics, ratio, gt_shape, original_coords, mask_camera, use_mask=False):
    '''
    change ground truth shape as (B, W, H, Z) for each level supervision
    '''

    # `gt_shape: (bs, num_classes, W, H, Z)`
    # `gt: (bs, W, H, Z)`
    gt = torch.zeros([gt_shape[0], gt_shape[2], gt_shape[3], gt_shape[4]]).to(voxel_semantics.device).long()
    mask = gt.clone() if use_mask else None
    # In the dataset provided by CVPR2023 challenge, all the voxels
    # which has no labels (0-16) are labeled as 17 (free or empty)
    gt += 17
    for i in range(gt.shape[0]):
        # Roughly calculate the downsampled label
        original_coords = original_coords.to(voxel_semantics.device)
        voxel_semantics_with_coords = torch.vstack([original_coords.T, voxel_semantics[i].reshape(-1)]).T
        voxel_semantics_with_coords = voxel_semantics_with_coords[voxel_semantics_with_coords[:, 3] < 17]
        downsampled_coords = torch.div(voxel_semantics_with_coords[:, :3].long(), ratio, rounding_mode='floor')
        gt[i, downsampled_coords[:, 0], downsampled_coords[:, 1], downsampled_coords[:, 2]] = \
                                                                            voxel_semantics_with_coords[:, 3]
        # downsample the mask camera
        if mask is not None:
            mask_camera_with_coords = torch.vstack([original_coords.T, mask_camera[i].reshape(-1)]).T
            downsampled_coords = torch.div(mask_camera_with_coords[:, :3].long(), ratio, rounding_mode='floor')
            mask[i, downsampled_coords[:, 0], downsampled_coords[:, 1], downsampled_coords[:, 2]] = \
                                                                            mask_camera_with_coords[:, 3]
    return gt, mask

def geo_scal_loss(pred, ssc_target, mask=None, semantic=True):

    # Get softmax probabilities
    if semantic:
        pred = F.softmax(pred, dim=1)

        # Compute empty and nonempty probabilities
        empty_probs = pred[:, 17, :, :, :]
    else:
        empty_probs = 1 - torch.sigmoid(pred)
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    if mask is None:
        mask = ssc_target != 255
    else:
        mask = mask.bool()
    nonempty_target = ssc_target != 17
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
        + F.binary_cross_entropy(recall, torch.ones_like(recall))
        + F.binary_cross_entropy(spec, torch.ones_like(spec))
    )


def sem_scal_loss(pred, ssc_target, mask=None):
    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    n_classes = pred.shape[1]
    for i in range(0, n_classes):

        # Get probability of class i
        p = pred[:, i, :, :, :]

        # Remove unknown voxels
        if mask is None:
            mask = ssc_target != 255
        else:
            mask = mask.bool()
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        completion_target_ori = torch.ones_like(target_ori).float()
        completion_target_ori[target_ori != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0
            if torch.sum(p) > 0:
                precision = nominator / (torch.sum(p))
                loss_precision = F.binary_cross_entropy(
                    precision, torch.ones_like(precision)
                )
                loss_class += loss_precision
            if torch.sum(completion_target) > 0:
                recall = nominator / (torch.sum(completion_target))
                loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                loss_class += loss_recall
            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                    torch.sum(1 - completion_target)
                )
                loss_specificity = F.binary_cross_entropy(
                    specificity, torch.ones_like(specificity)
                )
                loss_class += loss_specificity
            loss += loss_class
    return loss / count
