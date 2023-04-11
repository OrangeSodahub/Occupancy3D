# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/VoxFormer/blob/main/LICENSE

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from projects.mmdet3d_plugin.surroundocc.loss.loss_utils import BEC_ssc_loss


class LMSCNet_SS():
	def __init__(self,
					class_num=None,
					input_dimensions=None,
					out_scale=None,
					):

		super(LMSCNet_SS, self).__init__()
		'''
		SSCNet architecture
		:param N: number of classes to be predicted (i.e. 12 for NYUv2)
		'''

		super().__init__()
		self.out_scale=out_scale
		self.nbr_classes = class_num
		self.input_dimensions = input_dimensions # (D, W, H)
		f = self.input_dimensions[2]

		self.pool = nn.MaxPool2d(2)  # [F=2; S=2; P=0; D=1]

		self.pooling = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))

		self.Encoder_block1 = nn.Sequential(
		nn.Conv2d(f, f, kernel_size=3, padding=1, stride=1),
		nn.ReLU(),
		nn.Conv2d(f, f, kernel_size=3, padding=1, stride=1),
		nn.ReLU()
		)

		self.Encoder_block2 = nn.Sequential(
		nn.MaxPool2d(2),
		nn.Conv2d(f, int(f*1.5), kernel_size=3, padding=1, stride=1),
		nn.ReLU(),
		nn.Conv2d(int(f*1.5), int(f*1.5), kernel_size=3, padding=1, stride=1),
		nn.ReLU()
		)

		self.Encoder_block3 = nn.Sequential(
		nn.MaxPool2d(2),
		nn.Conv2d(int(f*1.5), int(f*2), kernel_size=3, padding=1, stride=1),
		nn.ReLU(),
		nn.Conv2d(int(f*2), int(f*2), kernel_size=3, padding=1, stride=1),
		nn.ReLU()
		)

		self.Encoder_block4 = nn.Sequential(
		nn.MaxPool2d(2),
		nn.Conv2d(int(f*2), int(f*2.5), kernel_size=3, padding=1, stride=1),
		nn.ReLU(),
		nn.Conv2d(int(f*2.5), int(f*2.5), kernel_size=3, padding=1, stride=1),
		nn.ReLU()
		)

		# Treatment output 1:8
		self.conv_out_scale_1_8 = nn.Conv2d(int(f*2.5), int(f/8), kernel_size=3, padding=1, stride=1)
		self.deconv_1_8__1_2    = nn.ConvTranspose2d(int(f/8), int(f/8), kernel_size=4, padding=0, stride=4)
		self.deconv_1_8__1_1    = nn.ConvTranspose2d(int(f/8), int(f/8), kernel_size=8, padding=0, stride=8)

		# Treatment output 1:4
		if self.out_scale=="1_4" or self.out_scale=="1_2" or self.out_scale=="1_1":
			self.deconv1_8          = nn.ConvTranspose2d(int(f/8), int(f/8), kernel_size=6, padding=2, stride=2)
			self.conv1_4            = nn.Conv2d(int(f*2) + int(f/8), int(f*2), kernel_size=3, padding=1, stride=1)
			self.conv_out_scale_1_4 = nn.Conv2d(int(f*2), int(f/4), kernel_size=3, padding=1, stride=1)
			self.deconv_1_4__1_1    = nn.ConvTranspose2d(int(f/4), int(f/4), kernel_size=4, padding=0, stride=4)

		# Treatment output 1:2
		if self.out_scale=="1_2" or self.out_scale=="1_1":
			self.deconv1_4          = nn.ConvTranspose2d(int(f/4), int(f/4), kernel_size=6, padding=2, stride=2)
			self.conv1_2            = nn.Conv2d(int(f*1.5) + int(f/4) + int(f/8), int(f*1.5), kernel_size=3, padding=1, stride=1)
			self.conv_out_scale_1_2 = nn.Conv2d(int(f*1.5), int(f/2), kernel_size=3, padding=1, stride=1)

		# Treatment output 1:1
		if self.out_scale=="1_1":
			self.deconv1_2          = nn.ConvTranspose2d(int(f/2), int(f/2), kernel_size=6, padding=2, stride=2)
			self.conv1_1            = nn.Conv2d(int(f/8) + int(f/4) + int(f/2) + int(f), f, kernel_size=3, padding=1, stride=1)

		if self.out_scale=="1_1":
			self.seg_head_1_1 = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])
		elif self.out_scale=="1_2":
			self.seg_head_1_2 = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])
		elif self.out_scale=="1_4":
			self.seg_head_1_4 = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])
		elif self.out_scale=="1_8":
			self.seg_head_1_8 = SegmentationHead(1, 8, self.nbr_classes, [1, 2, 3])

	def foward(self, depth, img_metas):
		"""
		depth: voxel_featurs from depth net, shape (B, X, Y ,Z)->(1, 200, 200, 16)
		"""

		# input: (B, D, W, H)->(B, H, W, D)
		# output: (B, D, W, H)
		input = depth.permute(0, 3, 2, 1)
		
		# Encoder block
		_skip_1_1 = self.Encoder_block1(input)		# (1, 16, 200, 200)
		_skip_1_2 = self.Encoder_block2(_skip_1_1)
		_skip_1_4 = self.Encoder_block3(_skip_1_2) 
		_skip_1_8 = self.Encoder_block4(_skip_1_4) 

		# Out 1_8
		out_scale_1_8__2D = self.conv_out_scale_1_8(_skip_1_8)

		if self.out_scale=="1_8":
			out_scale_1_8__3D = self.seg_head_1_8(out_scale_1_8__2D) # [1, 20, 16, 128, 128]
			ssc_pred = out_scale_1_8__3D.permute(0, 1, 4, 3, 2) # [1, 20, 128, 128, 16]

		elif self.out_scale=="1_4":
			# Out 1_4
			out = self.deconv1_8(out_scale_1_8__2D)
			out = torch.cat((out, _skip_1_4), 1)
			out = F.relu(self.conv1_4(out))
			out_scale_1_4__2D = self.conv_out_scale_1_4(out)

			out_scale_1_4__3D = self.seg_head_1_4(out_scale_1_4__2D) # [1, 20, 16, 128, 128]
			ssc_pred = out_scale_1_4__3D.permute(0, 1, 4, 3, 2) # [1, 20, 128, 128, 16]

		elif self.out_scale=="1_2":
			# Out 1_4
			out = self.deconv1_8(out_scale_1_8__2D)
			out = torch.cat((out, _skip_1_4), 1)
			out = F.relu(self.conv1_4(out))
			out_scale_1_4__2D = self.conv_out_scale_1_4(out)

			# Out 1_2
			out = self.deconv1_4(out_scale_1_4__2D)
			out = torch.cat((out, _skip_1_2, self.deconv_1_8__1_2(out_scale_1_8__2D)), 1)
			out = F.relu(self.conv1_2(out)) # torch.Size([1, 48, 128, 128])
			out_scale_1_2__2D = self.conv_out_scale_1_2(out) # torch.Size([1, 16, 128, 128])

			out_scale_1_2__3D = self.seg_head_1_2(out_scale_1_2__2D) # [1, 20, 16, 128, 128]
			ssc_pred = out_scale_1_2__3D.permute(0, 1, 4, 3, 2) # [1, 20, 128, 128, 16]

		elif self.out_scale=="1_1":
			# Out 1_4
			out = self.deconv1_8(out_scale_1_8__2D)
			print('out.shape', out.shape)  # [1, 4, 64, 64]
			out = torch.cat((out, _skip_1_4), 1)
			out = F.relu(self.conv1_4(out))
			out_scale_1_4__2D = self.conv_out_scale_1_4(out)

			# Out 1_2
			out = self.deconv1_4(out_scale_1_4__2D)
			print('out.shape', out.shape)  # [1, 8, 128, 128]
			out = torch.cat((out, _skip_1_2, self.deconv_1_8__1_2(out_scale_1_8__2D)), 1)
			out = F.relu(self.conv1_2(out)) # torch.Size([1, 48, 128, 128])
			out_scale_1_2__2D = self.conv_out_scale_1_2(out) # torch.Size([1, 16, 128, 128])

			# Out 1_1
			out = self.deconv1_2(out_scale_1_2__2D)
			out = torch.cat((out, _skip_1_1, self.deconv_1_4__1_1(out_scale_1_4__2D), self.deconv_1_8__1_1(out_scale_1_8__2D)), 1)
			out_scale_1_1__2D = F.relu(self.conv1_1(out)) # [bs, 32, 256, 256]

			out_scale_1_1__3D = self.seg_head_1_1(out_scale_1_1__2D)
			# Take back to [W, H, D] axis order
			ssc_pred = out_scale_1_1__3D.permute(0, 1, 4, 3, 2)  # [bs, C, H, W, D] -> [bs, C, D, W, H]

		return ssc_pred

		################################### test #####################################
		# y_pred = ssc_pred.detach().cpu().numpy() # [1, 20, 200, 200, 16]
		# y_pred = np.argmax(y_pred, axis=1).astype(np.uint8) # [1, 128, 128, 16]

		# #save query proposal 
		# img_path = img_metas[0]['img_filename'] 
		# frame_id = os.path.splitext(img_path[0])[0][-6:]

		# y_pred_bin = self.pack(y_pred)
		# y_pred_bin.tofile(save_query_path)

		# result = dict()
		# y_true = target.cpu().numpy()
		# result['y_pred'] = y_pred
		# result['y_true'] = y_true

		# return result


class SegmentationHead(nn.Module):
	'''
	3D Segmentation heads to retrieve semantic segmentation at each scale.
	Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
	'''
	def __init__(self, inplanes, planes, nbr_classes, dilations_conv_list):
		super().__init__()

		# First convolution
		self.conv0 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, stride=1)

		# ASPP Block
		self.conv_list = dilations_conv_list
		self.conv1 = nn.ModuleList(
		[nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
		self.bn1 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
		self.conv2 = nn.ModuleList(
		[nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
		self.bn2 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
		self.relu = nn.ReLU(inplace=True)

		# Convolution for output
		self.conv_classes = nn.Conv3d(planes, nbr_classes, kernel_size=3, padding=1, stride=1)

	def forward(self, x_in):

		# Dimension exapension
		x_in = x_in[:, None, :, :, :]

		# Convolution to go from inplanes to planes features...
		x_in = self.relu(self.conv0(x_in))

		y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
		for i in range(1, len(self.conv_list)):
			y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
			x_in = self.relu(y + x_in)  # modified

			x_in = self.conv_classes(x_in)

		return x_in
