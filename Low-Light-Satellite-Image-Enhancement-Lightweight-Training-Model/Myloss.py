import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16

import numpy as np


class L_color(nn.Module):

    def __init__(self, div):
        super(L_color, self).__init__()
        self.div = div

    def forward(self, x):
        b, c, h, w = x.shape
        # div = 8
        blockH = h // self.div
        blockW = w // self.div
        D = torch.FloatTensor([0]).cuda()
        for bH in range(self.div):
            for bW in range(self.div):
                x_block = x[:, :, bH * blockH:(bH + 1) * blockH, bW * blockW:(bW + 1) * blockW]
                x_block_reshape = torch.reshape(x_block, (b, c, blockH * blockW))
                mean_rgb = torch.mean(x_block, [2, 3], keepdim=True)
                max_rgb, _ = torch.max(x_block_reshape, 2, keepdim=True)
                min_rgb, _ = torch.min(x_block_reshape, 2, keepdim=True)
                meanR, meanG, meanB = torch.split(mean_rgb, 1, dim=1)
                maxR, maxG, maxB = torch.split(max_rgb, 1, dim=1)
                minR, minG, minB = torch.split(min_rgb, 1, dim=1)
                for ibatch in range(b):
                    if minR[ibatch] > maxG[ibatch] or maxR[ibatch] < minG[ibatch]:
                        D += 0
                    else:
                        D += torch.pow(meanR[ibatch] - meanG[ibatch], 2).squeeze()

                    if minG[ibatch] > maxB[ibatch] or maxG[ibatch] < minB[ibatch]:
                        D += 0
                    else:
                        D += torch.pow(meanG[ibatch] - meanB[ibatch], 2).squeeze()

                    if minR[ibatch] > maxB[ibatch] or maxR[ibatch] < minB[ibatch]:
                        D += 0
                    else:
                        D += torch.pow(meanR[ibatch] - meanB[ibatch], 2).squeeze()

        k = D / (self.div * self.div)
        return k


class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        b, c, h, w = org.shape

        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        weight_diff = torch.max(
            torch.FloatTensor([1]).cuda() + 10000 * torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),
                                                              torch.FloatTensor([0]).cuda()),
            torch.FloatTensor([0.5]).cuda())
        ## max(1+10000*min(org_pool-0.3, 0), 0.5)
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()), enhance_pool - org_pool)
        # unknown application

        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E = (D_left + D_right + D_up + D_down)
        return E


class L_exp(nn.Module):
    def __init__(self, patch_size):
        super(L_exp, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)

    def forward(self, x, mean_val):
        b, c, h, w = x.shape
        x = torch.mean(x, 1,
                       keepdim=True)  # TODO: Apply grey level formula => Grey level = 0.299 * red component + 0.587 * green component + 0.114 * blue component
        mean = self.pool(x)
        d = torch.mean(torch.pow(mean - torch.FloatTensor([mean_val]).cuda(), 2))
        return d


# Solution 2
class L_exp_gray(nn.Module):
    def __init__(self, patch_size):
        super(L_exp_gray, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)

    def forward(self, x, mean_val):
        b, c, h, w = x.shape
        x = (((x[:, 0, :, :] * 256 * 0.299) + (x[:, 1, :, :] * 256 * 0.587) + (x[:, 1, :, :] * 256 * 0.114)) / 256)
        # x2 = torch.mean(x,1,keepdim=True)
        # mean = (0.7 * self.pool(x1)) + (0.3 * self.pool(x2))
        mean = self.pool(x)
        d = torch.mean(torch.pow(mean - torch.FloatTensor([mean_val]).cuda(), 2))
        return d


class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

class L_KDL(nn.Module):
    def __init__(self, reduction):
        super(L_KDL, self).__init__()
        self.reduction = reduction
    def forward(self,target,predict):
        kl_loss = nn.KLDivLoss(reduction=self.reduction)
        # input = F.log_softmax(RGB[:, :3, :, :],dim=1)
        # target = F.softmax(x, dim=1)

        r_predict, g_predict, b_predict = predict[:,0,:,:], predict[:,1,:,:], predict[:,2,:,:]
        r_target, g_target, b_target = target[:, 0, :, :], target[:, 1, :, :], target[:, 2, :, :]

        # r_predict, g_predict, b_predict = F.log_softmax(predict[:, 0, :, :]), F.log_softmax(predict[:, 1, :, :]), F.log_softmax(predict[:, 2, :, :])
        # r_target, g_target, b_target = F.log_softmax(target[:, 0, :, :]), F.log_softmax(target[:, 1, :, :]), F.log_softmax(target[:, 2, :, :])

        # l_kl_rb = kl_loss(r_target - b_target, r_predict - b_predict)
        # l_kl_rg = kl_loss(r_target - g_target, r_predict - g_predict)
        # l_kl_gb = kl_loss(g_target - b_target, g_predict - b_predict)

        l_kl_rb = kl_loss(F.softmax(r_target - b_target), F.softmax(r_predict - b_predict))
        l_kl_rg = kl_loss(F.softmax(r_target - g_target), F.softmax(r_predict - g_predict,))
        l_kl_gb = kl_loss(F.softmax(g_target - b_target), F.softmax(g_predict - b_predict))

        l_kl = l_kl_rb + l_kl_rg + l_kl_gb

        return l_kl_rb, l_kl_rg, l_kl_gb, l_kl
