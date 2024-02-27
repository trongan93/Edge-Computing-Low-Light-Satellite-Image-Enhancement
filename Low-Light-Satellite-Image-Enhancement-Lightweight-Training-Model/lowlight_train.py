# Liscotech
# Chen-Kai Tsai
# 2/2/2023
# NTUST
# Trong-An Bui
# Updated in 3/20/2023

import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import Myloss
import numpy as np
from torchvision import transforms

# Suppress all warnings
import warnings
warnings.filterwarnings('ignore')

import wandb

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="low-light-EdgeCu",
    mode="disabled",
    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.02,
        "architecture": "EdgeCu",
        "dataset": "low-light",
        "epochs": 10,
    }
)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    scale_factor = config.scale_factor
    # DCE_net = model.simpleEnhanceNet(scale_factor).cuda() # proposed training model
    DCE_net = model.EnhanceNet(scale_factor).cuda()  # original DCENet

    # DCE_net.apply(weights_init)
    if config.load_pretrain == True:
        DCE_net.load_state_dict(torch.load(config.pretrain_dir))
    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)
    print(config.lowlight_images_path)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)



    L_color = Myloss.L_color(8)
    # L_color = Myloss.L_color(16)
    L_spa = Myloss.L_spa()
    L_exp = Myloss.L_exp(16)
    # L_exp = Myloss.L_exp(16,0.6)
    L_TV = Myloss.L_TV()
    #L_sa = Myloss.Sa_Loss()

    dataAmount = train_dataset.__len__()
    batchSize = config.train_batch_size
    
    
    optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    DCE_net.train()

    for epoch in range(config.num_epochs):
        completeSum = 0
        for iteration, img_lowlight in enumerate(train_loader):

            img_lowlight = img_lowlight.cuda()
            b, _, _, _ = img_lowlight.size()
            E = 0.575 # TODO: Paper with 0.6, Walter with 0.575, Try 0.5

            enhanced_image,A  = DCE_net(img_lowlight)

            # TODO: Weight for loss functions are different from the original paper
            Loss_TV = 7000*L_TV(A)
            # Loss_TV = 200*L_TV(A)			
            loss_spa = 100*torch.mean(L_spa(enhanced_image, img_lowlight))
            loss_col = torch.mean(L_color(enhanced_image))

            loss_exp = 50*torch.mean(L_exp(enhanced_image,E))
            #loss_sa =  10*torch.mean(L_sa(enhanced_image))
            
            # best_loss
            loss =  Loss_TV + loss_spa + loss_col + loss_exp
            wandb.log({"loss": loss, 'epoch': epoch})

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(DCE_net.parameters(),config.grad_clip_norm) # prevent gradient explode
            optimizer.step()

            
            completeSum += b 
            pComplete = int(completeSum / dataAmount * 100) // 2
            pUndo = int((1 - (completeSum / dataAmount)) * 100) // 2 
            
            if ((iteration+1) % config.display_iter) == 0:
                print("Epoch : "+ str(epoch + 1) +  "  [" + "-"*pComplete + ">" + " "*pUndo + "] - loss: " + str(loss.item()), "\r", end='')
                #print("Loss at iteration", iteration+1, ":", loss.item())
            # if ((iteration+1) % config.snapshot_iter) == 0:
            #     torch.save(DCE_net.state_dict(), config.snapshots_folder + "ZeroDCE_test" + str(epoch) + '.pth')
            #     torch.save(DCE_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')

        print()

        if ((epoch + 1) % 4) == 0:
            print("Saving Model " + config.snapshots_folder + "R6Epoch" + str((epoch + 1)) + '.pth' + " with Epoch " + str((epoch + 1)))
            torch.save(DCE_net.state_dict(), config.snapshots_folder + "R6Epoch" + str((epoch + 1)) + '.pth')
            print("Model Saved\n\n")
    
    # print("Saving Model")
    # torch.save(DCE_net.state_dict(), config.snapshots_folder + "R5Epoch" + str((epoch + 1)) + '.pth')
    # print("Model Saved")
        



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str, default="/mnt/d/ZeroDCEDataSet/ZeroDCE/train/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--display_iter', type=int, default=1)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--scale_factor', type=int, default=1) # TODO: check differeces if applied
    parser.add_argument('--snapshots_folder', type=str, default="./snapshots_weight_trongan93/")
    parser.add_argument('--load_pretrain', type=bool, default= False)
    # parser.add_argument('--pretrain_dir', type=str, default= "./snapshots_weight_trongan93/Epoch99.pth") # TODO: Need change the model path

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    # print arguments
    for arg in vars(config):
        print(arg, getattr(config, arg))

    train(config)
