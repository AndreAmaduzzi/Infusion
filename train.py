from email.policy import default
from typing import Iterator
import argparse
import numpy as np
from collections import OrderedDict
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from utils.misc import *
from models import *
from models.infusion import Infusion
from models.mapping_net import MappingNet
from dataset.text2shape_dataset import Text2Shape
from dataset.utils import *
from pycarus.metrics.chamfer_distance import chamfer
from pathlib import Path
from tqdm import tqdm
from torchsummary import summary
from datetime import datetime
import os

# Arguments
parser = argparse.ArgumentParser()

# MappingNet
parser.add_argument('--dmodel', type=int, help='depth of input', default=1024)
parser.add_argument('--nhead', type=int, help='n of multi-attention heads', default=8)
parser.add_argument('--nlayers', type=int, help='n of encoder layers', default=6)
parser.add_argument('--out_flatshape', type=int, help='flattened output shape', default=2048*3)

# PVD
parser.add_argument('--ckpt_pvd', type=str, help='ckpt of PVD model', default='../PVD/ckpt/generation/chair_1799.pth')
parser.add_argument('--schedule_type', type=str, help='beta scheduling', default='linear', choices=['linear', 'warm0.1', 'warm0.2', 'warm0.5'])
parser.add_argument('--beta_start', help='initial beta value', default=0.0001)
parser.add_argument('--beta_end', help='final beta value', default=0.02)
parser.add_argument('--time_num', help='n of steps of the diffusion models', default=1000)
parser.add_argument('--loss_type', default='mse')
parser.add_argument('--model_mean_type', default='eps')
parser.add_argument('--model_var_type', default='fixedsmall')


# PVCNN
parser.add_argument('--nc', default=3)
parser.add_argument('--attention', default=True)
parser.add_argument('--dropout', default=0.1)
parser.add_argument('--embed_dim', type=int, default=64)


# Training
parser.add_argument('--lr', type=int, help='learning rate', default=1e-4)
parser.add_argument('--seed', type=int, default=2022)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--exp_dir', type=str, help='directory for experiments logging', default="./exps/exp_1")
parser.add_argument('--val_freq', type=int, help='validation frequency', default=50)
parser.add_argument('--max_grad_norm', type=float, default=10)


args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

def _transform_(m):
    return nn.parallel.DataParallel(m)

def train(args, model, train_iter: Iterator, val_iter: Iterator, max_iters: int, optimizer, writer, logger):
    for i in tqdm(range(max_iters)):
        start = datetime.now()
        train_batch= next(train_iter)
        x = train_batch["text_embed"]
        x = torch.transpose(x, 1, 0)
        mask = train_batch["key_pad_mask"]
        gt_shape = train_batch["pointcloud"].to(args.device)
        
        optimizer.zero_grad()
        model.mapping_net.train()
        model.pvd.train()
        
        pred_shape = model(x=x, src_key_padding_mask = mask)
        gt_shape = torch.permute(gt_shape, (0, 2, 1))
        logger.info('generated shape of size: %s', pred_shape.shape)
        logger.info('reference shape: %s', gt_shape.shape)
        torch.save(pred_shape, os.path.join(args.exp_dir, f'pred_shape_{i}.pt'))
        torch.save(gt_shape, os.path.join(args.exp_dir, f'gt_shape_{i}.pt'))

        # compute loss between pred_shape and gt_shape
        chamfer_dist = chamfer(pred_shape, gt_shape) # chamfer_dist = (chamfer, accuracy, completeness, indices_pred_gt, indices_gt_pred)
        loss = chamfer_dist[0].mean()        

        # backprop
        optimizer.zero_grad()

        #for param in model.parameters():
        #    if param.requires_grad == False:
        #        print('req grad False: ', param)

        #loss.requires_grad=True
        #loss.backward()
        #optimizer.step()

        orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
        writer.add_scalar("Loss/train", loss, i)
        writer.add_scalar('Grad_norm/train', orig_grad_norm, i)
        
        logger.info('[Train] Iter %04d | Loss %.6f | Grad %.4f | LR %.6f'  % (
                    i, loss.item(), orig_grad_norm, optimizer.param_groups[0]['lr']
                    ))

        end = datetime.now()

        # Validation
        if (i+1) % args.val_freq:
            model.mapping_net.eval()
            model.pvd.eval()
            with torch.no_grad():
                val_batch = next(val_iter)
                x = val_batch["text_embed"]
                x = torch.transpose(x, 1, 0)
                mask = val_batch["key_pad_mask"]
                gt_shape = val_batch["pointcloud"].to(args.device)        
                pred_shape = model(x=x, src_key_padding_mask = mask)
                gt_shape = torch.permute(gt_shape, (0, 2, 1))
                # compute loss between pred_shape and gt_shape
                chamfer_dist = chamfer(pred_shape, gt_shape) # chamfer_dist = (chamfer, accuracy, completeness, indices_pred_gt, indices_gt_pred)
                loss = chamfer_dist[0].mean()
                writer.add_scalar("Loss/val", loss, i)
                writer.add_mesh('val_gen/pointcloud', pred_shape[:3], global_step=i)
                writer.add_mesh('val_ref/pointcloud', gt_shape[:3], global_step=i)

                logger.info('[Val] Iter %04d | Loss %.6f | Grad %.4f | LR %.6f'  % (
                            i, loss.item(), orig_grad_norm, optimizer.param_groups[0]['lr']
                            ))
                
                writer.flush()


        logger.info('Time for one iteration: %d', (end-start).total_seconds())
        writer.flush()


if __name__=="__main__":
    args = parser.parse_args()
    
    # Prepare Tensorboard logging
    writer = SummaryWriter(args.exp_dir)
    logger = setup_logging(args.exp_dir)


    # Load datasets
    ds_path = Path("/media/data2/aamaduzzi/datasets/Text2Shape/")
    train_dset = Text2Shape(root=ds_path,
                        split="test",
                        categories="chair",
                        from_shapenet_v1=True,
                        from_shapenet_v2=False,
                        conditional_setup=True,
                        language_model="t5-11b",
                        lowercase_text=False,
                        max_length=60,
                        scale_mode="shape_bbox")
    
    val_dset = Text2Shape(root=ds_path,
                        split="test",
                        categories="chair",
                        from_shapenet_v1=True,
                        from_shapenet_v2=False,
                        conditional_setup=True,
                        language_model="t5-11b",
                        lowercase_text=False,
                        max_length=60,
                        scale_mode="shape_bbox")

    train_iter = get_data_iterator(DataLoader(
        train_dset,
        batch_size=5,
        num_workers=0,
        shuffle=True,
    ))

    val_iter = get_data_iterator(DataLoader(
        val_dset,
        batch_size=5,
        num_workers=0,
        shuffle=True,
    ))

    # instantiate model
    model = Infusion(args)
    model = model.cuda()

    logging.info('Loaded pre-trained weights of PVD')

    model.multi_gpu_wrapper(_transform_) # self.pvd.model = nn.parallel.DataParallel(self.pvd.model) where self.pvd.model is PVCNN2

    # Load pre-trained PVD
    ckpt_pvd = torch.load(args.ckpt_pvd)
    model.pvd.load_state_dict(ckpt_pvd['model_state'])

    optimizer = torch.optim.Adam(model.mapping_net.parameters(), lr=args.lr)

    for params_pvd in model.pvd.parameters():
        params_pvd.requires_grad=False

    train(args, model=model, train_iter=train_iter, val_iter=val_iter, max_iters=100, optimizer=optimizer, writer=writer, logger=logger)
    writer.flush()