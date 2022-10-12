from email.policy import default
from types import new_class
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
from dataset.shapenet_data_pc import ShapeNet15kPointClouds

def set_seed(opt):
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    if opt.gpu is not None and torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.manualSeed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def getGradNorm(net):
    pNorm = torch.sqrt(sum(torch.sum(p ** 2) for p in net.parameters()))
    gradNorm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in net.parameters()))
    return pNorm, gradNorm

def get_shapenet_dataset(dataroot, npoints, category):
    tr_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
        categories=[category], split='train',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        random_subsample=True)

    val_dataset = ShapeNet15kPointClouds(root_dir=dataroot,
        categories=[category], split='val',
        tr_sample_size=npoints,
        te_sample_size=npoints,
        scale=1.,
        normalize_per_shape=False,
        normalize_std_per_axis=False,
        all_points_mean=tr_dataset.all_points_mean,
        all_points_std=tr_dataset.all_points_std
    )
    return tr_dataset, val_dataset

def get_text2shape_dataset(dataroot, category):
    tr_dataset = Text2Shape(root=Path(dataroot),
        split="train",
        categories=category,
        from_shapenet_v1=True,
        from_shapenet_v2=False,
        conditional_setup=True,
        language_model="t5-11b",
        lowercase_text=False,
        max_length=60,
        scale_mode="shape_unit")

    val_dataset = Text2Shape(root=Path(dataroot),
        split="val",
        categories=category,
        from_shapenet_v1=True,
        from_shapenet_v2=False,
        conditional_setup=True,
        language_model="t5-11b",
        lowercase_text=False,
        max_length=60,
        scale_mode="shape_unit"
        )
    return tr_dataset, val_dataset
    
def get_dataloader(opt, train_dataset, val_dataset=None):

    if opt.distribution_type == 'multi':
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=opt.world_size,
            rank=opt.rank
        )
        if val_dataset is not None:
            val_samples = torch.utils.data.distributed.DistributedSampler(
                val_dataset,
                num_replicas=opt.world_size,
                rank=opt.rank
            )
        else:
            val_samples = None
    else:
        train_sampler = None
        val_samples = None
    print('WORKERS: ', int(opt.workers))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs,sampler=train_sampler,
                                                   shuffle=train_sampler is None, num_workers=int(opt.workers), drop_last=True)
                                                    # drop_last drops the last incomplete batch, if the dataset size is not divisible by the batch size

    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.bs,sampler=val_samples,
                                                   shuffle=False, num_workers=int(opt.workers), drop_last=False)
    else:
        val_dataloader = None

    return train_dataloader, val_dataloader, train_sampler, val_samples

def train(gpu, opt, output_dir, dset, noises_init):

    set_seed(opt)
    logger = setup_logging(output_dir)
    if opt.distribution_type == 'multi':
        should_diag = gpu==0
    else:
        should_diag = True
    if should_diag:
        outf_syn = os.path.join(opt.output_dir, 'syn')
        if not os.path.exists(outf_syn):
            outf_syn = os.makedirs(outf_syn)

    if opt.distribution_type == 'multi':
        if opt.dist_url == "env://" and opt.rank == -1:
            opt.rank = int(os.environ["RANK"])

        base_rank =  opt.rank * opt.ngpus_per_node
        opt.rank = base_rank + gpu
        torch.distributed.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)

        opt.bs = int(opt.bs / opt.ngpus_per_node)
        opt.workers = 0

        opt.saveIter =  int(opt.saveIter / opt.ngpus_per_node)
        opt.diagIter = int(opt.diagIter / opt.ngpus_per_node)
        opt.vizIter = int(opt.vizIter / opt.ngpus_per_node)
        
    train_dataloader, val_dataloader, train_sampler, val_sampler = get_dataloader(opt, dset, None)

    '''
    create networks
    '''
    model = Infusion(opt)

    if opt.distribution_type == 'multi':  # Multiple processes, single GPU per process
        def _transform_(m):
            return nn.parallel.DistributedDataParallel(
                m, device_ids=[gpu], output_device=gpu)

        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        model.multi_gpu_wrapper(_transform_)


    elif opt.distribution_type == 'single':
        def _transform_(m):
            return nn.parallel.DataParallel(m)
        model = model.cuda()
        model.multi_gpu_wrapper(_transform_)

    elif gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    else:
        raise ValueError('distribution_type = multi | single | None')

    if should_diag:
        logger.info(opt)

    optimizer= torch.optim.Adam(model.pvd.parameters(), lr=opt.lr, weight_decay=opt.decay, betas=(opt.beta1, 0.999))

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, opt.lr_gamma)

    if opt.model != '':
        ckpt = torch.load(opt.model)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])

    if opt.model != '':
        start_epoch = torch.load(opt.model)['epoch'] + 1
    else:
        start_epoch = 0

    def new_x_chain(x, num_chain):
        return torch.randn(num_chain, *x.shape[1:], device=x.device)
    
    for param_map in model.mapping_net.parameters():
        param_map.requires_grad=True
    
    for param_pvd in model.pvd.parameters():
        param_pvd.requires_grad=True

    for epoch in range(start_epoch, opt.n_epochs):

        if opt.distribution_type == 'multi':
            train_sampler.set_epoch(epoch)

        lr_scheduler.step(epoch)

        for i, data in enumerate(train_dataloader):
            if opt.train_ds == 'shapenet':
                x = data['train_points'].transpose(1,2)                 # TODO: check if pointclouds have the same size for both ShapeNet and Text2Shape 
                noises_batch = noises_init[data['idx']].transpose(1,2)  # TODO: check if idx are the same for both ShapeNet and Text2Shape
                #text_embed = data["text_embed"]
                #mask = data["key_pad_mask"]
            elif opt.train_ds == 'text2shape':
                x = data['pointcloud'].transpose(1,2)                   # TODO: check if pointclouds have the same size for both ShapeNet and Text2Shape          
                noises_batch = noises_init[data['idx']].transpose(1,2)  # TODO: check if idx are the same for both ShapeNet and Text2Shape
                text_embed = data["text_embed"]
                mask = data["key_pad_mask"]
                text = data["text"]

            '''
            train diffusion
            '''
            if opt.distribution_type == 'multi' or (opt.distribution_type is None and gpu is not None):
                x = x.cuda()
                noises_batch = noises_batch.cuda()
                text_embed = text_embed.cuda()
                mask = mask.cuda()
            elif opt.distribution_type == 'single':
                x = x.cuda()
                noises_batch = noises_batch.cuda()
                text_embed = text_embed.cuda()
                mask = mask.cuda()

            model.mapping_net.train()
            model.pvd.train()

            # check input values
            if torch.isnan(x).any():
                    print(f'NaN values in x')
            if torch.isnan(text_embed).any():
                    print(f'NaN values in text embed')
            if torch.isnan(mask).any():
                    print(f'NaN values in mask')
            if torch.isnan(noises_batch).any():
                    print(f'NaN values in noises')
            

            loss = model.get_loss(x, noises_batch, text_embed, mask).mean()
            
            optimizer.zero_grad()
            loss.backward()
            netpNorm_pvd, netgradNorm_pvd = getGradNorm(model.pvd)
            netpNorm_mapnet, netgradNorm_mapnet = getGradNorm(model.mapping_net)

            for p in model.mapping_net.parameters():
                param = p
            
            if opt.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip, error_if_nonfinite=True)

            optimizer.step()
            
            print('CLS TOKEN: ', model.mapping_net.cls_token.data)
            #print('PROJ WEIGHTS: ', model.mapping_net.proj.weight.data)
            #print('PROJ BIAS: ', model.mapping_net.proj.bias.data)

            if i==0:
                old_cls_token = model.mapping_net.cls_token.data
                #old_proj_weight = model.mapping_net.proj.weight.data
                #old_proj_bias = model.mapping_net.proj.bias.data
            else:
                new_cls_token = model.mapping_net.cls_token.data
                #new_proj_weight = model.mapping_net.proj.weight.data
                #new_proj_bias = model.mapping_net.proj.bias.data
                
                if torch.equal(new_cls_token, old_cls_token):
                    print('CLS TOKEN not updated :( ')
                else:
                    print('CLS TOKEN updated correctly :) ')
                #if torch.equal(new_proj_weight, old_proj_weight):
                #    print('PROJ WEIGHT not updated :( ')
                #else:
                #    print('PROJ WEIGHT updated correctly :) ')
                #if torch.equal(new_proj_bias, old_proj_bias):
                #    print('PROJ BIAS not updated :( ')
                #else:
                #    print('PROJ BIAS updated correctly :) ')
                
                old_cls_token = new_cls_token
                #old_proj_weight = new_proj_weight
                #old_proj_bias = new_proj_bias

            if i % opt.print_freq == 0 and should_diag:

                logger.info('[{:>3d}/{:>3d}][{:>3d}/{:>3d}]    loss: {:>10.4f},    '
                             'netpNorm_PVD: {:>10.2f},   netgradNorm_PVD: {:>10.2f}     '
                             .format(
                        epoch, opt.n_epochs, i, len(train_dataloader),loss.item(),
                    netpNorm_pvd, netgradNorm_pvd,
                        ))
                
                logger.info('[{:>3d}/{:>3d}][{:>3d}/{:>3d}]    loss: {:>10.4f},    '
                             'netpNorm_MAPNET: {:>10.2f},   netgradNorm_MAPNET: {:>10.2f}     '
                             .format(
                        epoch, opt.n_epochs, i, len(train_dataloader),loss.item(),
                    netpNorm_mapnet, netgradNorm_mapnet,
                        ))
                
        
        if (epoch + 1) % opt.vizIter == 0 and should_diag:
            
            logger.info('Generating clouds for visualization on training set...')

            model.mapping_net.eval()
            model.pvd.eval()
            with torch.no_grad():

                x_gen_eval = model.get_clouds(text_embed, mask, x)
                x_gen_list = model.get_cloud_traj(text_embed, mask, x)
                x_gen_all = torch.cat(x_gen_list, dim=0)
                
                gen_stats = [x_gen_eval.mean(), x_gen_eval.std()]

                gen_eval_range = [x_gen_eval.min().item(), x_gen_eval.max().item()]

                logger.info('      [{:>3d}/{:>3d}]  '
                            'eval_gen_range: [{:>10.4f}, {:>10.4f}]     '
                            'eval_gen_stats: [mean={:>10.4f}, std={:>10.4f}]      '
                    .format(
                    epoch, opt.n_epochs,
                    *gen_eval_range, *gen_stats,
                ))

            visualize_pointcloud_batch('%s/epoch_%03d_samples_eval.png' % (outf_syn, epoch),
                                    x_gen_eval.transpose(1, 2), None, None,
                                    None)

            visualize_pointcloud_batch('%s/epoch_%03d_samples_eval_all.png' % (outf_syn, epoch),
                                    x_gen_all.transpose(1, 2), None,
                                    None,
                                    None)

            visualize_pointcloud_batch('%s/epoch_%03d_x.png' % (outf_syn, epoch), x.transpose(1, 2), None,
                                    None,
                                    None)            

        if (epoch + 1) % opt.diagIter == 0 and should_diag:

            logger.info('Computing KL for diagnosis on training set...')

            model.mapping_net.eval()
            model.pvd.eval()
            with torch.no_grad():
                condition = model.mapping_net(text_embed, mask)
                x_range = [x.min().item(), x.max().item()]
                kl_stats = model.pvd.all_kl(x, condition)

            logger.info('      [{:>3d}/{:>3d}]    '
                         'x_range: [{:>10.4f}, {:>10.4f}],   '
                         'total_bpd_b: {:>10.4f},    '
                         'terms_bpd: {:>10.4f},  '
                         'prior_bpd_b: {:>10.4f}    '
                         'mse_bt: {:>10.4f}  '
                .format(
                epoch, opt.n_epochs,
                *x_range,
                kl_stats['total_bpd_b'].item(),
                kl_stats['terms_bpd'].item(), kl_stats['prior_bpd_b'].item(), kl_stats['mse_bt'].item()
            ))

        if (epoch + 1) % opt.saveIter == 0:
            
            logger.info('Saving checkpoint...')
            if should_diag:
                save_dict = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()
                }
                torch.save(save_dict, '%s/epoch_%d.pth' % (output_dir, epoch))

            if opt.distribution_type == 'multi':
                torch.distributed.barrier()
                map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
                model.load_state_dict(
                    torch.load('%s/epoch_%d.pth' % (output_dir, epoch), map_location=map_location)['model_state'])

    torch.distributed.destroy_process_group()

def main():
    opt = parse_args()
    if opt.category == 'airplane':
        opt.beta_start = 1e-5
        opt.beta_end = 0.008
        opt.schedule_type = 'warm0.1'

    exp_id = os.path.splitext(os.path.basename(__file__))[0]
    dir_id = os.path.dirname(__file__)

    # get dataset
    if opt.train_ds == 'shapenet':
        train_dataset, val_dataset = get_shapenet_dataset(opt.sn_dataroot, opt.npoints, opt.category)
        noises_init = torch.randn(len(train_dataset), opt.npoints, opt.nc)
    elif opt.train_ds == 'text2shape':
        train_dataset, val_dataset = get_text2shape_dataset(opt.t2s_dataroot, opt.category)
        noises_init = torch.randn(len(train_dataset), opt.npoints, opt.nc)
    else:
        raise Exception('train_ds not specified correctly. Got ', opt.train_ds)

    print('Training dataset size: ', len(train_dataset))
    print('Val dataset size: ', len(val_dataset))

    if opt.dist_url == "env://" and opt.world_size == -1:
        opt.world_size = int(os.environ["WORLD_SIZE"])

    if opt.distribution_type == 'multi':
        opt.ngpus_per_node = torch.cuda.device_count()
        opt.world_size = opt.ngpus_per_node * opt.world_size
        torch.multiprocessing.spawn(train, nprocs=opt.ngpus_per_node, args=(opt, opt.output_dir, noises_init))
    else:
        train(opt.gpu, opt, opt.output_dir, train_dataset, noises_init)

def parse_args():

    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument('--train_ds', default='text2shape', choices=['shapenet', 'text2shape'], help='dataset to use for training')
    parser.add_argument('--sn_dataroot', default='../PVD/data/ShapeNetCore.v2.PC15k/', help="dataroot of ShapeNet")
    parser.add_argument('--t2s_dataroot', default='/media/data2/aamaduzzi/datasets/Text2Shape/', help="dataroot of ShapeNet")
    parser.add_argument('--category', default='chair')
    parser.add_argument('--bs', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, default=16, help='workers')
    parser.add_argument('--n_epochs', type=int, default=10000, help='number of epochs to train for')
    parser.add_argument('--nc', default=3)
    parser.add_argument('--npoints', default=2048)
    parser.add_argument('--output_dir', type=str, default="./exps/exp_7", help='directory for experiments logging',)


    # MappingNet
    parser.add_argument('--dmodel', type=int, help='depth of input', default=1024)
    parser.add_argument('--nhead', type=int, help='n of multi-attention heads', default=8)
    parser.add_argument('--nlayers', type=int, help='n of encoder layers', default=6)
    parser.add_argument('--out_flatshape', type=int, help='flattened output shape', default=2048*3)

    # PVD
    # noise schedule
    parser.add_argument('--beta_start', default=0.0001)
    parser.add_argument('--beta_end', default=0.02)
    parser.add_argument('--schedule_type', default='linear')
    parser.add_argument('--time_num', default=1000)

    #loss params
    parser.add_argument('--attention', default=True)
    parser.add_argument('--dropout', default=0.1)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--loss_type', default='mse')
    parser.add_argument('--model_mean_type', default='eps')
    parser.add_argument('--model_var_type', default='fixedsmall')

    # learning rate params
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate for E, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--decay', type=float, default=0, help='weight decay for EBM')
    parser.add_argument('--grad_clip', type=float, default=None, help='weight decay for EBM')
    parser.add_argument('--lr_gamma', type=float, default=0.998, help='lr decay for EBM')

    # path to checkpt of trained model
    parser.add_argument('--model', default='', help="path to model (to continue training)")


    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed nodes.')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:9991', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--distribution_type', default='single', choices=['multi', 'single', None],
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all available GPUs.')

    # evaluation params
    parser.add_argument('--saveIter', default=12, help='unit: epoch')  
    parser.add_argument('--diagIter', default=2, help='unit: epoch')
    parser.add_argument('--vizIter', default=2, help='unit: epoch')
    parser.add_argument('--print_freq', default=100, help='unit: iter')
    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')


    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    main()