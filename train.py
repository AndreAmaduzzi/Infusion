import copy
from email.policy import default, strict
from types import new_class
from typing import Iterator
import argparse
import numpy as np
from collections import OrderedDict
import random
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from utils.misc import *
from utils.evaluation import *
from models import *
from models.infusion import Infusion
from dataset.text2shape_dataset import Text2Shape, Text2Shape_subset_mid
from dataset.utils import *
from pycarus.metrics.chamfer_distance import chamfer
from pathlib import Path
from tqdm import tqdm
from torchinfo import summary
from datetime import datetime
import os
import math
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

def set_new_seed(opt):
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    np.random.seed(opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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
        chatgpt_prompts = True,
        split="train",
        categories=category,
        from_shapenet_v1=True,
        from_shapenet_v2=False,
        conditional_setup=True,
        language_model="t5-11b",
        lowercase_text=True,
        max_length=77,         
        padding=False,
        scale_mode="global_unit")    # global unit
    
    val_dataset = Text2Shape(root=Path(dataroot),
        chatgpt_prompts=True,
        split="val",
        categories=category,
        from_shapenet_v1=True,
        from_shapenet_v2=False,
        conditional_setup=True,
        language_model="t5-11b",
        lowercase_text=True,
        max_length=77,         
        padding=False,
        scale_mode="global_unit"     # global unit
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

def train(gpu, opt, train_dset, val_dset, noises_init):

    set_new_seed(opt)
    logger = setup_logging(opt.output_dir)
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

        opt.saveEpoch =  int(opt.saveEpoch / opt.ngpus_per_node)
        opt.diagEpoch = int(opt.diagEpoch / opt.ngpus_per_node)
        opt.vizEpoch = int(opt.vizEpoch / opt.ngpus_per_node)
    
    writer = torch.utils.tensorboard.SummaryWriter(opt.output_dir)
        
    train_dataloader, val_dataloader, train_sampler, val_sampler = get_dataloader(opt, train_dset, val_dset)

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
            return nn.parallel.DataParallel(m, device_ids=[0,1])
        model = model.cuda()
        model.multi_gpu_wrapper(_transform_)
    elif gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    else:
        raise ValueError('distribution_type = multi | single | None')

    if should_diag:
        logger.info(opt)
    
    optimizer= torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.decay, betas=(opt.beta1, 0.999))

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, opt.lr_gamma)

    if opt.model != '':
        logger.info('Loading conditional PVD from last checkpoint, to resume training...')
        ckpt = torch.load(opt.model)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = torch.load(opt.model)['epoch'] + 1
    else:
        if opt.pvd_model!='':
            logger.info('Loading unconditional pre-trained PVD')
            ckpt = torch.load(opt.pvd_model)
            weights_dict = prepare_pvd_weights(pvd_ckpt = ckpt)
            missing_keys, unexp_keys = model.load_state_dict(weights_dict, strict=False)
            print('missing keys: ', len(missing_keys))
            print('unexp keys: ', len(unexp_keys))
            assert len(unexp_keys) == 0             # we must have many missing keys, but ZERO unexp keys
            start_epoch = 0
        start_epoch = 0


    scaler = torch.cuda.amp.GradScaler()    # scaler for mixed precision training

    n_iters=0
    for epoch in range(start_epoch, opt.n_epochs):
        
        if opt.distribution_type == 'multi':
            train_sampler.set_epoch(epoch)
        torch.backends.cudnn.deterministic = False  # if True, I would get always the same noise when during training
        '''
        if (epoch+1) % opt.compEpoch == 0 and should_diag:
            logger.info('Computing chamfer distance between clouds from Conditional and Unconditional PVD, given the same seed')
            
            chamfer_dist, mean_chamfer = chamfer_cond_uncond(model, opt.bs, val_dataloader, epoch, opt.output_dir)
            
            logger.info('Conditional vs Unconditional mean CD: %.6f' %mean_chamfer)
        '''
        
        for i, data in enumerate(train_dataloader):
            start = datetime.now()
            n_iters += 1
            if opt.train_ds == 'shapenet':
                x = data['train_points'].transpose(1,2)                 
                noises_batch = noises_init[data['idx']].transpose(1,2)  # TODO: check this noise
                #text_embed = data["text_embed"]
            elif opt.train_ds == 'text2shape':
                x = data['pointcloud'].transpose(1,2)                             
                noises_batch = noises_init[data['idx']].transpose(1,2)  # TODO: check this noise
                text_embed = data["text_embed"]
                text = data["text"]         

            '''
            train diffusion
            '''

            if opt.distribution_type == 'multi' or (opt.distribution_type is None and gpu is not None):
                x = x.cuda()
                noises_batch = noises_batch.cuda()
                text_embed = text_embed.cuda()
            elif opt.distribution_type == 'single':
                x = x.cuda()
                noises_batch = noises_batch.cuda()
                text_embed = text_embed.cuda()
            
            if opt.pvd_model != '':
                model.pvd.eval()

            # check input values
            if torch.isnan(x).any():
                    print(f'NaN values in x')
            if torch.isnan(text_embed).any():
                    print(f'NaN values in text embed')
            if torch.isnan(noises_batch).any():
                    print(f'NaN values in noises')

            #print('is the text embed free from all zeros rows? ', torch.count_nonzero(sum_text_embed)==sum_text_embed.shape[0])
            
            text_embed = maxlen_padding(text_embed) # truncate or pad all text embeds to longest sequence in batch

            model.pvd.train()
        
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss, t = model.get_loss(x, noises_batch, text_embed, text, epoch)
                loss = loss.mean()
                assert loss.dtype is torch.float32
            
            optimizer.zero_grad()       
            scaler.scale(loss).backward()   # mixed precision training: loss is scaled and gradients are computed on scaled loss
            #loss.backward()

            if opt.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip, error_if_nonfinite=True)

            scaler.step(optimizer)          # mixed precision training: gradients are UNSCALED and optimizer step is done
            #optimizer.step()
 
            scaler.update()                 

            netpNorm, netgradNorm = getGradNorm(model)  # extracting gradients of params, to plot them

            step_loss = datetime.now()
            #print('time for optimizer step: ', (step_loss-after_bw).total_seconds())

            writer.add_scalar('train/loss', loss, n_iters)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], n_iters)
            writer.add_scalar('train/grad', netgradNorm, n_iters)

            if opt.bs==1: # if we have only 1 cloud in the batch, we plot its t value
                writer.add_scalar('train/t', t, n_iters) 
            
            writer.flush()
            '''
            # check model params update. TODO: check if ALL parameters of the model are updated correctly
            
            param_name = 'pvd.model.sa_layers.0.0.voxel_layers.7.q.weight'   # name of the param to check

            if i==0:
                old_param_val = copy.deepcopy(model.state_dict()[param_name])
            else:
                new_param_val = copy.deepcopy(model.state_dict()[param_name])
                        
                if torch.equal(old_param_val, new_param_val):
                    print(f'parameter {param_name} not updated :( ')
                else:
                    print(f'parameter {param_name} updated correctly :) ')
                    old_param_val = new_param_val
            
            # check if any parameter has grad=Nan
            for idx, p in enumerate(model.parameters()):
                param = p
                if torch.isnan(param.grad).any():
                    logger.info('Nan gradient in param')  # TODO: add name of the parameter
            '''

            if i % opt.print_freq == 0 and should_diag:
                logger.info('[{:>3d}/{:>3d}][{:>3d}/{:>3d}]     it: {:>3d}     loss: {:>10.4f},     lr: {:>10.8f}    '
                             'netpNorm_PVD: {:>10.2f},   netgradNorm_PVD: {:>10.2f}'
                             .format(
                        epoch, opt.n_epochs, i, len(train_dataloader), n_iters, loss.item(), optimizer.param_groups[0]['lr'],
                    netpNorm, netgradNorm,
                        ))
            
            end = datetime.now()
            #logger.info('time for a whole iteration: %.6f' % (end-start).total_seconds())
        
        '''
        if (epoch+1) % opt.diagEpoch == 0 and should_diag:
            logger.info('Computing KL for diagnosis on training set...')

            model.pvd.eval()
            with torch.no_grad():
                x_range = [x.min().item(), x.max().item()]
                kl_stats = model.pvd.all_kl(x, text_embed)

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
        '''
        if (epoch+1) % opt.saveEpoch == 0:
            model.pvd.eval()
            logger.info('Saving checkpoint...')
            if should_diag:
                save_dict = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()
                }
                torch.save(save_dict, '%s/epoch_%d.pth' % (opt.output_dir, epoch))

            if opt.distribution_type == 'multi':
                torch.distributed.barrier()
                map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
                model.load_state_dict(
                    torch.load('%s/epoch_%d.pth' % (opt.output_dir, epoch), map_location=map_location)['model_state'])

        if (epoch+1) % opt.vizEpoch == 0 and should_diag:
                model.pvd.eval()
                logger.info('Generating visualization grids...')

                model.pvd.eval()

                viz_folder = os.path.join(opt.output_dir, 'syn')

                visualize_shape_grids(model, train_batch=data, val_dl=val_dataloader, output_dir=viz_folder, epoch=epoch, logger=logger)

        if (epoch+1) % opt.valEpoch == 0 and should_diag:
                logger.info('Running validation...')
                # build reference histogram for validation set with first val_size elements (for example, 1000)
                ref_hist = dict()  
                if opt.val_size is None:
                    opt.val_size = len(val_dset)
                for i in range(0, opt.val_size):
                    if val_dset[i]["model_id"] in ref_hist.keys():
                        ref_hist[val_dset[i]["model_id"]] += 1
                    else:
                        ref_hist[val_dset[i]["model_id"]] = 1

                model.pvd.eval()

                val_folder = os.path.join(opt.output_dir, 'validation')
                val_results, mean_chamfer = run_validation(model, ref_hist, val_folder, epoch, opt.val_size, opt.bs, val_dataloader)

                # Display metrics on Tensorboard
                # CD related metrics
                writer.add_scalar('val/Coverage_CD', val_results['lgan_cov-CD'], n_iters)
                writer.add_scalar('val/MMD_CD', val_results['lgan_mmd-CD'], n_iters)
                writer.add_scalar('val/Mean_Chamfer', mean_chamfer, n_iters)
                #writer.add_scalar('val/1NN_CD', results['1-NN-CD-acc'], n_iters)
                # JSD
                writer.add_scalar('val/JSD', val_results['jsd'], n_iters)
                writer.flush()

                logger.info('[Val] Coverage  | CD %.6f | EMD n/a' % (val_results['lgan_cov-CD']))
                logger.info('[Val] MinMatDis | CD %.6f | EMD n/a' % (val_results['lgan_mmd-CD']))
                logger.info('[Val] JsnShnDis | %.6f ' % (val_results['jsd']))
                logger.info('[Val] Mean Chamfer Distance between Ref and Gen: %.6f ' % (mean_chamfer))

        # comment this line below if you want to keep a CONSTANT LEARNING RATE
        #lr_scheduler.step() # TODO: change position of this call => Exponential Scheduler has to be called at every epoch. 

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
        train_dataset, val_dataset = get_shapenet_dataset(opt.sn_dataroot, opt.npoints, opt.category, opt.val_size)
        noises_init = torch.randn(len(train_dataset), opt.npoints, opt.nc)
    elif opt.train_ds == 'text2shape':
        if opt.pvd_model != '':
            opt.category = "chair"
        train_dataset, val_dataset = get_text2shape_dataset(opt.t2s_dataroot, opt.category, opt.val_size)
        noises_init = torch.randn(len(train_dataset), opt.npoints, opt.nc)
    else:
        raise Exception('train_ds not specified correctly. Got ', opt.train_ds)

    print('Training dataset size: ', len(train_dataset))
    print('Val dataset size: ', len(val_dataset))

    if opt.dist_url == "env://" and opt.world_size == -1:
        opt.world_size = int(os.environ["WORLD_SIZE"])

    if opt.distribution_type == 'multi':
        opt.ngpus_per_node = torch.cuda.device_count()
        #opt.ngpus_per_node = 2  # n of currently available GPUs
        opt.world_size = opt.ngpus_per_node * opt.world_size
        print('sharing strategies: ', torch.multiprocessing.get_all_sharing_strategies())
        torch.multiprocessing.spawn(train, args=(opt, train_dataset, val_dataset, noises_init), nprocs=opt.ngpus_per_node)
    else:
        train(opt.gpu, opt, train_dataset, val_dataset, noises_init)

def parse_args():

    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument('--train_ds', default='text2shape', choices=['shapenet', 'text2shape'], help='dataset to use for training')
    parser.add_argument('--sn_dataroot', default='../PVD/data/ShapeNetCore.v2.PC15k/', help="dataroot of ShapeNet")
    parser.add_argument('--t2s_dataroot', default='/media/data2/aamaduzzi/datasets/Text2Shape/', help="dataroot of ShapeNet")
    parser.add_argument('--category', default='all')
    parser.add_argument('--bs', type=int, default=40, help='input batch size')
    parser.add_argument('--workers', type=int, default=0, help='workers')
    parser.add_argument('--n_epochs', type=int, default=111000, help='number of epochs to train for')
    parser.add_argument('--nc', default=3)
    parser.add_argument('--npoints', default=2048)
    parser.add_argument('--output_dir', type=str, default="./exps/exp_test/", help='directory for experiments logging',)

    # PVD
    # noise schedule
    parser.add_argument('--beta_start', default=0.0001)
    parser.add_argument('--beta_end', default=0.02)
    parser.add_argument('--schedule_type', default='linear')
    parser.add_argument('--time_num', default=500)

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
    parser.add_argument('--lr_gamma', type=float, default=0.9998, help='lr decay for EBM')

    # path to checkpt of trained model and PVD model
    parser.add_argument('--model', default='', help="path to model (to continue training)")
    parser.add_argument('--pvd_model', default='', help="path to pre-trained unconditional PVD")

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed nodes.')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:9991', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--distribution_type', default=None, choices=['multi', 'single', None],
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use. None means using all available GPUs.')

    # evaluation params
    parser.add_argument('--saveEpoch', default=50, help='unit: epoch when checkpoint is saved')  
    parser.add_argument('--diagEpoch', default=50, help='unit: epoch when diagnosis is done')
    parser.add_argument('--vizEpoch', default=50, help='unit: epoch when visualization is done')
    parser.add_argument('--valEpoch', default=1, help='unit: epoch when validation is done')
    parser.add_argument('--compEpoch', default=10000, help='unit: epoch when comparison with unconditional PVD is done')
    parser.add_argument('--val_size', default=None, help='number of clouds evaluated during validation')    # if None => validation is computed on whole validation dset
    parser.add_argument('--print_freq', default=100, help='unit: iter where gradients and step are printed')
    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')

    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    main()