import copy
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
from dataset.text2shape_dataset import Text2Shape
from dataset.utils import *
from pycarus.metrics.chamfer_distance import chamfer
from pathlib import Path
from tqdm import tqdm
from torchsummary import summary
from datetime import datetime
import os
import math
from dataset.shapenet_data_pc import ShapeNet15kPointClouds
sys.path.append('../diffusion-text-shape/')
from evaluation import evaluation_metrics

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
        max_length=77,
        padding=False,
        scale_mode="global_unit")

    val_dataset = Text2Shape(root=Path(dataroot),
        split="val",
        categories=category,
        from_shapenet_v1=True,
        from_shapenet_v2=False,
        conditional_setup=True,
        language_model="t5-11b",
        lowercase_text=False,
        max_length=77,
        padding=False,
        scale_mode="global_unit"
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

def train(gpu, opt, output_dir, train_dset, val_dset, noises_init):

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
    
    writer = torch.utils.tensorboard.SummaryWriter(output_dir)
        
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
    
    optimizer= torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.decay, betas=(opt.beta1, 0.999))

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, opt.lr_gamma)

    if opt.model != '':
        ckpt = torch.load(opt.model)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = torch.load(opt.model)['epoch'] + 1
    else:
        start_epoch = 0

    def new_x_chain(x, num_chain):
        return torch.randn(num_chain, *x.shape[1:], device=x.device)

    for epoch in range(start_epoch, opt.n_epochs):

        if opt.distribution_type == 'multi':
            train_sampler.set_epoch(epoch)

        for i, data in enumerate(train_dataloader):
            if opt.train_ds == 'shapenet':
                x = data['train_points'].transpose(1,2)                 # TODO: check if pointclouds have the same size for both ShapeNet and Text2Shape 
                noises_batch = noises_init[data['idx']].transpose(1,2)  # TODO: check if idx are the same for both ShapeNet and Text2Shape
                #text_embed = data["text_embed"]
            elif opt.train_ds == 'text2shape':
                x = data['pointcloud'].transpose(1,2)                   # TODO: check if pointclouds have the same size for both ShapeNet and Text2Shape          
                noises_batch = noises_init[data['idx']].transpose(1,2)  # TODO: check if idx are the same for both ShapeNet and Text2Shape
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

            # apply clever padding: truncate text embeds to length of longest sequence: add zeros if needed
            # find longest sequence in batch. text_embed has shape (B,60,1024)
            sum_text_embed = torch.sum(text_embed, dim=2)
            max_seq_len = 0
            max_seq_len_idx = 0
            for idx, embed in enumerate(sum_text_embed):
                zero_idx = torch.argmin(abs(embed), dim=0)
                if zero_idx.item()>max_seq_len:
                    max_seq_len=zero_idx
                    max_seq_len_idx = idx
            
            #print('longest len: ', max_seq_len)
            #print('longest sentence: ', text[max_seq_len_idx])

            text_embed = text_embed[:,:max_seq_len, :]

            model.pvd.train()
            loss = model.get_loss(x, noises_batch, text_embed).mean()
            
            optimizer.zero_grad()   # this command sets to zero the gradient of the params to optimize
            loss.backward()
            netpNorm, netgradNorm = getGradNorm(model)

            if opt.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip, error_if_nonfinite=True)

            optimizer.step()
            lr_scheduler.step()

            writer.add_scalar('train/loss', loss, i)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], i)
            writer.add_scalar('train/grad', netgradNorm, i)
            writer.flush()
            
            for idx, p in enumerate(model.parameters()):
                param = p
                # check if PVD params are updated correctly
                '''
                if idx==10:
                    if i==0:
                        old_param_val = copy.deepcopy(param.data)   # param.data is a pointer! In this way, we only get its value
                    else:
                        new_param_val = copy.deepcopy(param.data)
                        
                        if torch.equal(old_param_val, new_param_val):
                            print('PVD PARAM 10 not updated :( ')
                        else:
                            print('PVD PARAM 10 updated correctly :) ')
                        old_param_val = new_param_val
                '''
                if torch.isnan(param.grad).any():
                    print('Nan gradient in pvd param ')


            if i % opt.print_freq == 0 and should_diag:
                logger.info('[{:>3d}/{:>3d}][{:>3d}/{:>3d}]    loss: {:>10.4f},    '
                             'netpNorm_PVD: {:>10.2f},   netgradNorm_PVD: {:>10.2f}     '
                             .format(
                        epoch, opt.n_epochs, i, len(train_dataloader),loss.item(),
                    netpNorm, netgradNorm,
                        ))
                
        if (epoch + 1) % opt.vizIter == 0 and should_diag:
            model.pvd.eval()
            logger.info('Generating clouds for visualization on training set...')

            with torch.no_grad():
                x_gen_eval = model.get_clouds(text_embed, x)
                x_gen_list = model.get_cloud_traj(text_embed[0].unsqueeze(0), x)
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

            visualize_pointcloud_batch('%s/epoch_%03d_samples_eval_train.png' % (outf_syn, epoch),
                                    x_gen_eval.transpose(1, 2), None, None,
                                    None)

            visualize_pointcloud_batch('%s/epoch_%03d_samples_eval_all_train.png' % (outf_syn, epoch),
                                    x_gen_all.transpose(1, 2), None,
                                    None,
                                    None)

            visualize_pointcloud_batch('%s/epoch_%03d_x_train.png' % (outf_syn, epoch), x.transpose(1, 2), None,
                                    None,
                                    None)

            logger.info('Generating clouds for visualization on validation set...')
            with torch.no_grad():
                val_batch = next(iter(val_dataloader))
                text_embed_val = val_batch["text_embed"].cuda()
                x_val = val_batch['pointcloud'].transpose(1,2).cuda() 
                x_gen_eval = model.get_clouds(text_embed_val, x_val)
                x_gen_list = model.get_cloud_traj(text_embed_val[0].unsqueeze(0), x_val)
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

            visualize_pointcloud_batch('%s/epoch_%03d_samples_eval_valid.png' % (outf_syn, epoch),
                                    x_gen_eval.transpose(1, 2), None, None,
                                    None)

            visualize_pointcloud_batch('%s/epoch_%03d_samples_eval_all_valid.png' % (outf_syn, epoch),
                                    x_gen_all.transpose(1, 2), None,
                                    None,
                                    None)

            visualize_pointcloud_batch('%s/epoch_%03d_x_valid.png' % (outf_syn, epoch), x_val.transpose(1, 2), None,
                                    None,
                                    None)



        if (epoch + 1) % opt.diagIter == 0 and should_diag:
            model.pvd.eval()
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

        if (epoch + 1) % opt.saveIter == 0:
            model.pvd.eval()
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

        if (epoch + 1 ) % opt.valIter == 0 and should_diag:

            val_folder = os.path.join(output_dir,'validation')
            # build reference histogram for validation set
            ref_hist = dict()
            for i in range(len(val_dset)):
                if val_dset[i]["model_id"] in ref_hist.keys():
                    ref_hist[val_dset[i]["model_id"]] += 1
                else:
                    ref_hist[val_dset[i]["model_id"]] = 1

            model.pvd.eval()
            gen_pcs=[]
            ref_pcs=[]
            texts=[]
            model_ids=[]
            for i in tqdm(range(0, math.ceil(opt.val_size / opt.bs)), 'Generate'):
                with torch.no_grad():
                    val_batch = next(iter(val_dataloader))
                    text_embed_val = val_batch["text_embed"].cuda()
                    x_val = val_batch['pointcloud'].transpose(1,2).cuda() 
                    x_gen_eval = model.get_clouds(text_embed_val, x_val)
                    for text in val_batch["text"]:
                        texts.append(text)
                    for model_id in val_batch["model_id"]:
                        model_ids.append(model_id)
                    # transpose shapes because metrics want (2048, 3) instead of (3, 2048)
                    x_gen_eval = x_gen_eval.transpose(1,2)
                    x_val = x_val.transpose(1,2)
                    gen_pcs.append(x_gen_eval.detach().cpu())
                    ref_pcs.append(x_val.detach().cpu())
            gen_pcs = torch.cat(gen_pcs, dim=0)[:opt.val_size]
            ref_pcs = torch.cat(ref_pcs, dim=0)[:opt.val_size]

            texts = texts[:opt.val_size]
            model_ids = model_ids[:opt.val_size]
            gen_pcs = normalize_clouds_for_validation(gen_pcs, mode='shape_bbox', logger=logger)
            ref_pcs = normalize_clouds_for_validation(ref_pcs, mode='shape_bbox', logger=logger)

            print('size of ref pcs: ', ref_pcs.shape)
            print('size of gen pcs: ', gen_pcs.shape)
            print('len of gen texts: ', len(texts))

            # Save
            logger.info('Saving point clouds and text...')
            np.save(os.path.join(val_folder, 'out.npy'), gen_pcs.numpy())
            np.save(os.path.join(val_folder, 'ref.npy'), ref_pcs.numpy())
            count=0
            with open(os.path.join(val_folder, 'texts.txt'), 'w') as f:
                for text in texts:
                    f.write(text + "\n")
                    count +=1
            with open(os.path.join(val_folder, 'model_ids.txt'), 'w') as f:
                for model_id in model_ids:
                    f.write(f"{model_id}\n")

            # Compute metrics
            with torch.no_grad():
                results = evaluation_metrics.compute_all_metrics(gen_pcs.cuda(), ref_pcs.cuda(), opt.bs, model_ids=model_ids, texts=texts, save_dir=val_folder, ref_hist=ref_hist)
                results = {k:v.item() for k, v in results.items()}
                jsd = evaluation_metrics.jsd_between_point_cloud_sets(gen_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
                results['jsd'] = jsd

            for k, v in results.items():
                logger.info('%s: %.12f' % (k, v))
            
            # Display metrics on Tensorboard
            # CD related metrics
            writer.add_scalar('val/Coverage_CD', results['lgan_cov-CD'], i)
            writer.add_scalar('val/MMD_CD', results['lgan_mmd-CD'], i)
            #writer.add_scalar('val/1NN_CD', results['1-NN-CD-acc'], i)
            # JSD
            writer.add_scalar('val/JSD', results['jsd'], i)
            writer.flush()

            logger.info('[Val] Coverage  | CD %.6f | EMD n/a' % (results['lgan_cov-CD'], ))
            logger.info('[Val] MinMatDis | CD %.6f | EMD n/a' % (results['lgan_mmd-CD'], ))
            logger.info('[Val] JsnShnDis | %.6f ' % (results['jsd']))

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
        if opt.pvd_model != '':
            opt.category = "chair"
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
        train(opt.gpu, opt, opt.output_dir, train_dataset, val_dataset, noises_init)

def parse_args():

    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument('--train_ds', default='text2shape', choices=['shapenet', 'text2shape'], help='dataset to use for training')
    parser.add_argument('--sn_dataroot', default='../PVD/data/ShapeNetCore.v2.PC15k/', help="dataroot of ShapeNet")
    parser.add_argument('--t2s_dataroot', default='/media/data2/aamaduzzi/datasets/Text2Shape/', help="dataroot of ShapeNet")
    parser.add_argument('--category', default='all')
    parser.add_argument('--bs', type=int, default=64, help='input batch size')
    parser.add_argument('--workers', type=int, default=2, help='workers')
    parser.add_argument('--n_epochs', type=int, default=10000, help='number of epochs to train for')
    parser.add_argument('--nc', default=3)
    parser.add_argument('--npoints', default=2048)
    parser.add_argument('--output_dir', type=str, default="./exps/exp_9", help='directory for experiments logging',)

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

    # path to checkpt of trained model and PVD model
    parser.add_argument('--model', default='', help="path to model (to continue training)")
    parser.add_argument('--pvd_model', default='', help="path to PVD model (to freeze PVD")


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
    parser.add_argument('--saveIter', default=50, help='unit: epoch when checkpoint is saved')  
    parser.add_argument('--diagIter', default=100, help='unit: epoch when diagnosis is done')
    parser.add_argument('--vizIter', default=100, help='unit: epoch when visualization is done')
    parser.add_argument('--valIter', default=150, help='unit: epoch when validation is done')
    parser.add_argument('--val_size', default=1000, help='number of clouds evaluated during validation')
    parser.add_argument('--print_freq', default=100, help='unit: iter where gradients and step are printed')
    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')


    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    main()