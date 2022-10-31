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
from train import normalize_clouds_for_validation

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

def get_test_text2shape_dataset(dataroot, category):
    test_dataset = Text2Shape(root=Path(dataroot),
        split="test",
        categories=category,
        from_shapenet_v1=True,
        from_shapenet_v2=False,
        conditional_setup=True,
        language_model="t5-11b",
        lowercase_text=False,
        max_length=77,
        padding=False,
        scale_mode="global_unit")

    return test_dataset
    
def get_test_dataloader(opt, test_dataset):

    if opt.distribution_type == 'multi':
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset,
            num_replicas=opt.world_size,
            rank=opt.rank
        )
    else:
        test_sampler = None

    print('WORKERS: ', int(opt.workers))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.bs,sampler=test_sampler,
                                                   shuffle=test_sampler is None, num_workers=int(opt.workers), drop_last=True)
                                                    # drop_last drops the last incomplete batch, if the dataset size is not divisible by the batch size


    return test_dataloader, test_sampler


def generate(model, opt):

    test_dataset = get_test_text2shape_dataset(opt.t2s_dataroot, opt.category)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.bs,
                                                  shuffle=False, num_workers=int(opt.workers), drop_last=False)

    print('Test Dataset size: ', len(test_dataset))

    with torch.no_grad():

        samples = []
        ref = []

        for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Generating Samples'):

            x = data['pointcloud'].transpose(1,2)   # reference clouds
            text_embed = data['text_embed'].cuda()
            #m, s = data['mean'].float(), data['std'].float()   # in Text2Shape, we already do it when we initialize the dataset
            
            x_gen_eval = model.get_clouds(text_embed, x)
            gen = model.pvd.gen_samples(x.shape,
                                       'cuda', clip_denoised=False).detach().cpu()

            gen = gen.transpose(1,2).contiguous()
            x = x.transpose(1,2).contiguous()

            #gen = gen * s + m
            #x = x * s + m
            samples.append(gen)
            ref.append(x)

            visualize_pointcloud_batch(os.path.join(str(Path(opt.eval_path).parent), 'x.png'), gen[:64], None,
                                       None, None)

        samples = torch.cat(samples, dim=0)
        ref = torch.cat(ref, dim=0)
        samples = normalize_clouds_for_validation(samples, mode='shape_bbox', logger=logger)
        ref = normalize_clouds_for_validation(ref, mode='shape_bbox', logger=logger)

        torch.save(samples, opt.eval_path)

    return ref

def evaluate_gen(opt, ref_pcs, logger):

    if ref_pcs is None:
        test_dataset = get_test_text2shape_dataset(opt.dataroot, opt.category)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size,
                                                      shuffle=False, num_workers=int(opt.workers), drop_last=False)
        ref = []
        for data in tqdm(test_dataloader, total=len(test_dataloader), desc='Generating Samples'):
            x = data['pointcloud']  # reference clouds
            m, s = data['mean'].float(), data['std'].float()

            ref.append(x*s + m)

        ref_pcs = torch.cat(ref, dim=0).contiguous()

    logger.info("Loading sample path: %s"
      % (opt.eval_path))
    sample_pcs = torch.load(opt.eval_path).contiguous()

    logger.info("Generation sample size:%s reference size: %s"
          % (sample_pcs.size(), ref_pcs.size()))


    # Compute metrics
    results = evaluation_metrics.compute_all_metrics(sample_pcs, ref_pcs, opt.bs)
    results = {k: (v.cpu().detach().item()
                   if not isinstance(v, float) else v) for k, v in results.items()}

    print(results)
    logger.info(results)

    jsd = evaluation_metrics.jsd_between_point_cloud_sets(sample_pcs.numpy(), ref_pcs.numpy())
    print('JSD: {}'.format(jsd))
    logger.info('JSD: {}'.format(jsd))

def main():
    opt = parse_args()
    if opt.category == 'airplane':
        opt.beta_start = 1e-5
        opt.beta_end = 0.008
        opt.schedule_type = 'warm0.1'

    logger = setup_logging(opt.eval_dir)

    model = Infusion(opt)

    model.cuda()

    def _transform_(m):
        return nn.parallel.DataParallel(m)

    model = model.cuda()
    model.multi_gpu_wrapper(_transform_)

    model.pvd.eval()

    with torch.no_grad():

        logger.info("Resume Path:%s" % opt.model)

        resumed_param = torch.load(opt.model)
        model.load_state_dict(resumed_param['model_state'])
        epoch= resumed_param['epoch']
        outf_syn = os.path.join(opt.eval_dir + f'_{epoch}', 'syn')
        if not os.path.exists(outf_syn):
            outf_syn = os.makedirs(outf_syn)
        

        ref = None
        if opt.generate:
            opt.eval_path = os.path.join(outf_syn, 'samples.pth')
            Path(opt.eval_path).parent.mkdir(parents=True, exist_ok=True)
            ref=generate(model, opt)
            
        if opt.eval_gen:
            # Evaluate generation
            evaluate_gen(opt, ref, logger)

def parse_args():
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument('--train_ds', default='text2shape', choices=['shapenet', 'text2shape'], help='dataset to use for training')
    parser.add_argument('--sn_dataroot', default='../PVD/data/ShapeNetCore.v2.PC15k/', help="dataroot of ShapeNet")
    parser.add_argument('--t2s_dataroot', default='/media/data2/aamaduzzi/datasets/Text2Shape/', help="dataroot of ShapeNet")
    parser.add_argument('--category', default='chair')
    parser.add_argument('--bs', type=int, default=64, help='input batch size')
    parser.add_argument('--workers', type=int, default=2, help='workers')
    parser.add_argument('--n_epochs', type=int, default=10000, help='number of epochs to train for')
    parser.add_argument('--nc', default=3)
    parser.add_argument('--npoints', default=2048)
    parser.add_argument('--eval_dir', default='./evaluation/finetune_PVD_chair', help='directory for evaluation results')

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
    parser.add_argument('--generate', default=True)
    parser.add_argument('--eval_gen', default=True)


    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    main()