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
        lowercase_text=True,
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
    ref_hist = dict()
    texts = []
    model_ids = []
    for i in range(len(test_dataset)):
        if test_dataset[i]["model_id"] in ref_hist.keys():
            ref_hist[test_dataset[i]["model_id"]] += 1
        else:
            ref_hist[test_dataset[i]["model_id"]] = 1
        texts.append(test_dataset[i]["text"])
        model_ids.append(test_dataset[i]["model_id"])

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.bs,
                                                  shuffle=False, num_workers=int(opt.workers), drop_last=False)

    print('Test Dataset size: ', len(test_dataset))

    samples = []
    ref = []

    for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='Generating Samples'):
        x = data['pointcloud'].transpose(1,2).cuda()   # reference clouds
        text_embed = data['text_embed'].cuda()
        text_embed = maxlen_padding(text_embed)
        mean, std = data['mean'].float(), data['std'].float()   # in Text2Shape, we already do it when we initialize the dataset
            
        gen = model.get_clouds(text_embed, x).detach().cpu()
        gen = gen.transpose(1,2).contiguous()
        x = x.transpose(1,2).contiguous()

        # de-normalization
        gen = gen * std + mean
        x = x.detach().cpu() * std + mean
        
        samples.append(gen)
        ref.append(x)

        # visualize first batch of clouds
        if i==0:
            visualize_pointcloud_batch(os.path.join(str(Path(opt.eval_path).parent), 'gen.png'), gen[:10], None,
                                       None, None)

            visualize_pointcloud_batch(os.path.join(str(Path(opt.eval_path).parent), 'ref.png'), x[:10], None,
                                       None, None)

    samples = torch.cat(samples, dim=0)
    ref = torch.cat(ref, dim=0)
        
    #samples = normalize_clouds_for_validation(samples, mode='shape_bbox', logger=logger)
    #ref = normalize_clouds_for_validation(ref, mode='shape_bbox', logger=logger)

    torch.save(samples, opt.eval_path)

    return ref, ref_hist, texts, model_ids

def evaluate_gen(opt, ref_pcs, ref_hist, texts, model_ids, logger):

    if ref_pcs is None:
        test_dataset = get_test_text2shape_dataset(opt.dataroot, opt.category)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size,
                                                      shuffle=False, num_workers=int(opt.workers), drop_last=False)
        ref = []
        for data in tqdm(test_dataloader, desc='Generating Samples'):   # tqdm(dataloader) will automatically iterate over the WHOLE dataloader (=dataset)
            x = data['pointcloud']  # reference clouds
            m, s = data['mean'].float(), data['std'].float()
    
            # de-normalize clouds
            ref.append(x.detach().cpu() * s + m)

        ref_pcs = torch.cat(ref, dim=0).contiguous()

    logger.info("Loading sample path: %s"
      % (opt.eval_path))
    sample_pcs = torch.load(opt.eval_path).contiguous()

    logger.info("Generation sample size:%s reference size: %s"
          % (sample_pcs.size(), ref_pcs.size()))

    np.save(os.path.join(Path(opt.eval_path).parent, 'ref.npy'), ref_pcs.cpu().numpy())
    np.save(os.path.join(Path(opt.eval_path).parent, 'out.npy'), sample_pcs.cpu().numpy())

    # Compute metrics
    print('samples: ', sample_pcs.shape)
    print('ref: ', ref_pcs.shape)
    #print('computing chamfer distance between all clouds...')
    #chamfer_dist = chamfer(ref_pcs, sample_pcs)
    #chamfer_dist = chamfer_dist[0]
    #print('chamfer: ', chamfer_dist.shape)
    #np.save(os.path.join(Path(opt.eval_path).parent, 'chamfer.npy'), chamfer_dist.cpu().numpy())

    logger.info('Computing evaluation metrics')
    results = evaluation_metrics.compute_all_metrics(sample_pcs.cuda(), ref_pcs.cuda(), opt.bs, model_ids=model_ids, texts=texts, save_dir=Path(opt.eval_path).parent, ref_hist=ref_hist)
    results = {k: (v.cpu().detach().item()
                   if not isinstance(v, float) else v) for k, v in results.items()}

    print(results)
    logger.info(results)

    jsd = evaluation_metrics.jsd_between_point_cloud_sets(sample_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
    print('JSD: {}'.format(jsd))
    logger.info('JSD: {}'.format(jsd))

def main():
    opt = parse_args()
    if opt.category == 'airplane':
        opt.beta_start = 1e-5
        opt.beta_end = 0.008
        opt.schedule_type = 'warm0.1'
    
    model = Infusion(opt)

    if opt.distribution_type == 'multi':  # Multiple processes, single GPU per process
        def _transform_(m):
            return nn.parallel.DistributedDataParallel(
                m, device_ids=[opt.gpu], output_device=opt.gpu)

        torch.cuda.set_device(opt.gpu)
        model.cuda(opt.gpu)
        model.multi_gpu_wrapper(_transform_)

    elif opt.distribution_type == 'single':
        def _transform_(m):
            return nn.parallel.DataParallel(m, device_ids=[0,1])
        model = model.cuda()
        model.multi_gpu_wrapper(_transform_)
    elif opt.gpu is not None:
        torch.cuda.set_device(opt.gpu)
        model = model.cuda(opt.gpu)
    else:
        raise ValueError('distribution_type = multi | single | None')

    def _transform_(m):
        return nn.parallel.DataParallel(m, device_ids=[0,1])

    model.pvd.eval()

    with torch.no_grad():
        print('Loading model ', opt.model)
        resumed_param = torch.load(opt.model)
        weights_dict = resumed_param['model_state']

        if opt.distribution_type == 'single':
            model.multi_gpu_wrapper(_transform_)
            model.load_state_dict(weights_dict)
            epoch = resumed_param['epoch']  
        elif opt.distribution_type is None:
            # remove .module. from the weights dictionary
            new_weights_dict = {}
            for k, v in weights_dict.items():
                name = k
                name = name.replace('.module.', '.')    
                new_weights_dict[name] = v   
            model.load_state_dict(new_weights_dict)
            epoch = resumed_param['epoch']           
        elif opt.distribution_type == 'multi':
            raise Exception('multi processing not allowed during test')


        outf_dir = os.path.join(opt.eval_dir, f'epoch_{epoch}')
        print('outf_dir: ', outf_dir)
        if not os.path.exists(outf_dir):
            outf_dir = os.makedirs(outf_dir)
        logger = setup_logging(outf_dir)
        
        ref = None
        if opt.generate:
            opt.eval_path = os.path.join(outf_dir, 'samples.pth')
            Path(opt.eval_path).parent.mkdir(parents=True, exist_ok=True)
            ref_pcs, ref_hist, texts, model_ids = generate(model, opt)
            
        if opt.eval_gen:
            # Evaluate generation
            evaluate_gen(opt, ref_pcs, ref_hist, texts, model_ids, logger)

def parse_args():
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument('--sn_dataroot', default='../PVD/data/ShapeNetCore.v2.PC15k/', help="dataroot of ShapeNet")
    parser.add_argument('--t2s_dataroot', default='/media/data2/aamaduzzi/datasets/Text2Shape/', help="dataroot of ShapeNet")
    parser.add_argument('--category', default='all')
    parser.add_argument('--bs', type=int, default=150, help='input batch size')
    parser.add_argument('--workers', type=int, default=0, help='workers')
    parser.add_argument('--nc', default=3)
    parser.add_argument('--npoints', default=2048)
    parser.add_argument('--eval_dir', default='./evaluation/small_500_steps_fp16', help='directory for evaluation results')

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


    # path to checkpt of trained model and PVD model
    parser.add_argument('--model', default='./exps/smaller_pvd_500_fp16/epoch_99.pth', help="path to model (to continue training)")

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
    parser.add_argument('--gpu', default=1, type=int,
                        help='GPU id to use. None means using all available GPUs.')

    # evaluation params
    parser.add_argument('--generate', default=True)
    parser.add_argument('--eval_gen', default=True)


    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    main()