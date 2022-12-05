import logging
logger = logging.getLogger()
import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import math
from pycarus.metrics.chamfer_distance import chamfer


def setup_logging(output_dir):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger()
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    err_handler = logging.StreamHandler(sys.stderr)
    err_handler.setFormatter(log_format)
    logger.addHandler(err_handler)
    logger.setLevel(logging.INFO)

    return logger

def normalize_cloud(pcd: torch.Tensor):
    bc = torch.mean(pcd, dim=1, keepdim=True)
    dist = torch.cdist(pcd, bc)
    max_dist = torch.max(dist, dim=1)[0]
    new_pcd = (pcd - bc) / torch.unsqueeze(max_dist, dim=2)

    return new_pcd

def normalize_clouds_for_validation(pcs, mode, logger):
    if mode is None:
        logger.info('Will not normalize point clouds.')
        return pcs
    logger.info('Normalization mode: %s' % mode)
    for i in tqdm(range(pcs.size(0)), desc='Normalize'):
        pc = pcs[i]
        if mode == 'shape_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif mode == 'shape_bbox':
            pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
            pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
            shift = ((pc_min + pc_max) / 2).view(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        pc = (pc - shift) / scale
        pcs[i] = pc
    return pcs

def visualize_pointcloud_batch(path, pointclouds, pred_labels, labels, categories, vis_label=False, target=None,  elev=30, azim=225):
    batch_size = len(pointclouds)
    fig = plt.figure(figsize=(20,20))

    ncols = int(np.sqrt(batch_size))
    nrows = max(1, (batch_size-1) // ncols+1)
    for idx, pc in enumerate(pointclouds):
        if vis_label:
            label = categories[labels[idx].item()]
            pred = categories[pred_labels[idx]]
            colour = 'b' if label == pred else 'r'
        elif target is None:

            colour = 'b'
        else:
            colour = target[idx]
        pc = pc.cpu().numpy()
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
        ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1], c=colour, s=5)
        ax.view_init(elev=elev, azim=azim)
        ax.axis('off')
        if vis_label:
            ax.set_title('GT: {0}\nPred: {1}'.format(label, pred))

    plt.savefig(path)
    plt.close(fig)

def nested_from_batch(input: torch.Tensor):
    list_tensor=[]
    # convert text embed batch to nested tensor
    for el in input:    # el: [60, 1024]
        # sum all 1024 numbers for each token
        sum_el = torch.sum(el, dim=1) # I get 60 numbers
        # get index of first zero element of the tensor
        non_zero_index = sum_el[sum_el!=0] # shape: [<n of non zero elements>] => [22]
        new_el = el[0:non_zero_index.shape[0]] # new_el: example [22, 1024]
        # check if new_el contains zeros
        sum_new_el = torch.sum(new_el, dim=1)
        non_zeros = torch.count_nonzero(sum_new_el)
        assert non_zeros==new_el.shape[0]
        list_tensor.append(new_el)
    
    # define nested tensor from list
    nested = torch.nested_tensor(list_tensor, device=torch.device('cuda'))
    return nested

def prepare_pvd_weights(pvd_ckpt: dict):
    new_pretrained_dict = OrderedDict()
    pretrained_dict = pvd_ckpt["model_state"]
    for k, v in pretrained_dict.items():
        if not k.startswith('pvd'):
            name = "pvd." + k
        
        name = name.replace('pvd.model.module.sa_layers.1.0.voxel_layers.7.fc.0.weight', 'pvd.model.module.sa_layers.1.0.voxel_layers.9.fc.0.weight')
        name = name.replace('pvd.model.module.sa_layers.1.0.voxel_layers.7.fc.2.weight', 'pvd.model.module.sa_layers.1.0.voxel_layers.9.fc.2.weight')
        name = name.replace('pvd.model.module.sa_layers.3.mlps.0.layers.0.weight', 'pvd.model.module.sa_layers.3.0.mlps.0.layers.0.weight')
        name = name.replace('pvd.model.module.sa_layers.3.mlps.0.layers.0.bias', 'pvd.model.module.sa_layers.3.0.mlps.0.layers.0.bias')
        name = name.replace('pvd.model.module.sa_layers.3.mlps.0.layers.1.weight', 'pvd.model.module.sa_layers.3.0.mlps.0.layers.1.weight')
        name = name.replace('pvd.model.module.sa_layers.3.mlps.0.layers.1.bias', 'pvd.model.module.sa_layers.3.0.mlps.0.layers.1.bias')
        name = name.replace('pvd.model.module.sa_layers.3.mlps.0.layers.3.weight', 'pvd.model.module.sa_layers.3.0.mlps.0.layers.3.weight')
        name = name.replace('pvd.model.module.sa_layers.3.mlps.0.layers.3.bias', 'pvd.model.module.sa_layers.3.0.mlps.0.layers.3.bias')
        name = name.replace('pvd.model.module.sa_layers.3.mlps.0.layers.4.weight', 'pvd.model.module.sa_layers.3.0.mlps.0.layers.4.weight')
        name = name.replace('pvd.model.module.sa_layers.3.mlps.0.layers.4.bias', 'pvd.model.module.sa_layers.3.0.mlps.0.layers.4.bias')
        name = name.replace('pvd.model.module.sa_layers.3.mlps.0.layers.6.weight', 'pvd.model.module.sa_layers.3.0.mlps.0.layers.6.weight')
        name = name.replace('pvd.model.module.sa_layers.3.mlps.0.layers.6.bias', 'pvd.model.module.sa_layers.3.0.mlps.0.layers.6.bias')
        name = name.replace('pvd.model.module.sa_layers.3.mlps.0.layers.7.weight', 'pvd.model.module.sa_layers.3.0.mlps.0.layers.7.weight')
        name = name.replace('pvd.model.module.sa_layers.3.mlps.0.layers.7.bias', 'pvd.model.module.sa_layers.3.0.mlps.0.layers.7.bias')
        name = name.replace('pvd.model.module.fp_layers.1.1.voxel_layers.7.fc.0.weight', 'pvd.model.module.fp_layers.1.1.voxel_layers.10.fc.0.weight')
        name = name.replace('pvd.model.module.fp_layers.1.1.voxel_layers.7.fc.2.weight', 'pvd.model.module.fp_layers.1.1.voxel_layers.10.fc.2.weight')
        name = name.replace('pvd.model.module.sa_layers.0.0.voxel_layers.7.fc.0.weight', 'pvd.model.module.sa_layers.0.0.voxel_layers.9.fc.0.weight')
        name = name.replace('pvd.model.module.sa_layers.0.0.voxel_layers.7.fc.2.weight', 'pvd.model.module.sa_layers.0.0.voxel_layers.9.fc.2.weight')
        name = name.replace('pvd.model.module.sa_layers.0.1.voxel_layers.7.fc.0.weight', 'pvd.model.module.sa_layers.0.1.voxel_layers.9.fc.0.weight')
        name = name.replace('pvd.model.module.sa_layers.0.1.voxel_layers.7.fc.2.weight', 'pvd.model.module.sa_layers.0.1.voxel_layers.9.fc.2.weight')            
        name = name.replace('pvd.model.module.sa_layers.2.0.voxel_layers.7.fc.0.weight', 'pvd.model.module.sa_layers.2.0.voxel_layers.9.fc.0.weight')
        name = name.replace('pvd.model.module.sa_layers.2.0.voxel_layers.7.fc.2.weight', 'pvd.model.module.sa_layers.2.0.voxel_layers.9.fc.2.weight')  
        name = name.replace('pvd.model.module.fp_layers.0.1.voxel_layers.7.fc.0.weight', 'pvd.model.module.fp_layers.0.1.voxel_layers.9.fc.0.weight')
        name = name.replace('pvd.model.module.fp_layers.0.1.voxel_layers.7.fc.2.weight', 'pvd.model.module.fp_layers.0.1.voxel_layers.9.fc.2.weight')
        name = name.replace('pvd.model.module.fp_layers.0.2.voxel_layers.7.fc.0.weight', 'pvd.model.module.fp_layers.0.2.voxel_layers.9.fc.0.weight')
        name = name.replace('pvd.model.module.fp_layers.0.2.voxel_layers.7.fc.2.weight', 'pvd.model.module.fp_layers.0.2.voxel_layers.9.fc.2.weight')            
        name = name.replace('pvd.model.module.fp_layers.0.3.voxel_layers.7.fc.0.weight', 'pvd.model.module.fp_layers.0.3.voxel_layers.9.fc.0.weight')
        name = name.replace('pvd.model.module.fp_layers.0.3.voxel_layers.7.fc.2.weight', 'pvd.model.module.fp_layers.0.3.voxel_layers.9.fc.2.weight')
        name = name.replace('pvd.model.module.fp_layers.1.1.voxel_layers.10.fc.0.weight', 'pvd.model.module.fp_layers.1.1.voxel_layers.9.fc.0.weight')
        name = name.replace('pvd.model.module.fp_layers.1.1.voxel_layers.10.fc.2.weight', 'pvd.model.module.fp_layers.1.1.voxel_layers.9.fc.2.weight')            
        name = name.replace('pvd.model.module.fp_layers.1.2.voxel_layers.7.fc.0.weight', 'pvd.model.module.fp_layers.1.2.voxel_layers.9.fc.0.weight')
        name = name.replace('pvd.model.module.fp_layers.1.2.voxel_layers.7.fc.2.weight', 'pvd.model.module.fp_layers.1.2.voxel_layers.9.fc.2.weight')   
        name = name.replace('pvd.model.module.fp_layers.1.3.voxel_layers.7.fc.0.weight', 'pvd.model.module.fp_layers.1.3.voxel_layers.9.fc.0.weight')
        name = name.replace('pvd.model.module.fp_layers.1.3.voxel_layers.7.fc.2.weight', 'pvd.model.module.fp_layers.1.3.voxel_layers.9.fc.2.weight') 
        name = name.replace('pvd.model.module.fp_layers.2.1.voxel_layers.7.fc.0.weight', 'pvd.model.module.fp_layers.2.1.voxel_layers.9.fc.0.weight')
        name = name.replace('pvd.model.module.fp_layers.2.1.voxel_layers.7.fc.2.weight', 'pvd.model.module.fp_layers.2.1.voxel_layers.9.fc.2.weight')   
        name = name.replace('pvd.model.module.fp_layers.2.2.voxel_layers.7.fc.0.weight', 'pvd.model.module.fp_layers.2.2.voxel_layers.9.fc.0.weight')
        name = name.replace('pvd.model.module.fp_layers.2.2.voxel_layers.7.fc.2.weight', 'pvd.model.module.fp_layers.2.2.voxel_layers.9.fc.2.weight') 
        name = name.replace('pvd.model.module.fp_layers.3.1.voxel_layers.7.fc.0.weight', 'pvd.model.module.fp_layers.3.1.voxel_layers.9.fc.0.weight')
        name = name.replace('pvd.model.module.fp_layers.3.1.voxel_layers.7.fc.2.weight', 'pvd.model.module.fp_layers.3.1.voxel_layers.9.fc.2.weight')  
        name = name.replace('pvd.model.module.fp_layers.3.2.voxel_layers.7.fc.0.weight', 'pvd.model.module.fp_layers.3.2.voxel_layers.9.fc.0.weight')
        name = name.replace('pvd.model.module.fp_layers.3.2.voxel_layers.7.fc.2.weight', 'pvd.model.module.fp_layers.3.2.voxel_layers.9.fc.2.weight')                     

        new_pretrained_dict[name] = v
    return new_pretrained_dict

def maxlen_padding(text_embeddings):
    '''
    CAUTION: this padding must be applied to output of Text2Shape dataset with padding=False
    '''
    sum_text_embed = torch.sum(text_embeddings, dim=2)
    max_seq_len = 0
    for idx, embed in enumerate(sum_text_embed):
        seq_len = torch.count_nonzero(embed)
        if seq_len>max_seq_len:
            max_seq_len=seq_len
    
    #print('longest len: ', max_seq_len)
    #print('longest sentence: ', text[max_seq_len_idx])
    
    text_embeddings = text_embeddings[:,:max_seq_len, :]

    return text_embeddings
