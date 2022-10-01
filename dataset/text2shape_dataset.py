from curses.textpad import Textbox
from typing import List, Callable
from pathlib import Path
from torch.utils.data import Dataset
from pycarus.transforms.var import Compose
from pycarus.geometry.pcd import get_tensor_pcd_from_o3d
import pandas as pd
import open3d as o3d
import torch
from copy import copy
import numpy as np
import os
import random

DEFAULT_T5_NAME = 't5-11b'
   
class Text2Shape(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        categories: str,
        from_shapenet_v1: bool,
        from_shapenet_v2: bool,
        language_model: str,
        lowercase_text: bool,
        max_length: int,
        conditional_setup: bool,
        scale_mode: str,
        transforms: List[Callable] = []
    ) -> None:

        '''
        Class implementing the Text2Shape dataset proposed in:

            Chen, Kevin and Choy, Christopher B and Savva, Manolis and Chang, Angel X and Funkhouser, 
            Thomas and Savarese, Silvio (2019).
            Text2Shape: Generating Shapes from Natural Language by Learning Joint Embeddings
            arXiv preprint arXiv:1803.08495

            Args:
                root: The path to the folder containing the dataset.
                split: The name of the split to load.
                categories: A list of shape categories to select. Defaults to [].
                transforms_complete: The transform to apply to the complete cloud. Defaults to [].
                transforms_incomplete: The transform to apply to the incomplete cloud. Defaults to [].
                transforms_all: The transform to apply to both the point clouds. Defaults to [].

            Raises:
                FileNotFoundError: If the folder does not exist.
                ValueError: If the chosen split is not allowed.
            '''
        super().__init__()
        print('initializing dataset...')
        self.split = split
        self.splits = ["train", "val", "test"]
        if self.split not in self.splits:
            raise ValueError(f"{self.split} value not allowed, only allowed {self.splits}.")
        self.root = root # dataset_text2shape
        if not self.root.is_dir():
            raise FileNotFoundError(f"{self.root} not found.")
        self.transform = Compose(transforms)
        self.categories = categories
        self.scale_mode = scale_mode
        
        self.pointclouds=[]
        
        # get text_prompt, model_id, category
        if from_shapenet_v2:
            annotations_path = self.root / "annotations" / "from_shapenet_v2" / f"{self.split}_{self.categories}.csv"
        elif from_shapenet_v1:
            annotations_path = self.root / "annotations" / "from_shapenet_v1" / f"{self.split}_{self.categories}.csv"
        else:
            annotations_path = self.root / "annotations" / "from_text2shape" / f"{self.split}_{self.categories}.csv"


        print('annotations: ', annotations_path)
        df = pd.read_csv(annotations_path, quotechar='"')
        # if I am training only with shapes, I remove rows of the same shape and different text
        if not conditional_setup:
            df.drop_duplicates(subset=['modelId'], inplace=True)
        for idx, row in df.iterrows():
            # read_pcd
            text = row["description"]
            text = text.strip('"')
            model_id = row["modelId"]
            category = row["category"]
            if from_shapenet_v1:
                model_path = str(self.root / "shapes" / "shapenet_v1" / f"{model_id}.ply")
                o3d_pcd = o3d.io.read_point_cloud(model_path)
                pc = get_tensor_pcd_from_o3d(o3d_pcd)
            elif from_shapenet_v2:
                model_path = str(self.root / "shapes" / "shapenet_v2" / f"{model_id}.ply")
                o3d_pcd = o3d.io.read_point_cloud(model_path)
                pc = get_tensor_pcd_from_o3d(o3d_pcd)
            else:
                model_path = str(self.root / "shapes" / "text2shape" / f"{model_id}.ply")
                o3d_pcd = o3d.io.read_point_cloud(model_path)
                pc = get_tensor_pcd_from_o3d(o3d_pcd)
            #scaling the pcd (from diffusion-pointcloud code)
            if self.scale_mode == 'global_unit':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = self.stats['std'].reshape(1, 1) # stats does not exist! take the code from diffusion-point-cloud
            elif self.scale_mode == 'shape_unit':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1)
            elif self.scale_mode == 'shape_half':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1) / (0.5)
            elif self.scale_mode == 'shape_34':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1) / (0.75)
            elif self.scale_mode == 'shape_bbox':
                    pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
                    pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
                    shift = ((pc_min + pc_max) / 2).view(1, 3)
                    scale = (pc_max - pc_min).max().reshape(1, 1) / 2
            else:
                    shift = torch.zeros([1, 3])
                    scale = torch.ones([1, 1])

            pc = (pc - shift) / scale

            tensor_name = row["tensor"]
            if lowercase_text:
                text_embed_path = self.root / "text_embeds" / language_model / "lowercase" / f"{tensor_name}"
            else:
                text_embed_path = self.root / "text_embeds" / language_model / "std" / f"{tensor_name}"
            
            text_embed = torch.load(text_embed_path)[:max_length].to('cuda')

            # build mask for this text embedding (i have text embeds length and max length)
            key_padding_mask_false = torch.zeros((1+text_embed.shape[0]), dtype=torch.bool)            # False => elements will be processed
            key_padding_mask_true = torch.ones((max_length - text_embed.shape[0]), dtype=torch.bool) # True => elements will NOT be processed
            key_padding_mask = torch.cat((key_padding_mask_false, key_padding_mask_true), dim=0).to('cuda')


            # pad to length to max_length_t2s
            # add zeros at the end of text embed to reach max_length            
            pad = torch.zeros(max_length - text_embed.shape[0], text_embed.shape[1]).to('cuda')
            text_embed = torch.cat((text_embed, pad), dim=0)

            # avg pooling
            #text_embed = torch.mean(text_embed, dim=0)

            # max pooling
            #text_embed = torch.amax(text_embed, dim=0)

            self.pointclouds.append({
                    'pointcloud': pc,
                    'text' : text,
                    "text_embed": text_embed,
                    'model_id': model_id,
                    'cate': category,
                    'shift': shift,
                    'scale': scale,
                    'key_pad_mask': key_padding_mask,
                })

        # Deterministically shuffle the dataset
        self.pointclouds.sort(key=lambda data: data['model_id'], reverse=False)
        random.Random(2020).shuffle(self.pointclouds)


    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {k:v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.pointclouds[idx].items()}
        if self.transform is not None:
            data = self.transform(data)
        return data

