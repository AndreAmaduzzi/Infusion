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
from datetime import datetime
import matplotlib.pyplot as plt
import string


DEFAULT_T5_NAME = 't5-11b'

def shuffle_ids(ids, label, random_seed=None):
    ''' e.g. if [a, b] with label 0 makes it [b, a] with label 1.
    '''
    res_ids = ids.copy()   # initialization of output list
    if random_seed is not None:
        np.random.seed(random_seed)
    shuffle = np.random.shuffle
    idx = np.arange(0, len(ids))
    shuffle(idx)
    # if idx==0, no swap
    # if idx==1, swap elements of ids
    i=0
    for one_idx in idx:
        res_ids[i] = ids[one_idx]
        i += 1 

    target = idx[0]

    return res_ids, target

def visualize_data_sample(pointclouds, target, text, path):
    n_clouds = len(pointclouds)
    fig = plt.figure(figsize=(20,20))
    plt.title(label=text + f", target: {target}", fontsize=15)
    plt.axis('off')
    ncols = n_clouds
    nrows = 1
    for idx, pc in enumerate(pointclouds):
        colour = 'r' if target == idx else 'b'
        pc = pc.cpu().numpy()
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
        ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1], c=colour, s=10)
        ax.view_init(elev=30, azim=255)
        ax.axis('off')
    plt.savefig(path)
    plt.close(fig)

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
        padding: bool,
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
            train_path = self.root / "annotations" / "from_shapenet_v2" / f"train_{self.categories}.csv"
        elif from_shapenet_v1:
            annotations_path = self.root / "annotations" / "from_shapenet_v1" / f"{self.split}_{self.categories}.csv"
            train_path = self.root / "annotations" / "from_shapenet_v1" / f"train_{self.categories}.csv"
        else:
            annotations_path = self.root / "annotations" / "from_text2shape" / f"{self.split}_{self.categories}.csv"
            train_path = self.root / "annotations" / "from_tetx2shape" / f"train_{self.categories}.csv"


        print('annotations: ', annotations_path)
        df = pd.read_csv(annotations_path, quotechar='"')
        # if I am training only with shapes, I remove rows of the same shape and different text
        if not conditional_setup:
            df.drop_duplicates(subset=['modelId'], inplace=True)
        
        if self.scale_mode == 'global_unit':   # normalize with mean and std of training set of Text2Shape
            # read DataFrame of training set
            train_ds_df = pd.read_csv(train_path, quotechar='"')
            train_ds_df.drop_duplicates(subset=['modelId'], inplace=True)
            global_mean, global_std = self.get_ds_statistics(train_ds_df, from_shapenet_v1, from_shapenet_v2)
            print('normalization with mean and std of training set')
            print('global mean: ', global_mean)
            print('global std: ', global_std)


        for idx, row in df.iterrows():
            # read_pcd
            text = row["description"]
            text = text.strip('"')
            model_id = row["modelId"]
            category = row["category"]
            if from_shapenet_v1:
                model_path = str(self.root / "shapes" / "shapenet_v1" / f"{model_id}.ply")
                pc = o3d.io.read_point_cloud(model_path)
                pc = get_tensor_pcd_from_o3d(pc)
            elif from_shapenet_v2:
                model_path = str(self.root / "shapes" / "shapenet_v2" / f"{model_id}.ply")
                pc = o3d.io.read_point_cloud(model_path)
                pc = get_tensor_pcd_from_o3d(pc)
            else:
                model_path = str(self.root / "shapes" / "text2shape" / f"{model_id}.ply")
                pc = o3d.io.read_point_cloud(model_path)
                pc = get_tensor_pcd_from_o3d(pc)
            #scaling the pcd (from diffusion-pointcloud code)
            if self.scale_mode == "global_unit":
                shift = global_mean
                scale = global_std
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
                if padding:
                    text_embed_path = self.root / "text_embeds" / language_model / "lowercase" /  "padding" / f"{tensor_name}"
                else:
                    text_embed_path = self.root / "text_embeds" / language_model / "lowercase" / f"{tensor_name}"
            else:
                if padding:
                    text_embed_path = self.root / "text_embeds" / language_model / "std" /  "padding" / f"{tensor_name}"
                else:
                    text_embed_path = self.root / "text_embeds" / language_model / "std" / f"{tensor_name}"
            
            if padding:
                text_embed = torch.load(text_embed_path, map_location='cpu')  # TODO: check if I am still loading data on GPU or not
            else:
                text_embed = torch.load(text_embed_path, map_location='cpu')[:max_length]  # TODO: check if I am still loading data on GPU or not

                
            # build mask for this text embedding (i have text embeds length and max length)
            #key_padding_mask_false = torch.zeros((1+text_embed.shape[0]), dtype=torch.bool)             # False => elements will be processed
            #key_padding_mask_true = torch.ones((max_length - text_embed.shape[0]), dtype=torch.bool)    # True => elements will NOT be processed
            #key_padding_mask = torch.cat((key_padding_mask_false, key_padding_mask_true), dim=0)


            # pad to length to max_length_t2s
            # add zeros at the end of text embed to reach max_length     
            if not padding: #if the language model has not padded the embeddings, we have to do it by hand, to ensure correct batches in DataLoaders
                pad = torch.zeros(max_length - text_embed.shape[0], text_embed.shape[1])
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
                    'mean': shift,
                    'std': scale,
                    #'key_pad_mask': key_padding_mask,
                })

        # Deterministically shuffle the dataset
        self.pointclouds.sort(key=lambda data: data['model_id'], reverse=False)
        random.Random(2020).shuffle(self.pointclouds)

    def get_ds_statistics(self, dataframe, from_shapenet_v1, from_shapenet_v2): # compute mean and std dev across required dataset
        dataframe_ = dataframe.drop_duplicates(subset=['modelId'])
        pointclouds=[]
        print('Applying global normalization...')
        print('Computing mean and std across all dataset...')
        count=0
        for idx, row in dataframe_.iterrows():
            count +=1
            # read_pcd
            model_id = row["modelId"]
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
            pointclouds.append(pc)
        print('number of processed shapes: ', count)
        pcs = torch.cat(pointclouds, dim=0)
        mean = pcs.reshape(-1, 3).mean(axis=0).reshape(1,3)
        std = pcs.reshape(-1).std(axis=0).reshape(1,1)
        return mean, std

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {k:v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.pointclouds[idx].items()}
        data["idx"] = idx
        if self.transform is not None:
            data = self.transform(data)
        
        return data

class Text2Shape_pairs(Text2Shape):
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
        padding: bool,
        conditional_setup: bool,
        scale_mode: str,
        chinese_distractor: bool = False,
        transforms: List[Callable] = []
    ) -> None:
    
        super().__init__(root, split, categories, from_shapenet_v1, from_shapenet_v2, language_model, lowercase_text, 
                        max_length, padding, conditional_setup, scale_mode, transforms) # initialize parent Text2Shape
        
        self.max_len = max_length
        self.chinese_distractor = chinese_distractor


    def __getitem__(self, idx): # build pairs of clouds
        #data = {k:v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.pointclouds[idx].items()}
        #start = datetime.now()
        target_mid = self.pointclouds[idx]["model_id"]
        target_idx = idx
        dist_mid = target_mid

        if not self.chinese_distractor:
            while dist_mid == target_mid:   # we randomly sample a shape which is different from the current 
                dist_idx = random.randint(0, len(self.pointclouds)-1)
                dist_mid = self.pointclouds[dist_idx]["model_id"]
            
            # build pair
            idxs = [target_idx, dist_idx]
            target = 0
            idxs, target = shuffle_ids(idxs, target)    # shuffle ids
            clouds = torch.stack((self.pointclouds[idxs[0]]["pointcloud"], self.pointclouds[idxs[1]]["pointcloud"]))

        else:   # pick the CORRESPONDING prediction of Towards Implicit...
            #dist_idx = random.randint(0, len(self.pointclouds)-1)
            #dist_mid = self.pointclouds[dist_idx]["model_id"]
            
            dist_mid = target_mid
            text = self.pointclouds[target_idx]["text"]
            text = text.translate(str.maketrans('', '', string.punctuation))
            text = text.replace(" ", "")

            # get target cloud
            target_cloud = self.pointclouds[target_idx]["pointcloud"]
            
            # get distractor cloud from Chinese predictions
            chinese_root = "/media/data2/aamaduzzi/results/towards-implicit/res64/"
 
            for file in os.listdir(chinese_root):
                if "_pc" in str(file) and dist_mid in str(file):
                    #file_fix = file.replace(" ##", "")
                    #file_fix = file_fix.replace(".", " . ")
                    file_fix = file.translate(str.maketrans('', '', string.punctuation))
                    file_fix = file_fix.replace(" ", "")
                    if (text[:8]).lower() in str(file_fix):
                        pcd = o3d.io.read_point_cloud(os.path.join(chinese_root, file))
                        R = pcd.get_rotation_matrix_from_xyz((np.pi / 2, np.pi/2, 0))
                        pcd.rotate(R, center=(0, 0, 0))
                        dist_cloud = torch.Tensor(pcd.points)
                        break
                
            clouds = [target_cloud, dist_cloud]
            target=0
            clouds, target = shuffle_ids(clouds, target)

            clouds = torch.stack((clouds[0], clouds[1]))
            
    
        mean_text_embed = self.pointclouds[target_idx]["text_embed"]
        #mean_text_embed = torch.mean(mean_text_embed, dim=0) # when I compute the mean, if the sentence is small => I have many zeros => mean is small
        
        # compute mean, ignoring zeros of padding
        sum_embed = torch.sum(mean_text_embed, dim=1)

        seq_len = torch.count_nonzero(sum_embed)
        mean_text_embed = mean_text_embed[:seq_len,:]
        
        mean_text_embed = torch.mean(mean_text_embed, dim=0)
        
        text = self.pointclouds[target_idx]["text"]

        data = {"clouds": clouds,
                "target": target,
                "mean_text_embed": mean_text_embed,
                "text": text}
        
        #if idx%20==0:
        #    visualize_data_sample(clouds, target, text, f"sample_{datetime.now()}.png")   # RED:target, BLUE:distractor

        return data