from curses.textpad import Textbox
from typing import List, Callable
from pathlib import Path
from torch.utils.data import Dataset
from pycarus.transforms.var import Compose
from pycarus.geometry.pcd import get_tensor_pcd_from_o3d, farthest_point_sampling
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

def visualize_data_sample(pointclouds, target, img_title, path, idx):
    n_clouds = len(pointclouds)
    fig = plt.figure(figsize=(20,20))
    plt.title(label=img_title, fontsize=25)
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
        chatgpt_prompts: bool=False,
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
            if chatgpt_prompts:
                annotations_path = self.root / "annotations_chatgpt" / "from_shapenet_v2" / f"{self.split}_{self.categories}.csv"
                train_path = self.root / "annotations_chatgpt" / "from_shapenet_v2" / f"train_{self.categories}.csv"
            else:
                annotations_path = self.root / "annotations" / "from_shapenet_v2" / f"{self.split}_{self.categories}.csv"
                train_path = self.root / "annotations" / "from_shapenet_v2" / f"train_{self.categories}.csv"
        elif from_shapenet_v1:
            if chatgpt_prompts:
                annotations_path = self.root / "annotations_chatgpt" / "from_shapenet_v1" / f"{self.split}_{self.categories}.csv"
                train_path = self.root / "annotations_chatgpt" / "from_shapenet_v1" / f"train_{self.categories}.csv"
            else:
                annotations_path = self.root / "annotations" / "from_shapenet_v1" / f"{self.split}_{self.categories}.csv"
                train_path = self.root / "annotations" / "from_shapenet_v1" / f"train_{self.categories}.csv"                
        elif chatgpt_prompts:
            annotations_path = self.root / "annotations_chatgpt" / "from_text2shape" / f"{self.split}_{self.categories}.csv"
            train_path = self.root / "annotations_chatgpt" / "from_text2shape" / f"train_{self.categories}.csv"
        else:
            annotations_path = self.root / "annotations" / "from_text2shape" / f"{self.split}_{self.categories}.csv"
            train_path = self.root / "annotations" / "from_text2shape" / f"train_{self.categories}.csv"


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

        count_trunc = 0
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
                model_path = str(self.root / "shapes" / "text2shape_rgb" / f"{model_id}.ply")
                pc = o3d.io.read_point_cloud(model_path)
                pc = get_tensor_pcd_from_o3d(pc)
            #scaling the pcd (from diffusion-pointcloud code)
            if self.scale_mode == "global_unit":
                shift = global_mean
                scale = global_std
            elif self.scale_mode == 'shape_unit':
                    shift = pc[:,:3].mean(dim=0).reshape(1, 3)
                    scale = pc[:,:3].flatten().std().reshape(1, 1)
            elif self.scale_mode == 'shape_half':
                    shift = pc[:,:3].mean(dim=0).reshape(1, 3)
                    scale = pc[:,:3].flatten().std().reshape(1, 1) / (0.5)
            elif self.scale_mode == 'shape_34':
                    shift = pc[:,:3].mean(dim=0).reshape(1, 3)
                    scale = pc[:,:3].flatten().std().reshape(1, 1) / (0.75)
            elif self.scale_mode == 'shape_bbox':
                    pc_max, _ = pc[:,:3].max(dim=0, keepdim=True) # (1, 3)
                    pc_min, _ = pc[:,:3].min(dim=0, keepdim=True) # (1, 3)
                    shift = ((pc_min + pc_max) / 2).view(1, 3)
                    scale = (pc_max - pc_min).max().reshape(1, 1) / 2
            else:
                    shift = torch.zeros([1, 3])
                    scale = torch.ones([1, 1])
            print(pc.shape)
            pc[:,:3] = (pc[:,:3] - shift) / scale

            tensor_name = row["tensor"]
            if lowercase_text:
                if padding:
                    if chatgpt_prompts:
                        text_embed_path = self.root / "text_embeds_chatgpt" / language_model / "lowercase" /  "padding" / f"{tensor_name}"
                    else:
                        text_embed_path = self.root / "text_embeds" / language_model / "lowercase" /  "padding" / f"{tensor_name}"
                elif chatgpt_prompts:
                    text_embed_path = self.root / "text_embeds_chatgpt" / language_model / "lowercase" / f"{tensor_name}"
                else:
                    text_embed_path = self.root / "text_embeds" / language_model / "lowercase" / f"{tensor_name}"
            else:
                if padding:
                    if chatgpt_prompts:
                        text_embed_path = self.root / "text_embeds_chatgpt" / language_model / "std" /  "padding" / f"{tensor_name}"
                    text_embed_path = self.root / "text_embeds" / language_model / "std" /  "padding" / f"{tensor_name}"
                elif chatgpt_prompts:
                    text_embed_path = self.root / "text_embeds_chatgpt" / language_model / "std" / f"{tensor_name}"
                else:
                    text_embed_path = self.root / "text_embeds" / language_model / "std" / f"{tensor_name}"
            
            if padding:
                text_embed = torch.load(text_embed_path, map_location='cpu')
            else:
                text_embed = torch.load(text_embed_path, map_location='cpu')
                if text_embed.shape[0] > max_length:
                    count_trunc += 1
                    text_embed = text_embed[:max_length]       # truncate to max_length => [max_length, 1024]
                
                
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
        print(count_trunc ,' truncated text embeds')

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


class Text2Shape_subset_mid(Text2Shape):
    def __init__(
        self,
        root: Path,
        chatgpt_prompts: bool=False,
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
    
        '''
        Class implementing a subset of Text2Shape dataset, where:
        - we provide a single entry for each model_id
        - the entry (cloud, text embedding, text...) is randomly sampled from the entries of that specific model_id
        - dataset size of training dataset with tables and chairs: 12054 entries
        
        '''

        super().__init__(root, chatgpt_prompts, split, categories, from_shapenet_v1, from_shapenet_v2, language_model, lowercase_text, 
                        max_length, padding, conditional_setup=True, scale_mode=scale_mode, transforms=transforms) # initialize parent Text2Shape

        # get unique model_ids
        m_ids = [shape["model_id"] for shape in self.pointclouds]
        self.m_ids = list(set(m_ids))

    
    def __getitem__(self, idx): 
        m_id = self.m_ids[idx]
        # find number of occurences of the current shape
        count=0
        for cloud in self.pointclouds:
            if cloud["model_id"]==m_id:
                count+=1 
        
        # pick a random index
        rand_idx = random.randrange(0, count)

        # return the corresponding entry of the dataset
        count=0
        for cloud in self.pointclouds:
            if cloud["model_id"]==m_id:
                if count==rand_idx:
                    cloud["idx"] = idx
                    return cloud
                else:
                    count+=1 

    def __len__(self):
        return len(self.m_ids)
    

class Text2Shape_pairs(Text2Shape):
    def __init__(
        self,
        root: Path,
        chatgpt_prompts: bool=False,
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
        shape_gt: str,
        shape_dist: str,
        transforms: List[Callable] = []
    ) -> None:
    
        super().__init__(root, chatgpt_prompts, split, categories, from_shapenet_v1, from_shapenet_v2, language_model, lowercase_text, 
                        max_length, padding, conditional_setup, scale_mode, transforms) # initialize parent Text2Shape
        
        self.max_len = max_length
        self.shape_gt = shape_gt
        self.shape_dist = shape_dist


    def __getitem__(self, idx): # build pairs of clouds
        target_mid = self.pointclouds[idx]["model_id"]
        target_idx = idx
        dist_mid = target_mid

        if self.shape_gt is None:            # when shape_1 and shape_2 are None => we build pairs of GT T2S and random T2S
            assert self.shape_dist is None
            while dist_mid == target_mid:   # we randomly sample a shape which is different from the current 
                dist_idx = random.randint(0, len(self.pointclouds)-1)
                dist_mid = self.pointclouds[dist_idx]["model_id"]
            
            # build pair
            idxs = [target_idx, dist_idx]
            target = 0
            idxs, target = shuffle_ids(idxs, target)    # shuffle ids
            clouds = torch.stack((self.pointclouds[idxs[0]]["pointcloud"], self.pointclouds[idxs[1]]["pointcloud"]))
            cates = [self.pointclouds[idxs[0]]["cate"], self.pointclouds[idxs[1]]["cate"]]

            # replace name of objects with corresponding indices of ShapeNetPart
            class_labels = []
            for cate in cates:
                if cate=="Chair":
                    class_labels.append(4)
                elif cate=="Table":
                    class_labels.append(15)

            class_labels = torch.Tensor(class_labels)

        else:
            # build pair with GT and DIST shapes
            target_clouds =  np.load(os.path.join(self.shape_gt, 'out.npy'))
            target_clouds = torch.from_numpy(target_clouds)
            target_cloud = target_clouds[target_idx]

            dist_clouds =  np.load(os.path.join(self.shape_dist, 'out.npy'))
            dist_clouds = torch.from_numpy(dist_clouds)
            dist_cloud = dist_clouds[target_idx]

            clouds = [target_cloud, dist_cloud]

            target=0
            clouds, target = shuffle_ids(clouds, target)

            # if one shape has only 2025 points => sample 2025 points from the other cloud.
            # TODO: you can also sample 2048 pts from 2025 pts => investigate this...
            if clouds[0].shape[0] > clouds[1].shape[0]:
                clouds[0] = farthest_point_sampling(clouds[0], clouds[1].shape[0])
            elif clouds[0].shape[0] < clouds[1].shape[0]:
                clouds[1] = farthest_point_sampling(clouds[1], clouds[0].shape[0])

            clouds = torch.stack((clouds[0], clouds[1]))

            class_labels = torch.Tensor()

        mean_text_embed = self.pointclouds[target_idx]["text_embed"]
        
        # compute mean, ignoring zeros of padding
        sum_embed = torch.sum(mean_text_embed, dim=1)

        seq_len = torch.count_nonzero(sum_embed)
        mean_text_embed = mean_text_embed[:seq_len,:]
        
        mean_text_embed = torch.mean(mean_text_embed, dim=0)
        
        text = self.pointclouds[target_idx]["text"]

        data = {"clouds": clouds,
                "target": target,
                "mean_text_embed": mean_text_embed,
                "text_embed": self.pointclouds[target_idx]["text_embed"],
                "text": text,
                "class_labels": class_labels,                               # TODO: check class_labels
                "idx": target_idx}
        
        #if idx%1000==0:
        #    visualize_data_sample(clouds, target, text, f"{self.method}_{datetime.now()}.png", target_idx)   # RED:target, BLUE:distractor

        return data