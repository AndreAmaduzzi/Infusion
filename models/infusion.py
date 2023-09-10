from typing import Mapping
import torch
from torch import nn, Tensor
from models.pvd import *

class Infusion(nn.Module):
    def __init__(self, opt):
        super().__init__()
        betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
        self.pvd = PVD(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type, opt.use_concat, opt.max_text_len)

    def get_loss(self,
                cloud: Tensor,
                noises_batch: Tensor,
                text_embed: Tensor,
                text: list,
                epoch: int,
                save_matrices: bool = False,
                concat: bool = False,
                context_dim: int = 77,
                ):
                
        loss, t = self.pvd.get_loss_iter(data=cloud, noises=noises_batch, condition=text_embed, text=text, epoch=epoch, save_matrices=save_matrices, concat=concat, context_dim=context_dim)

        return loss, t
    
    def get_clouds(self,
                text_embed: Tensor,             # a batch of text embeds
                test_shapes: Tensor,            # a batch of shapes
                concat: bool = False,
                context_dim: int = 77):           
        
        clouds = self.pvd.gen_samples(self.new_x_chain(test_shapes, text_embed.shape[0]).shape, test_shapes.device, clip_denoised=False, condition=text_embed, concat=concat, context_dim=context_dim)                    # we get clouds for a batch of texts
        return clouds

    def get_cloud_traj(self,
                text_embed: Tensor,             # a single text embedding 
                test_shapes: Tensor,            # a single shape
                concat: bool = False,
                context_dim: int = 77):                     
        
        cloud_traj = self.pvd.gen_sample_traj(self.new_x_chain(test_shapes, 1).shape, test_shapes.device, freq=40, clip_denoised=False, condition=text_embed, concat=concat, context_dim=context_dim) # we get trajectory only for 1 shape
        return cloud_traj
        

    def multi_gpu_wrapper(self, f):
        self.pvd.model = f(self.pvd.model)

    def new_x_chain(self, x, num_chain):
        return torch.randn(num_chain, *x.shape[1:], device=x.device)
